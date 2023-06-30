import argparse
import numpy as np
import pickle
import time
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

import transformer.Constants as Constants
import Utils as Utils

from preprocess.Dataset import get_dataloader
from transformer.Models import Transformer, smurf_thp
from tqdm import tqdm

# import wandb
# wandb.init(project="thp")


def prepare_dataloader(opt):
    """ Load data and prepare dataloader. """

    def load_data(name, dict_name):
        """ Normal load data. """
        with open(name, 'rb') as f:
            data = pickle.load(f, encoding='latin-1')
            num_types = data['dim_process']
            data = data[dict_name]
            return data, int(num_types)

    print('[Info] Loading train data...')
    train_data, num_types = load_data(opt.data + 'train.pkl', 'train')
    
    # get naive sample & compute statistic for normalization
    time_flat = []
    for i in train_data:
        time_flat+=[elem['time_since_last_event'] for elem in i]
    time_flat = torch.tensor(time_flat, device=opt.device)
    rand_idx = torch.randint(0,len(time_flat),(1,opt.n_samples)).squeeze(0)
    train_sample = time_flat[rand_idx]

    if opt.normalize == 'normal':
        mean_data = time_flat.mean().item()
        train_sample /= mean_data
    if opt.normalize == 'log':
        train_sample = train_sample.log()
        mean_data = time_flat.log().mean().item()
        var_data = time_flat.log().std().item()
        train_sample = (train_sample-mean_data)/var_data

    print('[Info] Loading dev data...')
    dev_data, _ = load_data(opt.data + 'dev.pkl', 'dev')
    print('[Info] Loading test data...')
    test_data, _ = load_data(opt.data + 'test.pkl', 'test')

    # Load three datasets
    opt.max_len = 0
    trainloader = get_dataloader(train_data, opt, shuffle=True, split='train')
    devloader = get_dataloader(dev_data, opt, shuffle=False, split='dev')
    testloader = get_dataloader(test_data, opt, shuffle=False, split='test')
    print('time_quantile:', opt.time_quantile)
    print('time_mean:', opt.time_mean)
    print('time_std:', opt.time_std)
    print('time_min:', opt.time_min)
    print('time_max:', opt.time_max)

    return trainloader, devloader, testloader, num_types, train_sample
    


def train_epoch(model, training_data, optimizer, pred_loss_func, opt):
    """ Epoch operation in training phase. """

    model.train()

    total_event_ll = 0  # cumulative event log-likelihood
    total_event_rate = 0  # cumulative number of correct prediction
    total_num_event = 0  # number of total events
    total_num_pred = 0  # number of predictions
    for batch in tqdm(training_data, mininterval=2, desc='  - (Training)   ', leave=False):
        """ prepare data """
        event_time, time_gap, event_type = map(lambda x: x.to(opt.device), batch)

        """ forward """
        optimizer.zero_grad()
        # forward
        enc_out, prediction = model(event_type, event_time, time_gap)
        # compute the loss
        loss = model.compute_loss(enc_out, event_time, time_gap, event_type, prediction, pred_loss_func)
        # compute accuracy of predicted type
        truth = event_type[:, 1:] - 1
        if prediction is not None:
            pred_type = torch.max(prediction, dim=-1)[1]
            if prediction.ndim==4:
                pred_type = torch.mode(pred_type,2)[0]
            pred_num_event = torch.sum(pred_type == truth)
        else:
            pred_num_event = torch.tensor([0])

        """ update parameters """        
        loss.backward()
        optimizer.step()

        """ note keeping """
        total_event_ll += loss.item()
        total_event_rate += pred_num_event.item()
        total_num_event += event_type.ne(Constants.PAD).sum().item()
        # we do not predict the first event
        total_num_pred += event_type.ne(Constants.PAD).sum().item() - event_time.shape[0]

    return total_event_ll / total_num_event, total_event_rate / total_num_pred


def eval_epoch(model, validation_data, pred_loss_func, eval_langevin, train_sample, opt, langevin_init_step = 0, sample_init = None):
    """ Epoch operation in evaluation phase. """
    
    model.eval()

    total_event_ll = 0  # cumulative event log-likelihood
    total_event_rate = 0  # cumulative number of correct prediction
    total_num_event = 0  # number of total events
    total_num_pred = 0  # number of predictions
    total_coverage_single = torch.zeros(len(opt.eval_quantile))
    total_intlen = torch.zeros(len(opt.eval_quantile))
    total_crps = 0 

    start = time.time()

    """ Compute the loss and accuracy first -- no sampling """
    for idx, batch in enumerate(tqdm(validation_data, mininterval=2, desc='  - (Evaluating) ', leave=False)):
        # prepare data 
        event_time, time_gap, event_type = map(lambda x: x.to(opt.device), batch)
        # forward
        enc_out, prediction = model(event_type, event_time, time_gap)
        # compute the loss
        loss = model.compute_loss(enc_out, event_time, time_gap, event_type, prediction, pred_loss_func)
        # compute accuracy of predicted type
        truth = event_type[:, 1:] - 1
        if prediction is not None:
            pred_type = torch.max(prediction, dim=-1)[1]
            if prediction.ndim==4:
                pred_type = torch.mode(pred_type,2)[0]
            pred_num_event = torch.sum(pred_type == truth)
        else:
            pred_num_event = torch.tensor([0])
        # note keeping
        total_event_ll += loss.item()
        total_event_rate += pred_num_event.item()
        total_num_event += event_type.ne(Constants.PAD).sum().item()
        total_num_pred += event_type.ne(Constants.PAD).sum().item() - event_time.shape[0]
            
    print('  - (Testing) loss: {ll: 8.5f}, accuracy: {type: 8.5f}, elapse: {elapse:3.3f} min'
        .format(ll=total_event_ll / total_num_event, type=total_event_rate / total_num_pred, elapse=(time.time() - start) / 60))
    
    if not eval_langevin:
        return total_event_ll / total_num_event, total_event_rate / total_num_pred

    best_results = None
    # record results in every langevin step
    results_all = {'n_steps':[],'cs':[], 'accuracy':[], 'crps': [], 'intlen': [], 'coverage': []}
    if eval_langevin:
        # Langevin dynamics !!!!
        # Save samples every n_save_steps
        if True:
            cur_step = langevin_init_step
            last_sample = sample_init
            opt.is_last = False

            while cur_step < (opt.n_steps + langevin_init_step):
                if (cur_step + opt.n_save_steps) >= (opt.n_steps + langevin_init_step):
                    opt.is_last = True
                total_coverage_single = torch.zeros(len(opt.eval_quantile))
                total_intlen = torch.zeros(len(opt.eval_quantile))
                total_correct_list = torch.zeros(int(len(opt.eval_quantile)/2))
                total_num_list = torch.zeros(int(len(opt.eval_quantile)/2))
                total_crps = 0 
                total_corr_type = 0

                t_list = [] # sample time list
                gt_t_list = [] # gt time list
                gt_type_list = [] # gt type list
                pred_type_list = [] # sample type list
                    
                for idx, batch in enumerate(tqdm(validation_data, mininterval=2, desc='  - (Sampling) ', leave=False)):
                    """ prepare data """
                    event_time, time_gap, event_type = map(lambda x: x.to(opt.device), batch)

                    # forward
                    enc_out, prediction = model(event_type, event_time, time_gap)

                    # langevin dynamics only proceed n_save_steps (for saving purpose), not n_steps
                    t_sample, gt_t, type_sample = Utils.predict_langevin(model, event_time, time_gap, event_type, prediction, opt, (last_sample[0][idx],last_sample[1][idx]) if last_sample is not None else None)
                    coverage_single, intlen, crps, corr_type = Utils.evaluate_samples(t_sample, gt_t, type_sample, event_type, opt)
                    
                    total_coverage_single += coverage_single
                    total_intlen += intlen
                    total_crps += crps.item()
                    total_corr_type += corr_type.item()

                    t_list.append(t_sample)
                    # gt_type_list.append(event_type[:,1:]) # XXXXX: save event type
                    pred_type_list.append(type_sample)
                    # gt_t_list.append(gt_t)
                    
                cur_step += opt.n_save_steps
                last_sample = (t_list,pred_type_list)

                # note keeping
                start = time.time()
                total_coverage_single /= total_num_pred
                total_intlen /= total_num_pred
                total_crps /= total_num_pred
                total_corr_type /= total_num_pred
                    
                # compute calibration score from coverage
                big_idx_single = int(len(opt.eval_quantile)/2)
                cs_single = np.sqrt(((total_coverage_single[-4:] - opt.eval_quantile[-4:])**2).mean())
                cs_single_big = np.sqrt(((total_coverage_single[big_idx_single:] - opt.eval_quantile[big_idx_single:])**2).mean())
                    
                # XXXXX: save event type
                print('  - (Sampling) Langevin steps {cur_step}: loss: {ll: 8.5f}, '
                        'Accuracy: {type: 8.5f}, Single-bound calibration score: {cs1: 8.5f}/{csb1: 8.5f}, '
                        'CRPS: {crps: 8.5f}, Interval Length: {intlen: 8.5f}, elapse: {elapse:3.3f} min'
                        .format(cur_step=cur_step, ll=total_event_ll / total_num_event, type=total_corr_type, 
                        cs1=cs_single, csb1=cs_single_big, crps=total_crps, intlen=total_intlen[big_idx_single], elapse=(time.time() - start) / 60))
                print('coverage_single: ', total_coverage_single)
                print('Interval Length: ', total_intlen)
                results = {'n_steps':cur_step,'cs':(cs_single.item(),cs_single_big.item()), 
                    'accuracy':total_corr_type, 'crps': total_crps, 'intlen': total_intlen, 
                    'coverage': (total_coverage_single)}
                results_all['n_steps'].append(cur_step)
                results_all['cs'].append(cs_single_big.item())
                results_all['accuracy'].append(total_corr_type)
                results_all['crps'].append(total_crps)
                results_all['intlen'].append(total_intlen[big_idx_single])
                results_all['coverage'].append(total_coverage_single[big_idx_single])
                if best_results is not None:
                    if results['cs'][1]<best_results['cs'][1]:
                        best_results=results
                else:
                    best_results=results

        # Record the results as csv
        if opt.save_result is not None:
            results_all_pd = pd.DataFrame([results_all[i] for i in results_all.keys()]).transpose()
            results_all_pd.columns = results_all.keys()
            results_all_pd.to_csv(opt.save_result+f"_results_all.csv")
                        # print('Result saved to', opt.save_result+f".best_step.pth !!!!!")  
        
        print('  - (Best Langevin results) : loss: {ll: 8.3f}, Accuracy: {type: 8.3f}, Calibration score: {cs1: 8.3f}/{csb1: 8.3f}/{cs2: 8.3f}/{csb2: 8.3f}, '
            'CRPS: {crps: 8.3f}, Interval Length: {intlen: 8.3f}, elapse: {elapse:3.3f} min'
            .format(ll=total_event_ll / total_num_event, type=best_results['accuracy'], 
            cs1=best_results['cs'][0], csb1=best_results['cs'][1],
            crps=best_results['crps'], intlen=best_results['intlen'][big_idx_single], elapse=(time.time() - start) / 60))
    return best_results


def train(config=None):
    if True:

        # print('[Info] parameters: {}'.format(config))

        """ prepare dataloader """
        trainloader, devloader, testloader, num_types, train_sample = prepare_dataloader(config)
        config.num_types = num_types

        """ prepare model """
        if config.model == 'thp':
            model = Transformer(num_types, config)
        elif config.model == 'smurf_thp':
            model = smurf_thp(num_types, config)
        model.to(config.device)

        """ optimizer and scheduler """
        if config.optimizer == 'adam':
            optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()),
                                config.lr, betas=(0.9, 0.999), eps=1e-5)
        else:
            optimizer = optim.SGD(filter(lambda x: x.requires_grad, model.parameters()),
                                config.lr, momentum = 0.9, weight_decay=5e-4)
        
        if config.scheduler == 'cosLR':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 64, verbose=True)
        elif config.scheduler == 'reduce':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', verbose=True)

        """ prediction loss function, either cross entropy or label smoothing """
        pred_loss_func = Utils.LabelSmoothingLoss(config.smooth, num_types, ignore_index=-1)

        """ number of parameters """
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('[Info] Number of parameters: {}'.format(num_params))

        """ Start training. """
        best_loss = 1e5 # record the loss to save the best model
        for epoch_i in range(config.epoch):
            epoch = epoch_i + 1
            print('[ Epoch', epoch, ']')

            start = time.time()
            train_loss, train_acc = train_epoch(model, trainloader, optimizer, pred_loss_func, config)
            print('  - (Training)    loss: {ll: 8.5f}, accuracy: {type: 8.5f}, elapse: {elapse:3.3f} min'
                .format(ll=train_loss, type=train_acc, elapse=(time.time() - start) / 60))
            start = time.time()
            valid_loss, valid_acc = eval_epoch(model, devloader, pred_loss_func, False, train_sample, config)
            print('  - (Validating)    loss: {ll: 8.5f}, accuracy: {type: 8.5f}, elapse: {elapse:3.3f} min'
                .format(ll=valid_loss, type=valid_acc, elapse=(time.time() - start) / 60))
            
            ###save the best model
            if valid_loss <= best_loss:
                if config.save_path is not None:
                    print('!!! Saving best model to '+config.save_path+config.save_name+'.pth!!!')
                    best_loss = valid_loss
                    torch.save({'model':model.state_dict(),'best_loss':best_loss, 'epoch':epoch_i}, config.save_path+config.save_name+'.pth')
            
            if  epoch >= config.eval_epoch:
                start = time.time()
                test_loss, test_acc = eval_epoch(model, testloader, pred_loss_func, True, config)
                print('  - (Testing)   loss: {ll: 8.5f}, accuracy: {type: 8.5f}, elapse: {elapse:3.3f} min'
                    .format(ll=test_loss, type=test_acc, elapse=(time.time() - start) / 60))
            
            if config.scheduler == 'reduce':
                scheduler.step(valid_loss)
            else:
                scheduler.step()

def eval(config=None):

    assert config.load_path_name is not None, "No evaluation model provided!!!"

    if True:
        # default device is CUDA
        config.device = torch.device('cuda')

        # print('[Info] parameters: {}'.format(config))
        print('======== add_noise: {}, parametrize: {}, model: {}, seed: {} ======='.format(config.add_noise,config.parametrize,config.model,config.seed))

        """ prepare dataloader """
        _, devloader, testloader, num_types, train_sample = prepare_dataloader(config)
        config.num_types = num_types

        """ prepare model """
        if config.model == 'thp':
            model = Transformer(num_types, config)
        elif config.model == 'smurf_thp':
            model = smurf_thp(num_types, config)
        model.to(config.device)

        state_dict = torch.load(config.load_path_name)
        model.load_state_dict(state_dict['model'], strict=True)
        print('We get validation loss:',state_dict['best_loss'], 'in the training process!')
        print('Start evaluating:')

        """ prediction loss function, either cross entropy or label smoothing """
        pred_loss_func = Utils.LabelSmoothingLoss(config.smooth, num_types, ignore_index=-1)

        """ number of parameters """
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('[Info] Number of parameters: {}'.format(num_params))

        """sample initialization"""
        if config.sample_init is not None:
            sample_saved = torch.load(config.sample_init+'.pth')             
            sample_init =  sample_saved['sample']
            langevin_init_steps = sample_saved['n_steps']
            
            print(f"Init sample {config.sample_init}.pth loaded!!!!!")
        else:
            sample_init = None
            langevin_init_steps = 0

        """ Start evaluating. """
        start = time.time()
        best_results = eval_epoch(model, testloader, pred_loss_func, True, train_sample, config, langevin_init_steps, sample_init)
        # print('  - (Testing)   loss: {ll: 8.5f}, accuracy: {type: 8.5f}, elapse: {elapse:3.3f} min'
        #     .format(ll=test_loss, type=test_acc, elapse=(time.time() - start) / 60))
        return best_results


def main():
    """ Main function. """

    parser = argparse.ArgumentParser()

    #### data option
    parser.add_argument('-data', required=True)
    parser.add_argument('-normalize', type=str, default='None')
    parser.add_argument('-seed', type=int, default=2023)

    #### training option
    parser.add_argument('-epoch', type=int, default=50)
    parser.add_argument('-eval_epoch', type=int, default=1)
    parser.add_argument('-batch_size', type=int, default=16)
    parser.add_argument('-lr', type=float, default=1e-4)
    parser.add_argument('-optimizer', type=str, default='adam')
    parser.add_argument('-scheduler', type=str,  choices=['cosLR','reduce'], default='cosLR')
    parser.add_argument('-loss_lambda', type=float, default=1.0)

    #### model type
    parser.add_argument('-model', type=str, choices=['thp', 'smurf_thp'], default='smurf_thp')
    parser.add_argument('-decoder', type=str, choices=['encode'], default='encode')
    parser.add_argument('-parametrize', type=str, choices=['intensity','score'], default='intensity')

    # JSM type loss
    parser.add_argument('-type_loss', type=str, choices=['CE'], default='CE')

    #### model hyperparameter
    parser.add_argument('-d_model', type=int, default=64)
    parser.add_argument('-d_rnn', type=int, default=256)
    parser.add_argument('-d_inner_hid', type=int, default=128)
    parser.add_argument('-d_k', type=int, default=16)
    parser.add_argument('-d_v', type=int, default=16)
    parser.add_argument('-n_head', type=int, default=4)
    parser.add_argument('-n_layers', type=int, default=4)
    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-smooth', type=float, default=0.0)


    parser.add_argument('-thp_no_predictor', action='store_true')
    
    #### save option and eval option
    parser.add_argument('-save_path', type=str, default=None)
    parser.add_argument('-save_name', type=str, default=None)

    parser.add_argument('-just_eval', action='store_true')
    parser.add_argument('-load_path_name', type=str, default=None)
    parser.add_argument('-save_result', type=str, default=None)
    parser.add_argument('-sample_init', type=str, default=None)

    #### score matching sampling option
    parser.add_argument('-langevin_step', type=float, default=5e-2)
    parser.add_argument('-n_steps', type=int, default=100)
    parser.add_argument('-n_save_steps', type=int, default=100)
    parser.add_argument('-n_samples', type=int, default=100)
    parser.add_argument('-sampling_method', type=str, choices=['normal','truncated'], default='normal')
    parser.add_argument('-denoise_step', type=int, default=1)
    parser.add_argument('-train_size', type=float, default=1)

    parser.add_argument('-eval_quantile', type=float, default=-1)
    parser.add_argument('-eval_quantile_step', type=float, default=0.1)
    parser.add_argument('-conditional_sampling', action='store_true')

    #### score matching with noise; noise option
    parser.add_argument('-add_noise', type=str, choices=['None', 'denoise'], default=None)
    parser.add_argument('-noise_type', type=str, choices=['normal'], default='normal')

    parser.add_argument('-var_noise', type=float, default=1e-2)
    parser.add_argument('-var_noise_type', type=float, default=1)
    parser.add_argument('-num_noise', type=int, default=100)

    opt = parser.parse_args()
    # default device is CUDA
    opt.device = torch.device('cuda')
    opt.denoise_step = bool(opt.denoise_step)
    if opt.eval_quantile == -1:
        opt.eval_quantile = np.arange(opt.eval_quantile_step,1,opt.eval_quantile_step)
    else:
        NotImplementedError, "Please set eval_quantile to -1"

    # set seed  
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    torch.set_printoptions(precision=4)
    # opt = wandb.config.update(opt)
    print('[Info] parameters: {}'.format(opt))
    if not opt.just_eval:
        train(opt)
    else:
        best_results = eval(opt)


if __name__ == '__main__':
    main()
