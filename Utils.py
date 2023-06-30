import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from transformer.Models import get_non_pad_mask


def softplus(x, beta):
    # hard thresholding at 20
    temp = beta * x
    temp[temp > 20] = 20
    return 1.0 / beta * torch.log(1 + torch.exp(temp))

def compute_event(model, event_time, time_gap, event_type, non_pad_mask):
    
    type_mask = torch.zeros([*event_type.size(), model.num_types], device=event_time.device)
    for i in range(model.num_types):
        type_mask[:, :, i] = (event_type == i + 1).bool().to(event_time.device)

    all_lambda = model.get_intensity(time_gap, event_type, event_time, time_gap, non_pad_mask).squeeze(2)

    event = torch.sum(all_lambda * type_mask[:, 1:, :], dim=2)
    event += math.pow(10, -9)
    event.masked_fill_(~non_pad_mask[:,1:].squeeze(2).bool(), 1.0)
    result = torch.log(event+1e-10) * non_pad_mask[:,1:].squeeze(2)
    return result

def compute_integral_unbiased(model, event_time, time_gap, event_type, non_pad_mask):
    """ Log-likelihood of non-events, using Monte Carlo integration. """

    num_samples = 100
    if model.normalize == 'log':
        time_low = min(-1.0,time_gap.min()-1.0)
    else:
        time_low = 0
    temp_time = (time_gap.unsqueeze(2) - time_low) * \
                torch.rand([*time_gap.size(), num_samples], device=event_time.device) + time_low
    
    if model.num_types >= 100:
        all_lambda = None
        for i in range(num_samples):
            lambda_i = model.get_intensity(temp_time[:,:,i:i+1], event_type, event_time, time_gap, non_pad_mask)
            if all_lambda == None:
                all_lambda = torch.sum(lambda_i, dim=(2,3)) 
            else:
                all_lambda += torch.sum(lambda_i, dim=(2,3)) 
        all_lambda /= num_samples
    else:
        all_lambda = model.get_intensity(temp_time, event_type, event_time, time_gap, non_pad_mask)
        all_lambda = torch.sum(all_lambda, dim=(2,3)) / num_samples

    unbiased_integral = all_lambda * (time_gap - time_low) * non_pad_mask.squeeze(-1)[:,1:]
    return unbiased_integral


def log_likelihood(model, event_time, time_gap, event_type):
    """ Log-likelihood of sequence. """

    non_pad_mask = get_non_pad_mask(event_type)

    # event log-likelihood
    event_ll = compute_event(model, event_time, time_gap, event_type, non_pad_mask)
    event_ll = torch.sum(event_ll, dim=-1)

    # non-event log-likelihood, either numerical integration or MC integration
    non_event_ll = compute_integral_unbiased(model, event_time, time_gap, event_type, non_pad_mask)
    non_event_ll = torch.sum(non_event_ll, dim=-1)

    return event_ll, non_event_ll

def type_loss(prediction, types, loss_func, logit=True):
    """ Event prediction loss, cross entropy or label smoothing. """
    # convert [1,2,3] based types to [0,1,2]; also convert padding events to -1
    truth = types[:, 1:] - 1
    if prediction.ndim==4:
        truth=truth.unsqueeze(2)

    # compute cross entropy loss
    loss = loss_func(prediction, truth, logit)

    if prediction.ndim==4:
        loss = loss.mean(2)
    loss = torch.sum(loss)
    return loss


def predict_langevin(model, event_time, time_gap, event_type, prediction, opt, sample_init):
    # event_time & event_type (batch*len)
    # prediction (batch*len-1*num_types)

    if opt.model == 'smurf_thp':
        return predict_langevin_smurf(model, event_time, time_gap, event_type, prediction, opt, sample_init)
    else:
        return predict_langevin_baseline(model, event_time, time_gap, event_type, prediction, opt, sample_init)

def evaluate_samples(t_sample, gt_t, type_sample, event_type, opt):

    if opt.normalize == 'log':
        gt_t = torch.exp(gt_t*opt.var_log_data+opt.mean_log_data)
        t_sample = torch.exp(t_sample*opt.var_log_data+opt.mean_log_data)
            
    non_pad_mask = get_non_pad_mask(event_type) 

    eval_quantile = opt.eval_quantile
    q = torch.tensor([*eval_quantile,0.5],  dtype = torch.float32,device=event_type.device)
    t_sample_q = torch.quantile(t_sample, q, dim=-1) * non_pad_mask[:, 1:].squeeze(2) # quantile*batch*len-1
    t_sample_mean = t_sample.mean(-1) * non_pad_mask[:, 1:].squeeze(2)

    # compute both coverage 
    coverage_single = torch.sum((gt_t.unsqueeze(0) < t_sample_q[:-1])*non_pad_mask[:,1:].squeeze(2),(1,2))
    coverage_double = torch.tensor([torch.sum(((gt_t > t_sample_q[i])&(gt_t < t_sample_q[-i-2]))*non_pad_mask[:,1:].squeeze(2)) for i in range(int(len(eval_quantile)/2))])
    # compute interval length
    q_idx = int(len(eval_quantile)/2)
    intlen = t_sample_q.clamp(min=0.).sum((1,2))[:-1]
    # compute crps
    num_samples = t_sample.size(2)
    crps = torch.abs(t_sample-gt_t.unsqueeze(2)).mean(2) - torch.abs(t_sample.unsqueeze(3)-t_sample.unsqueeze(2)).sum((2,3))/2/num_samples**2
    crps = (crps * non_pad_mask[:, 1:].squeeze(2)).sum()

    # compute corr_type and ece
    type_sample, mode, _ = type_sample
    truth = event_type[:, 1:] - 1
    ece_loss = ECELoss(n_bins=int(len(opt.eval_quantile)/2),opt=opt)
    ece, corr_type, accuracy_list = ece_loss(type_sample,truth,non_pad_mask.squeeze(-1), mode)
    # print(len(accuracy_list))

    return coverage_single.cpu(), coverage_double.cpu(), intlen.cpu(), crps, corr_type, ece, accuracy_list

def predict_langevin_smurf(model, event_time, time_gap, event_type, prediction, opt, sample_init):

    non_pad_mask = get_non_pad_mask(event_type)    
    e = opt.langevin_step
    n_samples = opt.n_samples
    diff_time = time_gap * non_pad_mask[:, 1:].squeeze(2)

    if sample_init is None:
        t = torch.rand([*diff_time.size(), n_samples], device=event_type.device) * (opt.time_95 - opt.time_05) + opt.time_05
    else:
        assert sample_init[0].ndim == 3, 'Wrong sample init size!!!!!!'
        t, type_sample = sample_init

    
    sqrt_e = math.sqrt(e)
    if opt.sampling_method == 'truncated':
        for _ in range(opt.n_save_steps):
            z = torch.randn_like(t)
            t_new = t + 0.5 * e * model.get_score(t, None, None, None, non_pad_mask).detach() + sqrt_e * z
            t[t_new>0] = t_new[t_new>0]
    elif opt.sampling_method == 'normal':
        for _ in range(opt.n_save_steps):
            z = torch.randn_like(t)
            t = t + 0.5 * e * model.get_score(t, None, None, None, non_pad_mask).detach() + sqrt_e * z

    # A single denoising step if we add noise once
    # we repeat the last langevin step to ensure all samples are positive
    if opt.add_noise == 'denoise' and opt.is_last and opt.denoise_step:
        i = 0
        t_final = t + opt.var_noise**2 * model.get_score(t, None, None, None, non_pad_mask).detach()
        # while (t_final * non_pad_mask[:,1:] < -1e-5).any() and i<100:
        #     z = torch.randn_like(t)
        #     score = model.get_score(t, None, non_pad_mask).detach()
        #     t = t + 0.5 * e * score + math.sqrt(e) * z
        #     t_new = t + opt.var_noise**2 * model.get_score(t, None, non_pad_mask,idx=0).detach()
        #     t_final[(t_final<0) & (t_new>=0)] = t_new[(t_final<0) & (t_new>=0)]
        #     i += 1
    else:
        t_final = t
    t_final.required_grads=False


    if opt.conditional_sampling:
        t_type = diff_time
        type_prob = model.type_predictor.get_type(t_type, non_pad_mask).squeeze(2)# batch*len-1*num_samples*num_types
        samples = torch.multinomial(type_prob.reshape(-1,opt.num_types).softmax(-1), 100, replacement=True).reshape(t.size()[:3]) # batch*len-1*num_samples
        type_sample = (type_prob, 'logit', samples)
    else:
        t_type = t_final
        type_prob = model.type_predictor.get_type(t_type, non_pad_mask)# batch*len-1*num_samples*num_types
        samples = torch.multinomial(type_prob.detach().reshape(-1,opt.num_types).softmax(-1), 1, replacement=True).reshape(t.size()[:3]) # batch*len-1*num_samples
        type_sample = (samples, 'sample', None)
    return t_final, diff_time, type_sample

def predict_langevin_baseline(model, event_time, time_gap, event_type, prediction, opt, sample_init):

    non_pad_mask = get_non_pad_mask(event_type)    
    e = opt.langevin_step
    n_samples = opt.n_samples
    diff_time = time_gap * non_pad_mask[:, 1:].squeeze(2)
    prediction = prediction.detach()

    if sample_init is None:
        t = torch.rand([*diff_time.size(), n_samples], device=event_type.device) * (opt.time_95 - opt.time_05) + opt.time_05
        ori_size = prediction.size()
        # type_sample = torch.multinomial(prediction.reshape(-1,ori_size[2]).softmax(1), n_samples, replacement=True).reshape(*ori_size[:2],n_samples) # batch*len-1*num_samples  
        if opt.conditional_sampling and opt.thp_no_predictor:
            prob = prediction / (prediction.sum(-1,keepdim=True) + 1e-10) + 1e-10
            samples = torch.multinomial(prob.reshape(-1,ori_size[2]).softmax(1), n_samples, replacement=True).reshape(*ori_size[:2],n_samples) # batch*len-1*num_samples  
            type_sample = (prediction, 'intensity', samples)
        else:
            samples = torch.multinomial(prediction.reshape(-1,ori_size[2]).softmax(1), n_samples, replacement=True).reshape(*ori_size[:2],n_samples) # batch*len-1*num_samples  
            type_sample = (prediction, 'logit', samples)
        # else:
        #     NotImplementedError, 'Can\'t sample conditionally with predictor!!'
    else:
        assert sample_init[0].ndim == 3, 'Wrong sample init size!!!!!!'
        t, type_sample = sample_init

        
    if opt.conditional_sampling:
        type_score = (event_type[:,1:,None] -1) * non_pad_mask[:,1:].long()
    else:
        type_score = type_sample[2]
    
    sqrt_e = math.sqrt(e)
    if opt.sampling_method == 'truncated':
        for _ in range(opt.n_save_steps):
            z = torch.randn_like(t)
            t_new = t + 0.5 * e * model.get_score(t, type_score, event_time, time_gap, non_pad_mask).detach() + sqrt_e * z
            t[t_new>0] = t_new[t_new>0] 
    elif opt.sampling_method == 'normal':
        for uu in range(opt.n_save_steps):
            z = torch.randn_like(t)
            t = t + 0.5 * e * model.get_score(t, type_score, event_time, time_gap, non_pad_mask).detach() + sqrt_e * z
            t.required_grads=False
    
    
    return t, diff_time, type_sample

class LabelSmoothingLoss(nn.Module):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """

    def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=-100):
        assert 0.0 <= label_smoothing <= 1.0
        super(LabelSmoothingLoss, self).__init__()

        self.eps = label_smoothing
        self.num_classes = tgt_vocab_size
        self.ignore_index = ignore_index

    def forward(self, output, target, logit):
        """
        output (FloatTensor): (batch_size) x n_classes
        target (LongTensor): batch_size
        """

        non_pad_mask = target.ne(self.ignore_index).float()

        target[target.eq(self.ignore_index)] = 0
        one_hot = F.one_hot(target, num_classes=self.num_classes).float()
        one_hot = one_hot * (1 - self.eps) + (1 - one_hot) * self.eps / self.num_classes
        if logit:
            log_prb = F.log_softmax(output, dim=-1)
        else:
            log_prb = (output+1e-10).log()

        loss = -(one_hot * log_prb).sum(dim=-1)
        loss = loss * non_pad_mask
        return loss
