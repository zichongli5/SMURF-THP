import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# from transformer.Modules import GELU, BiasedPositionalEmbedding, PositionalEmbedding, EventEmbedding,TypeEmbedding, MultiHeadedAttention, SublayerConnection, PositionwiseFeedForward
import transformer.Constants as Constants
from transformer.Layers import Encoder, Predictor, RNN_layers, get_attn_key_pad_mask, get_non_pad_mask, get_subsequent_mask
from transformer.Score_modules import score_encode, intensity_encode, Predictor_with_time, get_obj, get_obj_denoise
from transformer.Modules import GELU
import Utils


def softplus(x, beta):
    # hard thresholding at 20
    temp = beta * x
    temp[temp > 20] = 20
    return 1.0 / beta * torch.log(1 + torch.exp(temp))

class Transformer(nn.Module):
    """ A sequence to sequence model with attention mechanism. """

    def __init__(self, num_types, config):
        super().__init__()

        self.encoder = Encoder(
            num_types=num_types,
            d_model=config.d_model,
            d_inner=config.d_inner_hid,
            n_layers=config.n_layers,
            n_head=config.n_head,
            d_k=config.d_k,
            d_v=config.d_v,
            dropout=config.dropout,
        )

        self.name = 'thp'
        self.num_types = num_types
        self.normalize = config.normalize
        self.use_predictor = not config.thp_no_predictor
        self.loss_lambda = config.loss_lambda

        # convert hidden vectors into a scalar
        self.linear = nn.Linear(config.d_model, num_types)
        # parameter for the weight of time difference
        self.alpha = nn.Parameter(torch.tensor(-0.1))
        # parameter for the softplus function
        self.beta = nn.Parameter(torch.tensor(1.0))
        # OPTIONAL recurrent layer, this sometimes helps
        self.rnn = RNN_layers(config.d_model, config.d_rnn)

        # prediction of next event type
        if self.use_predictor:
            self.type_predictor = Predictor(config.d_model, num_types)

    def forward(self, event_type, event_time, time_gap):
        """
        Return the hidden representations and predictions.
        For a sequence (l_1, l_2, ..., l_N), we predict (l_2, ..., l_N, l_{N+1}).
        Input: event_type: batch*seq_len;
               event_time: batch*seq_len.
        Output: enc_output: batch*seq_len*model_dim;
                type_prediction: batch*seq_len*num_classes (not normalized);
                time_prediction: batch*seq_len.
        """

        non_pad_mask = get_non_pad_mask(event_type)

        enc_output = self.encoder(event_type, event_time, non_pad_mask)
        self.enc_output = enc_output
        # enc_output = self.rnn(enc_output, non_pad_mask)
        if self.use_predictor:
            type_prediction = self.type_predictor(enc_output, non_pad_mask)
        else:
            type_prediction = self.get_intensity(time_gap, event_type, event_time, time_gap, non_pad_mask).squeeze(2)

        return enc_output, type_prediction
    
    def get_intensity(self, t, event_type, event_time, time_gap, non_pad_mask):
        # t size: batch*len-1*num_samples
        if t.ndim == 2:
            t = t.unsqueeze(2)
        assert t.ndim == 3

        temp_hid = self.linear(self.enc_output)[:, :-1, :] # batch*len-1*num_types

        all_lambda = softplus(temp_hid.unsqueeze(3) + self.alpha * t.unsqueeze(2)/(event_time[:,:-1,None,None]+1e-10), self.beta) #batch*len-1*num_type*1/num_samples
        all_lambda = all_lambda.transpose(2,3) * non_pad_mask[:,1:,None,:] #batch*len-1*1/num_samples*num_type
        
        return all_lambda
    
    def get_score(self, t, event_type, event_time, time_gap, non_pad_mask):

        if not t.requires_grad:
            t = torch.autograd.Variable(t, requires_grad=True)

        if event_type.ndim == 2:
            event_type = event_type.unsqueeze(2)

        all_intensity = self.get_intensity(t, event_type, event_time, time_gap, non_pad_mask) # batch*len*num_samples*num_type
        intensity_total = all_intensity.sum(-1)*non_pad_mask[:,1:,:]

        type_onehot = F.one_hot(event_type, num_classes=self.num_types)*non_pad_mask[:,1:,None,:] # batch*(len-1)*num_samples*num_types
        intensity_type = (type_onehot*all_intensity).sum(-1)
        # intensity_type = all_intensity.sum(-1)
        intensity_type_log = (intensity_type+1e-10).log()*non_pad_mask[:,1:,:]

        intensity_type_grad_t = torch.autograd.grad(intensity_type_log.sum(), t, retain_graph=True)[0]*non_pad_mask[:,1:,:]
        score = intensity_type_grad_t - intensity_total

        return score

    def compute_loss(self, enc_out, event_time, time_gap, event_type, prediction, pred_loss_func):
        non_pad_mask = get_non_pad_mask(event_type)
            
        event_ll, non_event_ll = Utils.log_likelihood(self, event_time, time_gap, event_type)
        event_loss = -torch.sum(event_ll - non_event_ll)
        # event_loss /= non_pad_mask[:,1:].sum()
        if self.use_predictor:
            pred_loss = Utils.type_loss(prediction, event_type, pred_loss_func)
            loss = self.loss_lambda * event_loss + pred_loss
        else:
            loss = event_loss
        return loss

class smurf_thp(nn.Module):
    """ 
    decoder: encode
    add_noise: None, denoise
    noise_type: normal, truncated
    parametrize: intensity, score
    """


    def __init__(
            self,
            num_types, config):
        
        super().__init__()

        self.encoder = Encoder(
            num_types=config.num_types,
            d_model=config.d_model,
            d_inner=config.d_inner_hid,
            n_layers=config.n_layers,
            n_head=config.n_head,
            d_k=config.d_k,
            d_v=config.d_v,
            dropout=config.dropout,
        )

        self.name = config.model
        self.config = config
        self.num_types = num_types
        self.gelu = GELU()
        self.loss_lambda = config.loss_lambda


        # prediction of next event type
        self.type_predictor = Predictor_with_time(config.d_model, num_types)

        if config.parametrize == 'score':
            if config.decoder == 'encode':
                self.score_decoder = score_encode(config.d_model,config.d_model,config.num_types, self.encoder, config)
        elif config.parametrize == 'intensity':
            if config.decoder == 'encode':
                self.score_decoder = intensity_encode(config.d_model,config.d_model,config.num_types, self.encoder, config)


    def forward(self, event_type, event_time, time_gap):
        """
        forward function wrapper for different noise option: no noise, one noise, multi noise
        """
        if self.config.add_noise == 'None':
            return self.forward_base(event_type, event_time, time_gap)
        elif self.config.add_noise == 'denoise':
            return self.forward_denoise(event_type, event_time, time_gap)

    def forward_base(self, event_type, event_time, time_gap):
        """
        Return the hidden representations and predictions.
        For a sequence (l_1, l_2, ..., l_N), we predict (l_2, ..., l_N, l_{N+1}).
        Input: event_type: batch*seq_len;
               event_time: batch*seq_len.
        Output: enc_output: batch*seq_len*model_dim;
                type_prediction: batch*seq_len*num_classes (not normalized);
        """
        non_pad_mask = get_non_pad_mask(event_type)
        enc_output = self.encoder(event_type, event_time, non_pad_mask)

        diff_time = time_gap
        diff_time *= non_pad_mask[:,1:].squeeze(-1)

        type_prediction = self.type_predictor(enc_output)
        type_prediction = self.type_predictor.get_type(diff_time, non_pad_mask).squeeze(2)
        
        t_var = torch.autograd.Variable(diff_time, requires_grad=True)

        _ = self.score_decoder(enc_output)
        score = self.score_decoder.get_score(t_var, event_type[:,1:], event_time, time_gap, non_pad_mask).squeeze(-1)

        # torch.autograd.set_detect_anomaly(True)
        obj = get_obj(t_var, score, non_pad_mask)
        
        return obj, type_prediction

    def forward_denoise(self, event_type, event_time, time_gap):
        """
        Return the hidden representations and predictions.
        For a sequence (l_1, l_2, ..., l_N), we predict (l_2, ..., l_N, l_{N+1}).
        Input: event_type: batch*seq_len;
               event_time: batch*seq_len.
        Output: enc_output: batch*seq_len*model_dim;
                type_prediction: batch*seq_len*num_classes (not normalized);
        """

        non_pad_mask = get_non_pad_mask(event_type)
        enc_output = self.encoder(event_type, event_time, non_pad_mask)
        
        diff_time = time_gap
        diff_time *= non_pad_mask[:,1:].squeeze(-1)

        if self.config.noise_type == 'normal':
            noise = self.config.var_noise * torch.randn([*diff_time.size(), self.config.num_noise], device = diff_time.device)
            t_noise = diff_time[:,:,None] + noise
            t_var = t_noise
        elif self.config.noise_type == 'truncated':
            noise = self.config.var_noise * torch.randn([*diff_time.size(), self.config.num_noise], device = diff_time.device)
            t_noise = diff_time[:,:,None] + noise
            while ((t_noise<0)*non_pad_mask[:,1:,:].bool()).any():           
                noise_new = self.config.var_noise * torch.randn([*diff_time.size(), self.config.num_noise], device = diff_time.device)
                t_noise_new = diff_time[:,:,None] + noise_new
                t_noise[t_noise<0] = t_noise_new[t_noise<0]
                # print((t_noise<0).sum())
            t_var = t_noise
        else:
            NotImplementedError, 'Wrong noise type!!'
        
        type_prediction = self.type_predictor(enc_output)
        type_prediction = self.type_predictor.get_type(t_var, non_pad_mask)
        
        if self.config.parametrize == 'intensity':
            t_var = torch.autograd.Variable(t_var, requires_grad=True)
        
        _ = self.score_decoder(enc_output)
        score = self.score_decoder.get_score(t_var, event_type[:,1:],event_time, time_gap, non_pad_mask).squeeze(-1)

        # torch.autograd.set_detect_anomaly(True)
        obj = get_obj_denoise(diff_time, t_var, score, self.config.var_noise, non_pad_mask)
        
        return obj, type_prediction

    def compute_loss(self, enc_out, event_time, time_gap, event_type, prediction, pred_loss_func):

        event_loss = enc_out.sum()
        pred_loss = Utils.type_loss(prediction, event_type, pred_loss_func)
        # print(event_loss, pred_loss)
        loss = self.loss_lambda * event_loss + pred_loss

        return loss
    
    def get_score(self, t, event_type, event_time, time_gap, non_pad_mask, idx=None):
        return self.score_decoder.get_score(t, event_type, event_time, time_gap, non_pad_mask, idx)
