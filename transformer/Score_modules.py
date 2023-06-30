import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from transformer.Modules import PositionwiseFeedForward

def get_type_mask(event_type, num_types):
    type_mask = F.one_hot(event_type, num_classes = num_types + 1)[...,1:]
    return type_mask

def get_obj(t_var, score, non_pad_mask):
    grad_t = torch.autograd.grad(score.sum(), t_var, retain_graph=True)[0].squeeze(-1)
    if grad_t.ndim==1:
        grad_t = grad_t.unsqueeze(-1)
    # grad_t = torch.cat([torch.autograd.grad([score[i,j] for j in range(len(score[0]))], t_var, create_graph=True)[0][i:i+1,:] for i in range(len(score))],0).squeeze(-1)
    # print(grad_t.size(),non_pad_mask.squeeze(-1)[:,1:].bool().size())
    grad_t = grad_t.masked_fill_(~non_pad_mask.squeeze(-1)[:,1:].bool(), 0.0)
    obj = grad_t + 0.5 * score ** 2
    obj *= non_pad_mask.squeeze(-1)[:,1:]

    return obj

def get_obj_denoise(t_true, t_var, score, var_noise, non_pad_mask):
    # t_true: (batch*len-1)
    # t_var: (batch*len-1*num_samples)
    var_noise = var_noise
    target = (t_true[:,:,None] - t_var)/var_noise**2
    obj = 0.5 * (score - target)**2
    obj *= var_noise**2
    obj *= non_pad_mask[:,1:,:]
    obj = obj.mean(2)
    return obj
    
class intensity_encode(nn.Module):
    """
    Core score matching module. Two main function:
    get_score: output score given time and type
    get_obj: output objectives given score, time and type
    """

    def __init__(self, d_model, d_inner, num_types, encoder, config):
        super().__init__()
        self.num_types = num_types
        self.config = config
        self.d_inner = d_inner
        self.encoder = encoder
 
        self.base_layer = nn.Sequential(
                nn.Linear(d_model, d_inner, bias=True)
                )

        self.affect_layer = nn.Sequential(
                nn.Linear(d_model, d_inner, bias=True),
                nn.Tanh()
                )
        self.intensity_layer = nn.Sequential(
                nn.Linear(d_inner, 1, bias = True),
                nn.Softplus(beta=1.0)
                )

    def forward(self, enc_output):
        # enc_output: output of the encoder (batch*len*d_model)
        self.affect = self.affect_layer(enc_output) # (batch*len*d_inner)
        self.base = self.base_layer(enc_output) # (batch*len*d_inner)

    def get_score(self, t, event_type, event_time, time_gap, non_pad_mask, idx=None):
        """Output scores for add_noise = None/denoise"""
        # t: time of events; t could be the training gt of size: (batch*len-1)/(batch*len-1*num_noise) or langevin sampling of size:(batch*len-1*num_samples)
        # event_type: event types; could be the gt or the prediction types (batch*len)
        # non_pad_mask: (batch*len*1)        

        if t.ndim == 2:
            t = t.unsqueeze(2)
        assert t.ndim == 3

        if not t.requires_grad:
            t = torch.autograd.Variable(t, requires_grad=True)

        # tem_enc = self.encoder.temporal_enc(t, non_pad_mask[:,1:,:,None]) # batch*len-1*1/num_samples*d_model

        intensity = torch.tanh(self.affect[:,:-1,None,:] * t.unsqueeze(3) + self.base[:,:-1,None,:]) # batch*len-1*1/num_samples*d_model
        intensity = self.intensity_layer(intensity).squeeze(3) # (batch*len-1*1/num_samples)
        intensity = intensity*non_pad_mask[:,1:,:]
        intensity_log = (intensity+1e-10).log()*non_pad_mask[:,1:,:]

        intensity_grad_t = torch.autograd.grad(intensity_log.sum(), t, retain_graph=True)[0]*non_pad_mask[:,1:,:]
        score = intensity_grad_t - intensity

        return score

class score_encode(nn.Module):
    """
    Core score matching module. Two main function:
    get_score: output score given time and type
    get_obj: output objectives given score, time and type
    """

    def __init__(self, d_model, d_inner, num_types, config):
        super().__init__()
        self.num_types = num_types
        self.config = config
        self.d_inner = d_inner
        self.base_layer = nn.Sequential(
            nn.Linear(d_model, d_inner, bias=True)
            )

        self.affect_layer = nn.Sequential(
            nn.Linear(d_model, d_inner, bias=True),
            nn.Tanh()
            )
        self.score_layer = nn.Sequential(
            nn.Linear(d_inner, 1, bias = True),
            nn.Softplus(beta=1.0)
            )
    
    def forward(self, enc_output):
        # enc_output: output of the encoder (batch*len*d_model)
        self.affect = self.affect_layer(enc_output) # (batch*len*d_inner)
        self.base = self.base_layer(enc_output) # (batch*len*d_inner)

    def get_score(self, t, event_type, non_pad_mask, idx=None):
        """Output scores for add_noise = None/denoise"""
        # t: time of events; t could be the training gt of size: (batch*len-1)/(batch*len-1*num_noise) or langevin sampling of size:(batch*len-1*num_samples)
        # event_type: event types; could be the gt or the prediction types (batch*len)
        # non_pad_mask: (batch*len*1)        
    
        if t.ndim == 2:
            t = t.unsqueeze(2)
        assert t.ndim == 3

        if not t.requires_grad:
            t = torch.autograd.Variable(t, requires_grad=True)


        score = torch.tanh(self.affect[:,:-1,None,:] * t.unsqueeze(3) + self.base[:,:-1,None,:]) # batch*len-1*1/num_samples*d_model
        score = self.score_layer(score).squeeze(3) # (batch*len-1*1/num_samples)
        score = score*non_pad_mask[:,1:,:]

        return score


class Predictor_with_time(nn.Module):
    """ Prediction of next event type. """

    def __init__(self, dim, num_types):
        super().__init__()
        self.base_layer = nn.Linear(dim, int(dim/2), bias=True)

        self.affect_layer = nn.Sequential(
                nn.Linear(dim, int(dim/2), bias=True),
                nn.Tanh()
            )
        self.classifier = nn.Linear(int(dim/2), num_types, bias=True)
        # nn.init.xavier_normal_(self.classifier.weight)

    def forward(self, enc_output):
        # enc_output: output of the encoder (batch*len*d_model)
        self.affect = self.affect_layer(enc_output) # (batch*len*d_inner)
        self.base = self.base_layer(enc_output) # (batch*len*d_inner)

    def get_type(self, t, non_pad_mask):
        
        if t.ndim == 2:
            t = t.unsqueeze(2)
        assert t.ndim == 3
        
        out = torch.tanh(self.affect[:,:-1,None,:] * t.unsqueeze(3)+self.base[:,:-1,None,:])
        out = self.classifier(out)
        out = out * non_pad_mask[:,1:,:,None]
        return out


