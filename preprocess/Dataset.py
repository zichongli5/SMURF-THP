import numpy as np
import torch
import torch.utils.data
import math

from transformer import Constants
import functools


class EventData(torch.utils.data.Dataset):
    """ Event stream dataset. """

    def __init__(self, data, opt, split):
        """
        Data should be a list of event streams; each event stream is a list of dictionaries;
        each dictionary contains: time_since_start, time_since_last_event, type_event
        """
        self.time = [[elem['time_since_start'] for elem in inst] for inst in data]
        self.time_gap = [[elem['time_since_last_event'] for elem in inst[1:]] for inst in data]
        # plus 1 since there could be event type 0, but we use 0 as padding
        self.event_type = [[elem['type_event'] + 1 for elem in inst] for inst in data]

        self.length = len(data)

        time_flat = []
        for i in self.time_gap:
            time_flat += i
        time_flat = np.array(time_flat)
        # print(self.time_gap[0], self.time[0])
        # get time percentile for future sampling
        if split == 'train':
            # compute normalize and statistics
            if opt.normalize == 'normal':
                mean_data = time_flat.mean().item()
                opt.mean_data = mean_data
                time_flat /= mean_data
                # self.time = [[elem/mean_data for elem in inst] for inst in self.time]
                self.time_gap = [[elem/mean_data for elem in inst] for inst in self.time_gap]
                self.time = [[elem/mean_data for elem in inst] for inst in self.time]
            if opt.normalize == 'log':
                mean_data = time_flat.mean().item()
                # mean_data=1
                time_flat /= mean_data
                mean_log_data = np.log(time_flat+1e-9).mean().item()
                var_log_data = np.log(time_flat+1e-9).std().item()
                print(time_flat.mean())
                time_flat = (np.log(time_flat+1e-9)-mean_log_data)/var_log_data
                print(time_flat.mean())
                opt.mean_data = mean_data
                opt.mean_log_data = mean_log_data
                opt.var_log_data = var_log_data
                # self.time = [[(math.log(elem+1e-5)-mean_data)/var_data for elem in inst] for inst in self.time]
                self.time_gap = [[(math.log(elem/opt.mean_data+1e-9)-opt.mean_log_data)/opt.var_log_data for elem in inst] for inst in self.time_gap]
            
            opt.time_min = np.min(time_flat)
            opt.time_max = np.max(time_flat)
            opt.time_mean = np.mean(time_flat)
            opt.time_std = np.std(time_flat)
            opt.time_median = np.quantile(time_flat,0.5)
            opt.time_05 = np.quantile(time_flat,0.05)
            opt.time_95 = np.quantile(time_flat,0.95)
            opt.time_99 = np.quantile(time_flat,0.99)
            opt.time_quantile = np.array([np.quantile(time_flat,i*0.1) for i in range(1,10)])
        else:
            # normalize
            if opt.normalize == 'normal':
                self.time_gap = [[elem/opt.mean_data for elem in inst] for inst in self.time_gap]
                self.time = [[elem/opt.mean_data for elem in inst] for inst in self.time]
            if opt.normalize == 'log':
                self.time_gap = [[(math.log(elem/opt.mean_data+1e-9)-opt.mean_log_data)/opt.var_log_data for elem in inst] for inst in self.time_gap]
            
        self.max_len = max([len(inst) for inst in data])
        # print('Dataset Quantile:',[np.quantile(np.array(timeflat),i*0.1) for i in range(1,10)])


    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """ Each returned element is a list, which represents an event stream """
        return self.time[idx], self.time_gap[idx], self.event_type[idx]


def pad_time(insts):
    """ Pad the instance to the max seq length in batch. """

    max_len = max(len(inst) for inst in insts)

    batch_seq = np.array([
        inst + [Constants.PAD] * (max_len - len(inst))
        for inst in insts])

    return torch.tensor(batch_seq, dtype=torch.float32)


def pad_type(insts):
    """ Pad the instance to the max seq length in batch. """

    max_len = max(len(inst) for inst in insts)

    batch_seq = np.array([
        inst + [Constants.PAD] * (max_len - len(inst))
        for inst in insts])

    return torch.tensor(batch_seq, dtype=torch.long)


def collate_fn(insts):
    """ Collate function, as required by PyTorch. """

    time, time_gap, event_type = list(zip(*insts))
    time = pad_time(time)
    time_gap = pad_time(time_gap)
    event_type = pad_type(event_type)
    return time, time_gap, event_type


def get_dataloader(data, opt, shuffle=True, split='train'):
    """ Prepare dataloader. """

    ds = EventData(data, opt, split)
    dl = torch.utils.data.DataLoader(
        ds,
        num_workers=2,
        batch_size=opt.batch_size,
        collate_fn=collate_fn,
        shuffle=shuffle
    )
    opt.max_len = max(opt.max_len, ds.max_len)

    return dl