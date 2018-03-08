import numpy as np
import torch



class Dataset():
    def __init__(self, opt, train_dataset=None, stats=None):
        if train_dataset is None:
            tmp = np.load(opt.train_path)
        else:
            tmp = np.load(opt.val_path)
        self.data_type = opt.data_type
        self.target_type = opt.target_type
        data = tmp['EnergyDeposit'].astype('float32')
        target = tmp['ParticleMomentum'][:, 2].astype('float32')
        mid = data.shape[1]//2
        if opt.image_size < data.shape[1]:
            data = data[:, mid-opt.image_size//2:mid+opt.image_size//2,
                           mid-opt.image_size//2:mid+opt.image_size//2]
        elif opt.image_size > data.shape[1]:
            pad = (opt.image_size - data.shape[1])//2
            data = np.pad(data, [(0, 0), (pad, pad), (pad, pad)], 
                               'constant', constant_values=0)
            assert data.shape[1] == opt.image_size, 'image_size must have same parity as data.shape[1]'
        if train_dataset is None:
            self.mean = data.mean(0)
            self.std = data.std(0)
            if opt.data_type == 'norm':
                stats.mean = torch.from_numpy(self.mean.astype('float32')[None]).cuda()
                stats.std = torch.from_numpy(self.std.astype('float32')[None]).cuda()
            elif opt.data_type == 'none':
                stats.mean = None
                stats.std = None
        else:
            self.mean = None
            self.std = None
        self.data = data.copy()
        self.input_data = self.get_input(data, self.data_type)
        if stats is not None:
            self.input_stats = stats.calc_stats_numpy(self.data)
        else:
            self.input_stats = target

    def get_input(self, data, data_type):
        if data_type == 'none':
            input_data = data
        elif data_type == 'norm':
            if self.mean is None: self.mean = data.mean(0)
            if self.std is None: self.std = data.std(0)
            input_data = (data - self.mean) / (self.std + 1e-8)
        return input_data

    def get_output(self, input_data, data_type):
        if data_type == 'none':
            output_data = input_data
        elif data_type == 'norm':
            output_data = input_data * (self.std + 1e-8) + self.mean
        return output_data

    def __getitem__(self, index):
        return self.input_data[index][None], self.input_stats[index]

    def __len__(self):
        return self.input_data.shape[0]