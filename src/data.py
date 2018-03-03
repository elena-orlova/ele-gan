import numpy as np



class Dataset():
    def __init__(self, opt, train_dataset=None):
        if train_dataset is None:
            tmp = np.load(opt.train_path)
        else:
            tmp = np.load(opt.val_path)
        self.data_type = opt.data_type
        self.target_type = opt.target_type
        self.data = tmp['EnergyDeposit'].astype('float32')
        self.target = tmp['ParticleMomentum'][:, 2].astype('float32')
        mid = self.data.shape[1]//2
        if opt.image_size < self.data.shape[1]:
            self.data = self.data[:, mid-opt.image_size//2:mid+opt.image_size//2,
                                     mid-opt.image_size//2:mid+opt.image_size//2]
        elif opt.image_size > self.data.shape[1]:
            pad = (opt.image_size - data.shape[1])//2
            self.data = np.pad(self.data, [(0, 0), (pad, pad), (pad, pad)], 
                               'constant', constant_values=0)
            assert self.data.shape[1] == opt.image_size, 'image_size must have same parity as data.shape[1]'
        if train_dataset is None:
            self.mean = self.data.mean(0)
            self.std = self.data.std(0)
        else:
            self.mean = None
            self.std = None
        self.input_data = self.get_input(self.data, self.data_type)
        self.input_target = self.target

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
        return self.input_data[index][None], self.input_target[index][None]

    def __len__(self):
        return self.input_data.shape[0]