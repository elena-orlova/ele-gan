from torch.utils.data import Dataset, DataLoader
import numpy as np
from scipy import stats
import os
import torch



def get_dataloader(opt):

    dataset_train = MyDataset(opt, opt.train_data_path)
    dataset_val = MyDataset(opt, opt.val_data_path)

    dataloader_train = DataLoader(
        dataset_train,
        opt.batch_size,
        shuffle = True,
        num_workers = opt.num_workers,
        drop_last = True)

    dataloader_val = DataLoader(
        dataset_val,
        opt.batch_size,
        shuffle = True,
        num_workers = opt.num_workers,
        drop_last = True)

    return (dataset_train,
            dataloader_train,  
            dataset_val,
            dataloader_val)


class Preprocessing(object):

    def __init__(
        self, 
        input, 
        name, 
        boxcox = False, 
        rewrite = False):
        super(Preprocessing, self).__init__()

        if not os.path.exists('data/utils'):
            os.makedirs('data/utils')

        path = 'data/utils/%s_params.npz' % name

        # Check if params require rewriting
        if os.path.exists(path):

            data = np.load(path)

            if (boxcox and data['l'] is None or
                input.shape[1:] != data['mean'].shape):

                rewrite = True

        # Compute or read preprocessing params
        if not os.path.exists(path) or rewrite:

            if boxcox:

                # Compute boxcox transformation parameters
                input_ = input.reshape(input.shape[0], -1)
                l = np.empty(input_.shape[1])                
                for i in range(len(l)):
                    l[i] = stats.boxcox_normmax(input_[:, i] + 1e-8, method='mle')
                self.l = l.reshape(input.shape[1:]).astype('float32')

                l = np.broadcast_to(self.l, input.shape) + 1e-8
                input = ((input + 1e-8)**l - 1) / l
            else:

                self.l = np.array([])

            self.mean = input.mean(0).astype('float32')
            self.std = input.std(0).astype('float32')
            self.min = input.min(0).astype('float32')
            self.max = input.max(0).astype('float32')
            np.savez(path, 
                     l = self.l,
                     mean = self.mean, 
                     std = self.std,
                     min = self.min,
                     max = self.max)
        else:

            self.l = data['l']
            self.mean = data['mean']
            self.std = data['std']
            self.min = data['min']
            self.max = data['max']

    def __call__(self, input, phase='to'):

        if phase == 'to':

            if len(self.l):
                l = np.broadcast_to(self.l, input.shape) + 1e-8
                input = ((input + 1e-8)**l - 1) / l

            input = (input - self.mean) / (self.std + 1e-8)
            input = torch.from_numpy(input)

        elif phase == 'from':

            input = input.cpu().numpy()
            input = input * (self.std + 1e-8) + self.mean

            if len(self.l):
                l = np.broadcast_to(self.l, input.shape) + 1e-8
                input = (input * l + 1)**(1./l) - 1e-8
        else:

            assert False, 'Unknown phase'

        return input


class MyDataset(Dataset):

    def __init__(self, opt, data_path):
        super(MyDataset, self).__init__()

        data = np.load(data_path)

        # TODO: replace these datasets with unified ones
        inputs = data['ParticleMomentum'].astype('float32')
        input_idx = [int(i) for i in opt.input_idx.split(',')]
        self.inputs = inputs[:, input_idx]

        targets = data['EnergyDeposit'].astype('float32')
        mid = targets.shape[1]//2
        rad = opt.image_size//2
        l, r = mid-rad, mid+rad
        self.targets = targets[:, l:r, l:r]

        self.transform_input = Preprocessing(
            input = self.inputs, 
            name = 'ParticleMomentum', 
            boxcox= True)
        self.transform_target = Preprocessing(
            input = self.targets, 
            name = 'EnergyDeposit', 
            boxcox = opt.target_boxcox)

    def __getitem__(self, index):

        input_orig = self.inputs[index]
        input = self.transform_input(self.inputs[index])
        target = self.transform_target(self.targets[index])[None]

        return input, input_orig, target

    def __len__(self):

        return self.inputs.shape[0]
