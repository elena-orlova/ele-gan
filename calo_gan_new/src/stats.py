import numpy as np
import xgboost as xgb
import statsmodels.stats.api as sms
from matplotlib import pyplot as plt
plt.switch_backend('agg')
import torch
from torch.autograd import Variable



class Stats():
    def __init__(self, opt):
        self.regressor = xgb.Booster({'nthread': 4})
        self.regressor.load_model(opt.regressor_path)
        self.n_bins = 30
        self.e_init_min = 10
        self.e_init_max = 100
        self.radius = 4
        self.mid = opt.image_size//2
        self.masks = []
        for r in range(1, self.radius+1):
            mask = np.zeros((opt.image_size, opt.image_size), dtype=int)
            mask[self.mid-r:self.mid+r, self.mid-r] = 1
            mask[self.mid-r:self.mid+r, self.mid+r] = 1
            mask[self.mid-r, self.mid-r+1:self.mid+r-1] = 1
            mask[self.mid+r, self.mid-r+1:self.mid+r-1] = 1
            # expand to the data size
            mask = mask[None]
            self.masks.append(mask)
        self.masks_torch = []
        for mask in self.masks:
            self.masks_torch.append(torch.from_numpy(mask.repeat(opt.batch_size, 0)).cuda() > 0)
        for i in range(len(self.masks)):
            self.masks[i] = self.masks[i] > 0
        self.e_bins = np.linspace(self.e_init_min, self.e_init_max, self.n_bins+1)
        self.real_stats = []
        self.fake_stats = []
        # training and val losses
        self.train_dis_adv_loss = []
        self.train_gen_adv_loss = []
        self.val_dis_adv_loss = []
        self.val_gen_adv_loss = []
        self.train_dis_stats_loss = []
        self.train_gen_stats_loss = []
        self.val_dis_stats_loss = []
        self.val_gen_stats_loss = []
        self.step = opt.val_every_epoch

    def calc_stats(self, data, real=False):
        if not real:
            self.fake_stats = []
        # calculate initial energy distribution
        input = xgb.DMatrix(data[:, self.mid-2:self.mid+2, self.mid-2:self.mid+2].reshape(-1, 16))
        self.e_init = self.regressor.predict(input)
        hist_e_init = np.histogram(self.e_init, self.n_bins, normed=True)[0]
        if real:
            self.real_stats.append(hist_e_init)
        else:
            self.fake_stats.append(hist_e_init)
        # calculate normalized energy stds over calo areas
        data_norm = data / self.e_init.reshape(-1, 1, 1)
        e_calo_norm_std = np.empty((self.n_bins, self.radius))
        # calculate Ei/E0
        e_i = self.ei_by_e0(data)
        if real:
            self.e_i = e_i
        e_i_mean = np.empty((self.n_bins, self.radius-1))
        e_i_cint = np.empty((self.n_bins, self.radius-1, 2))
        # calculate RMS of (E_calo / E_true) / <E_calo / E_true> 
        e_calo_by_e_init = data.sum((1, 2)) / self.e_init
        e_dim_rms = np.empty((self.n_bins))
        for i in range(self.n_bins):
            idx = np.where((self.e_init >= self.e_bins[i]) * (self.e_init < self.e_bins[i+1]))[0]
            for r in range(1, self.radius+1):
                # energy stds
                tmp = data_norm[idx]
                e_calo_norm_std[i, r-1] = tmp[self.masks[r-1].repeat(tmp.shape[0], 0)].std()
                if r < self.radius:
                    # Ei/E0
                    e_i_mean[i, r-1] = e_i[idx, r-1].mean()
                    if not real:
                        cm = sms.CompareMeans(
                            sms.DescrStatsW(self.e_i[idx, r-1]),
                            sms.DescrStatsW(e_i[idx, r-1]))
                        e_i_cint[i, r-1, :] = cm.tconfint_diff(usevar='unequal')
            # RMS
            tmp = e_calo_by_e_init[idx]
            tmp /= tmp.mean()
            e_dim_rms[i] = (tmp**2).mean()**0.5 
        if real:
            self.real_stats.append(e_calo_norm_std)
            self.e_i_mean = e_i_mean
            self.real_stats.append(e_i_mean)
            self.real_stats.append(e_dim_rms)
        else:
            self.fake_stats.append(e_calo_norm_std)
            e_i_cint = e_i_cint - self.e_i_mean[:, :, None] + e_i_mean[:, :, None]
            e_i_cint[:, :, 0] *= -1
            self.fake_stats.append(e_i_mean)
            self.fake_stats.append(e_i_cint)
            self.fake_stats.append(e_dim_rms)

    def calc_stats_numpy(self, data, real=True):
        stats = np.empty((data.shape[0], 9), dtype='float32')
        self.mid = data.shape[1]//2
        input = xgb.DMatrix(data[:, self.mid-2:self.mid+2, self.mid-2:self.mid+2].reshape(-1, 16))
        e_init = self.regressor.predict(input)
        stats[:, 0] = e_init.astype('float32')
        data_norm = data / e_init.reshape(-1, 1, 1)
        masks = []
        for mask in self.masks:
            masks.append(mask.repeat(data.shape[0], 0))
        offset = 1
        for r in range(1, self.radius+1):
            stats[:, offset+r-1] = data_norm[masks[r-1]].reshape(data.shape[0], -1).sum(1)
        offset += self.radius
        e0 = data[masks[0]].reshape(data.shape[0], -1).sum(1) + 1e-8
        for r in range(2, self.radius+1):
            stats[:, offset+r-2] = data[masks[r-1]].reshape(data.shape[0], -1).sum(1) / e0
        offset += self.radius-1
        stats[:, offset] = data_norm.sum((1, 2))
        if real:
            self.stats_mean = stats.mean(0, keepdims=True).astype('float32')
            self.stats_mean_torch = torch.from_numpy(self.stats_mean).cuda()
            self.stats_std = stats.std(0, keepdims=True).astype('float32')
            self.stats_std_torch = torch.from_numpy(self.stats_std).cuda()
        stats = (stats - self.stats_mean) / self.stats_std
        return stats

    def calc_stats_torch(self, data):
        if self.mean is not None:
            data = data.clone() * (self.std + 1e-8) + self.mean
        stats = torch.cuda.FloatTensor(data.size(0), 9)
        self.mid = data.shape[1]//2
        input = xgb.DMatrix(data[:, self.mid-2:self.mid+2, self.mid-2:self.mid+2].cpu().numpy().reshape(-1, 16))
        e_init = self.regressor.predict(input)
        stats[:, 0] = torch.from_numpy(e_init.astype('float32'))
        e_init = stats[:, 0]
        data_norm = data / e_init.clone().view(-1, 1, 1)
        offset = 1
        for r in range(1, self.radius+1):
            stats[:, offset+r-1] = data_norm[self.masks_torch[r-1]].view(data.shape[0], -1).sum(1)
        offset += self.radius
        e0 = data[self.masks_torch[0]].view(data.shape[0], -1).sum(1) + 1e-8
        for r in range(2, self.radius+1):
            stats[:, offset+r-2] = data[self.masks_torch[r-1]].view(data.shape[0], -1).sum(1) / e0
        offset += self.radius-1
        stats[:, offset] = data_norm.sum(1).sum(1)
        stats = (stats - self.stats_mean_torch) / self.stats_std_torch
        stats = Variable(stats)
        return stats

    def radial_std(self, data_norm, r, e_init_min, e_init_max):
        # get data samples with required energy
        return 

    def ei_by_e0(self, data):
        ei = []
        for r in range(self.radius+1):
            ei.append(data[:,self.mid-r:self.mid+r,self.mid-r:self.mid+r].sum((1, 2)))
        ei = np.asarray(ei)
        ei = ei[1:] - ei[:-1]
        ei /= (ei[0] + 1e-8)
        return ei[1:].T

    def get_plot(self):
        assert len(self.real_stats) and len(self.fake_stats), 'must compute real and fake stats before get_plot'
        # stats = [hist_e_init, e_calo_norm_std, e_i_mean, e_i_cint]
        x_epoch = np.arange(0, len(self.train_dis_adv_loss)*self.step, self.step)+self.step
        x_energy = (self.e_bins[1:] + self.e_bins[:-1])/2.
        f, ax = plt.subplots(7, 2, figsize=(12, 36))
        ax[0, 0].set_title('Adversarial loss')
        ax[0, 0].plot(x_epoch, self.train_dis_adv_loss)
        legend = ['train dis']
        if len(self.train_gen_adv_loss):
            ax[0, 0].plot(x_epoch, self.train_gen_adv_loss)
            legend += ['train gen']
        if len(self.val_dis_adv_loss):
            ax[0, 0].plot(x_epoch, self.val_dis_adv_loss)
            legend += ['val dis']
        if len(self.val_gen_adv_loss):
            ax[0, 0].plot(x_epoch, self.val_gen_adv_loss)
            legend += ['val gen']
        ax[0, 0].legend(legend)
        ax[0, 0].set_xlabel('epoch')
        ax[0, 0].set_ylabel('loss')
        ax[0, 1].set_title('Stats MSE loss')
        legend = []
        if len(self.train_dis_stats_loss):
            ax[0, 1].plot(x_epoch, self.train_dis_stats_loss)
            legend += ['train dis']
        if len(self.train_gen_stats_loss):
            ax[0, 1].plot(x_epoch, self.train_gen_stats_loss)
            legend += ['train gen']
        if len(self.val_dis_stats_loss):
            ax[0, 1].plot(x_epoch, self.val_dis_stats_loss)
            legend += ['val dis']
        if len(self.val_gen_stats_loss):
            ax[0, 1].plot(x_epoch, self.val_gen_stats_loss)
            legend += ['val gen']
        ax[0, 1].legend(legend)
        ax[0, 1].set_xlabel('epoch')
        ax[0, 1].set_ylabel('loss')
        offset = 1
        # hist_e_init
        ax[offset, 0].set_title('Distribution of E_init')
        ax[offset, 0].plot(x_energy, self.real_stats[0])
        ax[offset, 0].plot(x_energy, self.fake_stats[0])
        ax[offset, 0].legend(['real', 'fake'])
        ax[offset, 0].set_xlabel('initial energy')
        ax[offset, 0].set_ylabel('p')
        # rms
        ax[offset, 1].set_title('RMS of E_calo/E_init / <E_calo/E_init>')
        ax[offset, 1].plot(x_energy, self.real_stats[-1])
        ax[offset, 1].plot(x_energy, self.fake_stats[-1])
        ax[offset, 1].legend(['real', 'fake'])
        ax[offset, 1].set_xlabel('initial energy')
        ax[offset, 1].set_xlabel('rms')
        offset += 1
        # e_calo_norm_std
        for i in range(self.real_stats[1].shape[1]):
            ax[offset+i//2, i%2].set_title('Std of E_calo/E_init in square boundary, r = %d' % (i+1))
            ax[offset+i//2, i%2].plot(x_energy, self.real_stats[1][:, i])
            ax[offset+i//2, i%2].plot(x_energy, self.fake_stats[1][:, i])
            ax[offset+i//2, i%2].legend(['real', 'fake'])
            ax[offset+i//2, i%2].set_xlabel('initial energy')
            ax[offset+i//2, i%2].set_ylabel('std')
        offset += self.real_stats[1].shape[1]//2
        # e_i
        for i in range(self.real_stats[2].shape[1]):
            ax[offset+i, 0].set_title('Mean of E%d/E0' % (i+1))
            ax[offset+i, 0].plot(x_energy, self.real_stats[2][:, i])
            ax[offset+i, 0].plot(x_energy, self.fake_stats[2][:, i])
            ax[offset+i, 0].legend(['real', 'fake'])
            ax[offset+i, 0].set_xlabel('initial energy')
            ax[offset+i, 0].set_ylabel('mean')
            ax[offset+i, 1].set_title('Difference between real and fake mean E%d/E0' % (i+1))
            ax[offset+i, 1].errorbar(x_energy, self.real_stats[2][:, i]-self.fake_stats[2][:, i], self.fake_stats[3][:, i].T)
            ax[offset+i, 1].set_xlabel('initial energy')
            ax[offset+i, 1].set_ylabel('diff')
        offset += self.real_stats[2].shape[1]
        return f