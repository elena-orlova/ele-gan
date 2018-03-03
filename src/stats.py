import numpy as np
import xgboost as xgb
import statsmodels.stats.api as sms
from matplotlib import pyplot as plt



class Stats():
    def __init__(self, opt):
        self.regressor = xgb.Booster({'nthread': 4})
        self.regressor.load_model(opt.regressor_path)
        self.n_bins = 90
        self.e_init_min = 10
        self.e_init_max = 100
        self.radius = 4
        self.e_bins = np.linspace(self.e_init_min, self.e_init_max, self.n_bins+1)
        self.real_stats = []
        self.fake_stats = []
        # training and val losses
        self.train_loss = []
        self.val_loss = []
        self.step = opt.val_epoch_freq

    def calc_stats(self, data, real=False):
        if not real:
            self.fake_stats = []
        # calculate initial energy distribution
        self.mid = data.shape[1]//2
        input = xgb.DMatrix(data[:, self.mid-2:self.mid+2, self.mid-2:self.mid+2].reshape(-1, 16))
        self.e_init = self.regressor.predict(input)
        hist_e_init = np.histogram(self.e_init, self.n_bins, normed=True)[0]
        if real:
            self.real_stats.append(hist_e_init)
        else:
            self.fake_stats.append(hist_e_init)
        # divide data by initial energy
        data_norm = data / self.e_init.reshape(-1, 1, 1)
        # calculate normalized energy stds over calo areas
        e_calo_norm_std = np.empty((self.n_bins, self.radius))
        for i in range(self.n_bins):
            for r in range(1, self.radius+1):
                e_calo_norm_std[i, r-1] = self.radial_std(data_norm, r, self.e_bins[i], self.e_bins[i+1])
        if real:
            self.real_stats.append(e_calo_norm_std)
        else:
            self.fake_stats.append(e_calo_norm_std)
        # calculate ei/e0
        e_i = self.ei_by_e0(data)
        if real:
            self.e_i = e_i
        e_i_mean = np.empty((self.n_bins, self.radius-1))
        e_i_cint = np.empty((self.n_bins, self.radius-1, 2))
        for i in range(self.n_bins):
            for r in range(1, self.radius):
                idx = np.where((self.e_init >= self.e_bins[i]) * (self.e_init < self.e_bins[i+1]))[0]
                e_i_mean[i, r-1] = e_i[idx, r-1].mean()
                if not real:
                    cm = sms.CompareMeans(
                        sms.DescrStatsW(self.e_i[idx, r-1]),
                        sms.DescrStatsW(e_i[idx, r-1]))
                    e_i_cint[i, r-1, :] = cm.tconfint_diff(usevar='unequal')
        if real:
            self.e_i_mean = e_i_mean
            self.real_stats.append(e_i_mean)
        else:
            e_i_cint = e_i_cint - self.e_i_mean[:, :, None] + e_i_mean[:, :, None]
            e_i_cint[:, :, 0] *= -1
            self.fake_stats.append(e_i_mean)
            self.fake_stats.append(e_i_cint)

    def radial_std(self, data_norm, r, e_init_min, e_init_max):
        # get data samples with required energy
        idx = np.where((self.e_init >= e_init_min) * (self.e_init < e_init_max))[0]
        data_norm = data_norm[idx]
        # get the mask of radius r square boundary
        mask = np.zeros(data_norm.shape[1:], dtype=bool)
        mask[self.mid-r:self.mid+r, self.mid-r] = 1
        mask[self.mid-r:self.mid+r, self.mid+r] = 1
        mask[self.mid-r, self.mid-r+1:self.mid+r-1] = 1
        mask[self.mid+r, self.mid-r+1:self.mid+r-1] = 1
        # expand to the data size
        mask = mask[None].repeat(data_norm.shape[0], 0)
        return data_norm[mask].std()

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
        x = (self.e_bins[1:] + self.e_bins[:-1])/2.
        f, ax = plt.subplots(6, 2, figsize=(12, 32))
        # hist_e_init
        ax[0, 0].set_title('WGAN loss')
        ax[0, 0].plot(np.arange(0, len(self.train_loss)*self.step, self.step)+self.step, self.train_loss)
        ax[0, 0].plot(np.arange(0, len(self.train_loss)*self.step, self.step)+self.step, self.val_loss)
        ax[0, 0].legend(['train', 'val'])
        ax[0, 0].set_xlabel('epoch')
        ax[0, 0].set_ylabel('loss')
        ax[0, 1].set_title('Distribution of E_init')
        ax[0, 1].plot(x, self.real_stats[0])
        ax[0, 1].plot(x, self.fake_stats[0])
        ax[0, 1].legend(['real', 'fake'])
        ax[0, 1].set_xlabel('initial energy')
        ax[0, 1].set_ylabel('p')
        offset = 1
        # e_calo_norm_std
        for i in range(self.real_stats[1].shape[1]):
            ax[offset+i//2, i%2].set_title('Std of E_calo/E_init in square boundary, r = %d' % (i+1))
            ax[offset+i//2, i%2].plot(x, self.real_stats[1][:, i])
            ax[offset+i//2, i%2].plot(x, self.fake_stats[1][:, i])
            ax[offset+i//2, i%2].legend(['real', 'fake'])
            ax[offset+i//2, i%2].set_xlabel('initial energy')
            ax[offset+i//2, i%2].set_ylabel('std')
        offset += self.real_stats[1].shape[1]//2
        # e_i
        for i in range(self.real_stats[2].shape[1]):
            ax[offset+i, 0].set_title('Mean of E%d/E0' % (i+1))
            ax[offset+i, 0].plot(x, self.real_stats[2][:, i])
            ax[offset+i, 0].plot(x, self.fake_stats[2][:, i])
            ax[offset+i, 0].legend(['real', 'fake'])
            ax[offset+i, 0].set_xlabel('initial energy')
            ax[offset+i, 0].set_ylabel('mean')
            ax[offset+i, 1].set_title('Difference between real and fake mean E%d/E0' % (i+1))
            ax[offset+i, 1].errorbar(x, self.real_stats[2][:, i]-self.fake_stats[2][:, i], self.fake_stats[3][:, i].T)
            ax[offset+i, 1].set_xlabel('initial energy')
            ax[offset+i, 1].set_ylabel('diff')
        return f