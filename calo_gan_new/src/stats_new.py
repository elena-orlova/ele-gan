import numpy as np
import xgboost as xgb
import statsmodels.stats.api as sms
from matplotlib import pyplot as plt
import torch
from torch.autograd import Variable



class Stats(object):
    """ Calculate stats over original data """
    
    def __init__(self, opt, target_min, target_max):
        super(Stats, self).__init__()

        target_idx = opt.target_idx.split(',')
        self.regressors = [xgb.Booster({'nthread': 4})] * len(target_idx)
        for reg, i in zip(self.regressors, target_idx):
            reg.load_model('data/utils/regressor_%s.model' % i)
        self.regressor_size = opt.regressor_size

        self.n_bins = opt.stats_bins # for histograms w.r.t. initial momentum
        self.radius = opt.stats_radius # max radius for radial stats
        self.mid = opt.image_size//2
        self.target_min = target_min
        self.target_max = target_max

        # Masks of the circles in L_inf norm with radiuses r
        self.masks = []

        for r in range(1, self.radius+1):

            mask = np.zeros((opt.image_size, opt.image_size), dtype=int)
            
            mask[self.mid-r:self.mid+r, self.mid-r] = 1
            mask[self.mid-r:self.mid+r, self.mid+r] = 1
            mask[self.mid-r, self.mid-r+1:self.mid+r-1] = 1
            mask[self.mid+r, self.mid-r+1:self.mid+r-1] = 1
            
            self.masks.append(mask)

    def calc_stats(self, input):

        stats = []

        # Predict initial values
        rad = self.regressor_size//2
        l, r = self.mid-rad, self.mid+rad

        input = xgb.DMatrix(input[:, l:r, l:r].reshape(input.shape[0], -1))
        target = []

        for reg in self.regressor:
            target += [reg.predict(input)[None]]
        target = np.concatenate(target, 1)

        # Calculate histogram over initial values
        for i in range(target.shape[1]):
            


    def __call__(self, input_real, input_fake):

        pass