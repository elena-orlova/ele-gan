import importlib
import torch
import numpy as np
from torch.autograd import Variable
from torch.optim import Adam
from src.dataset import get_dataloader
from torch.utils.data import DataLoader
from src.stats import Stats
from matplotlib import pyplot as plt
from torch import nn
import argparse
import os



parser = argparse.ArgumentParser(conflict_handler='resolve')
parser.add = parser.add_argument

# Dataset opts
parser.add('--train_data_path', default='',
           help='path to .npz with train data')
parser.add('--val_data_path', default='',
           help='path to .npz with val data')

parser.add('--input_idx', type=str, default='2',
           help='inputs to use during training')
parser.add('--target_boxcox', action='store_true', default=False,
           help='preprocess target using boxcox transform')
parser.add('--num_workers', type=int,
           help='number of data loading workers', default=6)
parser.add('--batch_size', type=int,
           default=64, help='batch size')
parser.add('--image_size', type=int, default=16,
           help='size of the calorimeter image')

# Training opts
parser.add('--gpu_id', type=int, default=6)
parser.add('--num_epoch', type=int, default=100,
           help='number of epochs to train for')
parser.add('--lr', type=float, default=0.0001,
           help='learning rate, default=0.0001')
parser.add('--beta1', type=float, default=0.9,
           help='beta1 for adam. default=0.9')
parser.add('--manual_seed', type=int, default=123, help='manual seed')
parser.add('--experiment_name', default='')
parser.add('--experiments_dir', default='./data/experiments/',
           help='folder to output images and model checkpoints')
parser.add('--val_every_epoch', default=1, type=int)

# Model opts
parser.add('--model_type', type=str, default='dcgan')
parser.add('--adv_loss_type', default='wgan', type=str)
parser.add('--regressor_path', type=str, default='data/utils/regression.model')

# Shared opts between generator and discriminator
parser.add('--num_channels', default=64, type=int)
parser.add('--max_channels', default=256, type=int)

# Generator opts
parser.add('--in_channels', default=4, type=int)
parser.add('--latent_size', default=4, type=int)
parser.add('--nonlinearity', default='relu', type=str)
parser.add('--norm', default='batch', type=str)

# Discriminator opts
parser.add('--kernel_size', default=3, type=int)
parser.add('--num_preds', default=1, type=int)
parser.add('--norm_dis', action='store_true', default=False)

opt, _ = parser.parse_known_args()
opt.adv_loss_type = opt.adv_loss_type.lower()

path = opt.experiments_dir + opt.experiment_name

def save_checkpoint(state):
    torch.save(state, path + '/checkpoints/%d.pkl' % state['epoch'])

# Model opts
m = importlib.import_module('models.' + opt.model_type)
m.get_args(parser)

opt, _ = parser.parse_known_args()

print(opt)

(dataset_train,
 dataloader_train, 
 dataset_val,
 dataloader_val) = get_dataloader(opt)

transform_target = dataset_train.transform_target

model = m.Model(opt)

optimizer_G = Adam(model.gen_params, lr=opt.lr, betas=(opt.beta1, 0.999))
optimizer_D = Adam(model.dis_params, lr=opt.lr, betas=(opt.beta1, 0.999))

stats = Stats(opt)
stats.calc_stats(dataset_val.targets, True)

losses = []

# Make directories
if not os.path.exists(opt.experiments_dir):
    os.makedirs(path)
else:
    if not os.path.exists(path + '/checkpoints'):
        os.makedirs(path + '/checkpoints')
    if not os.path.exists(path + '/figures'):
        os.makedirs(path + '/figures')

# Save options
file_name = path + '/opt.txt'
with open(file_name, 'wt') as opt_file:
    for k, v in sorted(vars(opt).items()):
        opt_file.write('%s: %s\n' % (str(k), str(v)))

for epoch in range(1, opt.num_epoch + 1):

    model.train()
    loss_adv_train = []
    loss_mse_train = []

    for i, inputs in enumerate(dataloader_train, 1):

        model.forward(inputs)

        if (opt.model_type == 'dcgan' and 
            (opt.adv_loss_type == 'gan' or
             opt.adv_loss_type == 'wgan' and not i % 5)):

            for p in model.dis_params:
                p.requires_grad = False

            optimizer_G.zero_grad()
            model.backward_G()
            optimizer_G.step()

            for p in model.dis_params:
                p.requires_grad = True

            loss_mse_train += [model.loss_mse.data[0]]

        optimizer_D.zero_grad()
        model.backward_D()
        optimizer_D.step()

        loss_adv_train += [model.loss_adv.data[0]]

    loss_adv_train = sum(loss_adv_train) / len(loss_adv_train)
    loss_mse_train = sum(loss_mse_train) / len(loss_mse_train)

    if epoch % opt.val_every_epoch:
        continue

    model.eval()

    loss_adv_val = 0
    loss_mse_val = 0
    val_data_fake = []

    stats.train_dis_adv_loss.append(loss_adv_train)

    for i, inputs in enumerate(dataloader_val, 1):

        model.forward(inputs)
        model.backward_G()
        model.backward_D()

        loss_adv_val += model.loss_adv.data[0]
        loss_mse_val += model.loss_mse.data[0]

        val_data_fake += [model.fake.data[:, 0]]

    loss_adv_val /= i
    loss_mse_val /= i

    if opt.adv_loss_type == 'wgan':
        stats.val_dis_adv_loss.append(loss_adv_val)

    val_data_fake = torch.cat(val_data_fake, 0)
    val_data_fake = transform_target(val_data_fake, 'from')

    stats.calc_stats(val_data_fake)
    f = stats.get_plot()
    f.savefig(path + '/figures/%d.png' % epoch)

    print('[{epoch}/{num_epoch}] '
          'loss_adv/loss_mse: {loss_adv:.8f}/{loss_mse:.8f} '
          ''.format(epoch=epoch,
                    num_epoch=opt.num_epoch,
                    loss_adv=model.loss_adv.data[0],
                    loss_mse=model.loss_mse.data[0]))
    losses += [(model.loss_adv.data[0],  
                model.loss_mse.data[0])]
    np.savetxt(path + '/losses.txt', losses, fmt='%.8f')

    model.save_checkpoint(epoch)