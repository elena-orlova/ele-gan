import torch
from torch import nn
from torch.nn.functional import l1_loss
from .gan_loss import GANLoss
from models.generator import Generator
from models.discriminator import Discriminator
from itertools import chain
from torch.autograd import Variable



def get_args(parser):

    parser.add('--mse_weight', default=10., type=float)


class Model(object):

    def __init__(self, opt):
        super(Model, self).__init__()

        # Generator
        self.gen = Generator(opt).cuda(opt.gpu_id)

        self.gen_params = self.gen.parameters()

        num_params = 0
        for p in self.gen.parameters():
            num_params += p.numel()
        print(self.gen)
        print(num_params)

        # Discriminator
        self.dis = Discriminator(opt).cuda(opt.gpu_id)

        self.dis_params = self.dis.parameters()

        num_params = 0
        for p in self.dis.parameters():
            num_params += p.numel()
        print(self.dis)
        print(num_params)

        # Regressor
        if opt.mse_weight:
            self.reg = torch.load('data/utils/classifier.pth').cuda(opt.gpu_id).eval()
        else:
            self.reg = None

        # Losses
        self.criterion_gan = GANLoss(opt, self.dis)
        self.criterion_mse = lambda x, y: l1_loss(x, y) * opt.mse_weight

        self.loss_mse = Variable(torch.zeros(1).cuda())
        self.loss_adv = Variable(torch.zeros(1).cuda())
        self.loss = Variable(torch.zeros(1).cuda())

        self.path = opt.experiments_dir + opt.experiment_name + '/checkpoints/'
        self.gpu_id = opt.gpu_id
        self.noise_channels = opt.in_channels - len(opt.input_idx.split(','))

    def forward(self, inputs):

        input, input_orig, target = inputs

        self.input = Variable(input.cuda(self.gpu_id))
        self.input_orig = Variable(input_orig.cuda(self.gpu_id))
        self.target = Variable(target.cuda(self.gpu_id))

        noise = Variable(torch.randn(self.input.size(0), self.noise_channels).cuda(self.gpu_id))

        self.fake = self.gen(torch.cat([self.input, noise], 1))

    def backward_G(self):

        # Regressor loss
        if self.reg is not None:

            fake_input = self.reg(self.fake)

            self.loss_mse = self.criterion_mse(fake_input, self.input_orig)

        # GAN loss
        loss_adv, _ = self.criterion_gan(self.fake)

        loss_G = self.loss_mse + loss_adv
        loss_G.backward()

    def backward_D(self):

        loss_adv, self.loss_adv = self.criterion_gan(self.target, self.fake)

        loss_D = loss_adv
        loss_D.backward()

    def train(self):

        self.gen.train()
        self.dis.train()

    def eval(self):

        self.gen.eval()
        self.dis.eval()

    def save_checkpoint(self, epoch):

        torch.save({
            'epoch': epoch,
            'gen_state_dict': self.gen.state_dict(),
            'dis_state_dict': self.dis.state_dict()},
            self.path + '%d.pkl' % epoch)
        
    def load_checkpoint(self, path, pretrained=True):

        weights = torch.load(path)

        self.gen.load_state_dict(weights['gen_state_dict'])
        self.dis.load_state_dict(weights['dis_state_dict'])
