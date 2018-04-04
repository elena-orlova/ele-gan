import torch
from torch import nn
from torch.autograd import Variable, grad



class GANLoss(nn.Module):

    def __init__(self, opt, dis):
        super(GANLoss, self).__init__()

        self.adv_loss_type = opt.adv_loss_type
        self.dis = dis

        # Set target labels for GAN or LSGAN
        if self.adv_loss_type == 'gan' or self.adv_loss_type == 'lsgan':

            # Set target labels for each dis output
            self.real_label = Variable(torch.ones(size).cuda(opt.gpu_id))
            self.fake_label = Variable(torch.zeros(size).cuda(opt.gpu_id))

        # Set criterion for GAN or LSGAN
        if self.adv_loss_type == 'gan':
            self.criterion = nn.BCEWithLogitsLoss()
        elif self.adv_loss_type == 'lsgan':
            self.criterion = nn.MSELoss()

        self.gpu_id = opt.gpu_id

    def calc_gradient_penalty(self, real, fake):

        alpha = torch.rand(real.data.size(0), 1, 1, 1)
        alpha = alpha.expand(real.data.size())
        alpha = alpha.cuda(self.gpu_id)

        interpolates = alpha * real.data + ((1 - alpha) * fake.data)
        interpolates = interpolates.cuda(self.gpu_id)
        interpolates = Variable(interpolates, requires_grad=True)

        disc_interpolates = self.dis(interpolates)

        gradients = grad(outputs=disc_interpolates, inputs=interpolates,
                         grad_outputs=torch.ones(disc_interpolates.size()).cuda(self.gpu_id),
                         create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradients = gradients.view(gradients.size(0), -1)

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10

        return gradient_penalty

    def __call__(self, real, fake=None):

        pred_real = self.dis(real)

        if fake is not None:
            pred_fake = self.dis(fake.detach())

        loss = 0
        loss_real = 0
        loss_fake = 0

        if self.adv_loss_type == 'gan' or self.adv_loss_type == 'lsgan':

            # Calculate loss for real
            loss = self.criterion(pred_real, self.real_label)

            # Calculate loss for fake
            if fake is not None:
                loss += self.criterion(pred_fake, self.fake_label)
                loss *= 0.5

            # Loss to print in logs
            display_loss = loss

        elif self.adv_loss_type == 'wgan':

            loss = -pred_real.mean()
            display_loss = loss

            if fake is not None:
                loss += pred_fake.mean()
                display_loss = -loss.clone()
                loss += self.calc_gradient_penalty(real, fake)

        return loss, display_loss
