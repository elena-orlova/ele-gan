import torch
from torch import nn
from torch.autograd import Variable, grad



class Loss(nn.Module):
    def __init__(self, opt, dis, stats):
        super(Loss, self).__init__()
        self.dis = dis
        if opt.adv_loss_type == 'gan':
            self.label_0 = Variable(torch.zeros(opt.batch_size).cuda())
            self.label_1 = Variable(torch.ones(opt.batch_size).cuda())
            self.crit_adv = nn.BCEWithLogitsLoss()
        else:
            self.crit_adv = None
        if opt.num_pred == 10:
            self.stats = stats
            self.crit_stats = nn.MSELoss()
            self.stats_weight = opt.stats_weight
        else:
            self.crit_stats = None
            self.stats_weight = 0

    def calc_gradient_penalty(self, real_data, fake_data, weight=10):
        b, c, h, w = real_data.size()
        alpha = torch.rand(b, 1)
        alpha = alpha.expand(b, real_data.nelement()//b).contiguous().view(b, c, h, w)
        alpha = alpha.cuda()

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)
        interpolates = interpolates.cuda()
        interpolates = Variable(interpolates, requires_grad=True)

        disc_interpolates = self.dis(interpolates)

        gradients = grad(outputs=disc_interpolates, inputs=interpolates,
                         grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                         create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradients = gradients.view(gradients.size(0), -1)

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * weight
        return gradient_penalty

    def forward(self, real_input, fake_input=None):
        if self.crit_stats is not None:
            real, real_stats = real_input
            if fake_input is not None:
                fake, fake_stats = fake_input
        else:
            real = real_input[0]
            if fake_input is not None:
                fake = fake_input[0]
        output_real = self.dis(real)
        pred_real = output_real[:, 0]
        loss, loss_adv, loss_stats = 0, 0, 0
        if fake_input is not None:
            output_fake = self.dis(fake.detach())
            pred_fake = output_fake[:, 0]
        if self.crit_adv is not None:
            # gan objective
            loss_adv += self.crit_adv(pred_real, self.label_1)
            if fake_input is not None:
                loss_adv += self.crit_adv(pred_fake, self.label_0)
                loss_adv *= 0.5
        else:
            # wgan objective
            loss_adv -= pred_real.mean()
            if fake_input is not None:
                loss_adv += pred_fake.mean()
                loss += self.calc_gradient_penalty(real.data, fake.data)
        # calculate aux objective
        if self.crit_stats is not None:
            loss_stats += self.crit_stats(output_real[:, 1:], real_stats)
            if fake_input is not None:
                loss_stats += self.crit_stats(output_fake[:, 1:], fake_stats)
                loss_stats *= 0.5
        loss += loss_adv + loss_stats * self.stats_weight
        return loss, loss_adv, loss_stats