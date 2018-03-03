import torch
from torch.autograd import Variable, grad



def calc_gradient_penalty(netD, real_data, fake_data, weight=10):
    b, c, h, w = real_data.size()
    alpha = torch.rand(b, 1)
    alpha = alpha.expand(b, real_data.nelement()//b).contiguous().view(b, c, h, w)
    alpha = alpha.cuda()

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates = interpolates.cuda()
    interpolates = Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = grad(outputs=disc_interpolates, inputs=interpolates,
                     grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                     create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * weight
    return gradient_penalty