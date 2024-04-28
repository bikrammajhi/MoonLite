import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps2 = eps ** 2

    def forward(self, inp, target):
        return ((nn.functional.mse_loss(inp, target, reduction='none') + self.eps2) ** .5).mean()


class OutlierAwareLoss(nn.Module):
    def __init__(self, kernel_size=False):
        super(OutlierAwareLoss, self).__init__()
        self.unfold = torch.nn.Unfold(kernel_size)
        self.kernel = kernel_size

    def forward(self, out, lab):
        b, c, h, w = out.shape
        p = self.kernel // 2
        delta = out - lab
        if self.kernel:
            delta_ = torch.nn.functional.pad(delta, (p, p, p, p))
            patch = self.unfold(delta_).reshape(b, c,
                                                self.kernel, self.kernel,
                                                h, w).detach()
            var = patch.std((2, 3)) / (2 ** .5)
            avg = patch.mean((2, 3))
        else:
            var = delta.std((2, 3), keepdims=True) / (2 ** .5)
            avg = delta.mean((2, 3), True)
        weight = 1 - (-((delta - avg).abs() / var)).exp().detach()
        # weight = 1 - (-1 / var.detach()-1/ delta.abs().detach()).exp()
        loss = (delta.abs() * weight).mean()
        return loss


class LossWarmup(nn.Module):
    def __init__(self):
        super(LossWarmup, self).__init__()
        self.loss_cb = CharbonnierLoss(1e-8)
        self.loss_cs = nn.CosineSimilarity()

    def forward(self, inp, gt, warmup1, warmup2):
        loss = self.loss_cb(warmup2, inp) + \
               (self.loss_cb(warmup1, gt) + (1 - self.loss_cs(warmup1.clip(0, 1), gt)).mean())
        return loss


class LossISP(nn.Module):
    def __init__(self):
        super(LossISP, self).__init__()
        self.loss_cs = nn.CosineSimilarity()
        self.loss_oa = OutlierAwareLoss()

    def forward(self, out, gt):
        loss = (self.loss_oa(out, gt) + (1 - self.loss_cs(out.clip(0, 1), gt)).mean())
        return loss


class LossLLE(nn.Module):
    def __init__(self):
        super(LossLLE, self).__init__()
        self.loss_cs = nn.CosineSimilarity()
        self.loss_oa = OutlierAwareLoss()

    def forward(self, out, gt):
        loss = (self.loss_oa(out, gt) + (1 - self.loss_cs(out.clip(0, 1), gt)).mean())
        return loss


class LossSR(nn.Module):
    def __init__(self):
        super(LossSR, self).__init__()
        self.loss_oa = OutlierAwareLoss()

    def forward(self, out, gt):
        loss = self.loss_oa(out, gt)
        return loss


def import_loss(training_task):
    if training_task == 'isp':
        return LossISP()
    elif training_task == 'lle':
        return LossLLE()
    elif training_task == 'sr':
        return LossSR()
    elif training_task == 'warmup':
        return LossWarmup()
    else:
        raise ValueError('unknown training task, please choose from [isp, lle, sr, warmup].')



# Load the pre-trained VGG model
vgg = models.vgg19(pretrained=True).features.eval()

# Define the layers for the VGG perceptual loss
vgg_loss_layers = {
    'conv1_2': vgg[:6],    # First convolutional layer
    'conv2_2': vgg[6:13],  # Second convolutional layer
    'conv3_3': vgg[13:26], # Third convolutional layer
    'conv4_3': vgg[26:39]  # Fourth convolutional layer
}

# Custom perceptual loss function using VGG features
class VGGPerceptualLoss(nn.Module):
    def __init__(self, layers):
        super(VGGPerceptualLoss, self).__init__()
        self.layers = layers
        self.criterion = nn.MSELoss()

    def forward(self, x, y):
        loss = 0.0
        for layer_name, layer in self.layers.items():
            x_feat = layer(x)
            y_feat = layer(y)
            loss += self.criterion(x_feat, y_feat)
        return loss
