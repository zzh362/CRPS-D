"""Defines the detector network structure."""
import torch
from torch import nn
from model.network import define_halve_unit, define_detector_block
import torch.nn.functional as F
from util import ramps


class YetAnotherDarknet(nn.modules.Module):
    """Yet another darknet, imitating darknet-53 with depth of darknet-19."""
    def __init__(self, input_channel_size, depth_factor):
        super(YetAnotherDarknet, self).__init__()
        layers = []
        # 0
        layers += [nn.Conv2d(input_channel_size, depth_factor, kernel_size=3,
                             stride=1, padding=1, bias=False)]
        layers += [nn.BatchNorm2d(depth_factor)]
        layers += [nn.LeakyReLU(0.1)]
        # 1
        layers += define_halve_unit(depth_factor)
        layers += define_detector_block(depth_factor)
        # 2
        depth_factor *= 2
        layers += define_halve_unit(depth_factor)
        layers += define_detector_block(depth_factor)
        # 3
        depth_factor *= 2
        layers += define_halve_unit(depth_factor)
        layers += define_detector_block(depth_factor)
        layers += define_detector_block(depth_factor)
        # 4
        depth_factor *= 2
        layers += define_halve_unit(depth_factor)
        layers += define_detector_block(depth_factor)
        layers += define_detector_block(depth_factor)
        # 5
        depth_factor *= 2
        layers += define_halve_unit(depth_factor)
        layers += define_detector_block(depth_factor)
        self.model = nn.Sequential(*layers)

    def forward(self, *x):
        return self.model(x[0])

def _l2_normalize(d):
    # Normalizing per batch axis
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
    return d

def get_r_adv_t(x, t_model, s_model, it=1, xi=1e-1, eps=10.0):

    # stop bn
    t_model.eval()
    s_model.eval()

    x_detached = x.detach()

    with torch.no_grad():
        # pred = t_model.decoder(x_detached) * alpha + s_model.decoder(x_detached) * (1 - alpha)
        pred_t = t_model.decoder(x_detached)
        pred_s = s_model.decoder(x_detached)


    d = torch.rand(x.shape).sub(0.5).to(x.device)
    d = _l2_normalize(d)

    # assist students to find the effective va-noise
    for _ in range(it):
        d.requires_grad_()
        # pred_hat = t_model.decoder(x_detached + xi * d) * alpha + s_model.decoder(x_detached + xi * d) * (1 - alpha)
        t_pred_hat = t_model.decoder(x_detached + xi * d)
        s_pred_hat = s_model.decoder(x_detached + xi * d)
        t_mse_loss = F.mse_loss(t_pred_hat, pred_t)
        s_mse_loss = F.mse_loss(s_pred_hat, pred_s)
        if t_mse_loss <= s_mse_loss:
            t_mse_loss.backward()
        else:
            s_mse_loss.backward()
        # mse_loss.backward()
        d = _l2_normalize(d.grad)
        t_model.zero_grad()
        s_model.zero_grad()

    r_adv = d * eps
    # reopen bn, but freeze other params.
    # https://discuss.pytorch.org/t/why-is-it-when-i-call-require-grad-false-on-all-my-params-my-weights-in-the-network-would-still-update/22126/16

    t_model.train()
    s_model.train()
    
    return r_adv

# def get_r_adv_t(x, t_model, s_model, it=1, xi=1e-1, eps=10.0):

#     # stop bn
#     t_model.eval()
#     s_model.eval()

#     alpha = 1

#     x_detached = x.detach()
#     with torch.no_grad():

#         pred = t_model.decoder(x_detached) * alpha + s_model.decoder(x_detached) * (1 - alpha)

#     d = torch.rand(x.shape).sub(0.5).to(x.device)
#     d = _l2_normalize(d)

#     # assist students to find the effective va-noise
#     for _ in range(it):
#         d.requires_grad_()
#         pred_hat = t_model.decoder(x_detached + xi * d) * alpha + s_model.decoder(x_detached + xi * d) * (1 - alpha)
#         mse_loss = F.mse_loss(pred_hat, pred)
#         mse_loss.backward()
#         d = _l2_normalize(d.grad)
#         t_model.zero_grad()
#         s_model.zero_grad()

#     r_adv = d * eps

#     # reopen bn, but freeze other params.
#     # https://discuss.pytorch.org/t/why-is-it-when-i-call-require-grad-false-on-all-my-params-my-weights-in-the-network-would-still-update/22126/16

#     t_model.train()
#     s_model.train()

#     return r_adv

# def get_r_adv_t(x, t_model, s_model, it=1, xi=1e-1, eps=10.0):

#     # stop bn
#     t_model.eval()
#     s_model.eval()

#     alpha = 1

#     x_detached = x.detach()
#     with torch.no_grad():
        
#         pred = F.softmax(t_model.decoder(x_detached) * alpha + s_model.decoder(x_detached) * (1 - alpha), dim=1)

#     d = torch.rand(x.shape).sub(0.5).to(x.device)
#     d = _l2_normalize(d)

#     # assist students to find the effective va-noise
#     for _ in range(it):
#         d.requires_grad_()
#         pred_hat = t_model.decoder(x_detached + xi * d) * alpha + s_model.decoder(x_detached + xi * d) * (1 - alpha)
#         logp_hat = F.log_softmax(pred_hat, dim=1)
#         adv_distance = F.kl_div(logp_hat, pred, reduction='batchmean')
#         adv_distance.backward()
#         d = _l2_normalize(d.grad)
#         t_model.zero_grad()
#         s_model.zero_grad()

#     r_adv = d * eps

#     # reopen bn, but freeze other params.
#     # https://discuss.pytorch.org/t/why-is-it-when-i-call-require-grad-false-on-all-my-params-my-weights-in-the-network-would-still-update/22126/16

#     t_model.train()
#     s_model.train()

#     return r_adv

class TeacherDetector(nn.modules.Module):
    """Detector for point with direction."""
    def __init__(self, input_channel_size, depth_factor, output_channel_size):
        super(TeacherDetector, self).__init__()
        self.encoder = YetAnotherDarknet(input_channel_size,
                                                 depth_factor)
        layers = []
        layers += define_detector_block(16 * depth_factor)
        layers += define_detector_block(16 * depth_factor)
        layers += [nn.Conv2d(32 * depth_factor, output_channel_size,
                             kernel_size=1, stride=1, padding=0, bias=False)]

        self.decoder = nn.Sequential(*layers)

    def forward(self, *x):
        prediction = self.decoder(self.encoder(x[0]))
        # 4 represents that there are 4 value: confidence, shape, offset_x,
        # offset_y, whose range is between [0, 1].
        point_pred, angle_pred, type_pred = torch.split(prediction, 4, dim=1)
        point_pred = torch.sigmoid(point_pred)
        angle_pred = torch.tanh(angle_pred)
        type_pred = torch.sigmoid(type_pred)
        return torch.cat((point_pred, angle_pred, type_pred), dim=1)

class StudentDetector(nn.modules.Module):
    """Detector for point with direction."""
    def __init__(self, input_channel_size, depth_factor, output_channel_size):
        super(StudentDetector, self).__init__()
        self.encoder = YetAnotherDarknet(input_channel_size,
                                                 depth_factor)
        layers = []
        layers += define_detector_block(16 * depth_factor)
        layers += define_detector_block(16 * depth_factor)
        layers += [nn.Conv2d(32 * depth_factor, output_channel_size,
                             kernel_size=1, stride=1, padding=0, bias=False)]

        self.decoder = nn.Sequential(*layers)

    def forward(self, *x, t_model=None, s_model=None):
        f = self.encoder(x[0])
        if t_model is not None:
            # eps_n = 10 * ramps.sigmoid_rampup(epoch, 10)
            r_adv = get_r_adv_t(f, t_model, s_model, it=1, xi=1e-6, eps=10.0)
            f += r_adv
        prediction = self.decoder(f)

        point_pred, angle_pred, type_pred = torch.split(prediction, 4, dim=1)
        point_pred = torch.sigmoid(point_pred)
        angle_pred = torch.tanh(angle_pred)
        type_pred = torch.sigmoid(type_pred)
        return torch.cat((point_pred, angle_pred, type_pred), dim=1)
    
class DirectionalPointDetector(nn.modules.Module):
    """Detector for point with direction."""
    def __init__(self, input_channel_size, depth_factor, output_channel_size):
        super(DirectionalPointDetector, self).__init__()
        self.extract_feature = YetAnotherDarknet(input_channel_size,
                                                 depth_factor)
        layers = []
        layers += define_detector_block(16 * depth_factor)
        layers += define_detector_block(16 * depth_factor)
        layers += [nn.Conv2d(32 * depth_factor, output_channel_size,
                             kernel_size=1, stride=1, padding=0, bias=False)]
        self.predict = nn.Sequential(*layers)

    def forward(self, *x):
        prediction = self.predict(self.extract_feature(x[0]))
        # 4 represents that there are 4 value: confidence, shape, offset_x,
        # offset_y, whose range is between [0, 1].
        point_pred, angle_pred, type_pred = torch.split(prediction, 4, dim=1)
        point_pred = torch.sigmoid(point_pred)
        angle_pred = torch.tanh(angle_pred)
        type_pred = torch.sigmoid(type_pred)
        return torch.cat((point_pred, angle_pred, type_pred), dim=1)
