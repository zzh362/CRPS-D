"""Train directional marking point detector."""
import math
import random
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import config
import data
import yaml
import util
from model import StudentDetector, TeacherDetector
from model.effnetv2 import effnetv2_base
from util import ramps
import torch.nn.functional as F
import numpy as np

ema_decay = 0.99
consistency_rampup = 5
consistency = 100

def plot_prediction(logger, image, marking_points, prediction):
    """Plot the ground truth and prediction of a random sample in a batch."""
    rand_sample = random.randint(0, image.size(0)-1)
    sampled_image = util.tensor2im(image[rand_sample])
    logger.plot_marking_points(sampled_image, marking_points[rand_sample],
                               win_name='gt_marking_points')
    sampled_image = util.tensor2im(image[rand_sample])
    pred_points = data.get_predicted_points(prediction[rand_sample], 0.01)
    if pred_points:
        logger.plot_marking_points(sampled_image,
                                   list(list(zip(*pred_points))[1]),
                                   win_name='pred_marking_points')

def generate_objective(marking_points_batch, device, labeled_bs, batch_size, teacher_con, is512 = True):
    """Get regression objective and gradient for directional point detector."""
    # batch_size = 24
    size = 16 if is512 else 15#30
    objective = torch.zeros(batch_size, config.NUM_FEATURE_MAP_CHANNEL,
                            size, size,
                            device=device)
    
    gradient = torch.zeros(labeled_bs, config.NUM_FEATURE_MAP_CHANNEL,
                            size, size,
                            device=device)
    
    gradient_c = torch.zeros((batch_size - labeled_bs), config.NUM_FEATURE_MAP_CHANNEL,
                            size, size,
                            device=device)
    
    gradient_c[:, 0].fill_(1.)

    teacher_con_cp = teacher_con.clone()
    for h in range(0, 16):
        for w in range(0, 16):
            indices = torch.where(teacher_con[:, 0, h, w] < 0.9)[0]
            teacher_con_cp[indices, 0, h, w] = 0

    for i in range(batch_size - labeled_bs):
        for j in range(1, 9):
            gradient_c[i, j, :, :] = teacher_con_cp[i, 0, :, :]

    
    gradient[:, 0].fill_(1.)
    for batch_idx, marking_points in enumerate(marking_points_batch):
        for marking_point in marking_points:
            col = math.floor(marking_point.x * size)
            row = math.floor(marking_point.y * size)
            # Confidence Regression
            objective[batch_idx, 0, row, col] = 1.
            # Makring Point Shape Regression
            objective[batch_idx, 1, row, col] = marking_point.shape
            # Offset Regression
            objective[batch_idx, 2, row, col] = marking_point.x*size - col
            objective[batch_idx, 3, row, col] = marking_point.y*size - row
            # Direction Regression
            direction0 = marking_point.direction0
            objective[batch_idx, 4, row, col] = (math.cos(direction0) +1)/2
            objective[batch_idx, 5, row, col] = (math.sin(direction0) +1)/2
            direction1 = marking_point.direction1
            objective[batch_idx, 6, row, col] = math.cos(direction1)/2 +0.5
            objective[batch_idx, 7, row, col] = math.sin(direction1)/2 + 0.5
            # Marking Point Type Regression
            objective[batch_idx, 8, row, col] = marking_point.type
            # Assign Gradient
            gradient[batch_idx, 1:9, row, col].fill_(2.)
            gradient[batch_idx, 4:8, row, col].fill_(6.)

    gradient_all = torch.cat((gradient, gradient_c), dim=0)

    return objective, gradient_all

# def generate_objective(marking_points_batch, device, batch_size, is512 = True):
#     """Get regression objective and gradient for directional point detector."""
#     # batch_size = 24
#     size = 16 if is512 else 15#30
#     objective = torch.zeros(batch_size, config.NUM_FEATURE_MAP_CHANNEL,
#                             size, size,
#                             device=device)
#     gradient = torch.zeros_like(objective)
#     gradient[:, 0].fill_(1.)
#     for batch_idx, marking_points in enumerate(marking_points_batch):
#         for marking_point in marking_points:
#             col = math.floor(marking_point.x * size)
#             row = math.floor(marking_point.y * size)
#             # Confidence Regression
#             objective[batch_idx, 0, row, col] = 1.
#             # Makring Point Shape Regression
#             objective[batch_idx, 1, row, col] = marking_point.shape
#             # Offset Regression
#             objective[batch_idx, 2, row, col] = marking_point.x*size - col
#             objective[batch_idx, 3, row, col] = marking_point.y*size - row
#             # Direction Regression
#             direction0 = marking_point.direction0
#             objective[batch_idx, 4, row, col] = (math.cos(direction0) +1)/2
#             objective[batch_idx, 5, row, col] = (math.sin(direction0) +1)/2
#             direction1 = marking_point.direction1
#             objective[batch_idx, 6, row, col] = math.cos(direction1)/2 +0.5
#             objective[batch_idx, 7, row, col] = math.sin(direction1)/2 + 0.5
#             # Marking Point Type Regression
#             objective[batch_idx, 8, row, col] = marking_point.type
#             # Assign Gradient
#             gradient[batch_idx, 1:9, row, col].fill_(2.)
#             gradient[batch_idx, 4:8, row, col].fill_(6.)
#     return objective, gradient


def get_loss(pred, target):

    assert pred.shape == target.shape, "预测值和目标值形状不匹配"
    
    diff = pred - target
    loss = diff ** 2

    return loss

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return consistency * ramps.sigmoid_rampup(epoch, consistency_rampup)

def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def train_detector(args):
    """Train directional point detector."""
    global_step = 0

    args.cuda = not args.disable_cuda and torch.cuda.is_available()
    device = torch.device('cuda:' + str('0') if args.cuda else 'cpu')
    torch.set_grad_enabled(True)

    model_stu = StudentDetector(
        3, args.depth_factor, config.NUM_FEATURE_MAP_CHANNEL)
    model_tea = TeacherDetector(
        3, args.depth_factor, config.NUM_FEATURE_MAP_CHANNEL)
    
    
    model_stu.to(device)
    model_tea.to(device)
    for param in model_tea.parameters():
        param.detach_()

    # if args.detector_weights:
    #     print("Loading weights: %s" % args.detector_weights)
    #     dp_detector.load_state_dict(torch.load(args.detector_weights))

    model_stu.train()
    model_tea.train()

    optimizer = torch.optim.Adam(model_stu.parameters(), 1e-4)
    # optimizer = torch.optim.SGD(dp_detector.parameters(), 1e-4)

    # if args.optimizer_weights:
    #     print("Loading weights: %s" % args.optimizer_weights)
    #     optimizer.load_state_dict(torch.load(args.optimizer_weights))

    logger = util.Logger(args.enable_visdom, ['train_loss'])

    file_path = 'yaml/data_root.yaml'

    with open(file_path, 'r') as file:
        yaml_data = yaml.safe_load(file)

    # data_root = yaml_data['data_root']
    # point_train = yaml_data['point_train']
    # path = data_root + point_train

    path_label = yaml_data['with_label_12']
    path_nolabel = yaml_data['without_label_12']

    
    data_loader_label = DataLoader(data.ParkingSlotDatasetWithLabel(path_label),
                             batch_size=args.batch_size_label, shuffle=True,
                             num_workers=args.data_loading_workers,
                             collate_fn=lambda x: list(zip(*x)))

    data_loader_nolabel = DataLoader(data.ParkingSlotDatasetWithoutLabel(path_nolabel),
                             batch_size=args.batch_size_nolabel, shuffle=True,
                             num_workers=args.data_loading_workers,
                             collate_fn=lambda x: list(zip(*x)))
    
    model = model_stu
    ema_model = model_tea

    for epoch_idx in range(args.num_epochs):
        
        for iter_idx, ((images, marking_points), (ema_images_week, ema_images_strong)) in enumerate(zip(data_loader_label, data_loader_nolabel)):

            ema_images_week = torch.stack(ema_images_week).to(device)
            ema_images_strong = torch.stack(ema_images_strong).to(device)
            images = torch.stack(images).to(device)

            labeled_bs = len(images)
            images = torch.cat((images, ema_images_strong), dim=0)

            optimizer.zero_grad()

            ema_model_out = ema_model(ema_images_week) # 无标签数据输入教师模型
            model_out = model(images, t_model=ema_model, s_model=model)   


            objective, gradient = generate_objective(marking_points, device, labeled_bs, len(images), ema_model_out[:, 0:1, :, :], True)

            # print(gradient)
            # p

            supervised_loss = get_loss(model_out[:labeled_bs], objective[:labeled_bs]) # 有监督损失

            consistency_loss = get_loss(model_out[labeled_bs:],ema_model_out) # 无监督损失

            # loss = supervised_loss + consistency_loss
            loss = torch.cat((supervised_loss, consistency_loss), dim=0)

            loss.backward(gradient)
            optimizer.step()

            global_step += 1
            update_ema_variables(model, ema_model, ema_decay, global_step)  # teacher 模型的更新

            train_loss_all = torch.sum(loss*gradient).item() / loss.size(0)
            train_supervised_loss = torch.sum(supervised_loss*gradient[:labeled_bs]).item() / supervised_loss.size(0)
            train_consistency_loss = torch.sum(consistency_loss*gradient[labeled_bs:]).item() / consistency_loss.size(0)

            logger.log(epoch=epoch_idx, 
                       iter=iter_idx, 
                       train_supervised_loss=train_supervised_loss,
                       train_consistency_loss=train_consistency_loss,
                       train_loss_all=train_loss_all)

        
        # torch.save(model.state_dict(),
        #            'weights_stu/stu_%d.pth' % (epoch_idx))
        torch.save(ema_model.state_dict(),
                   'weights_eps10/tea_%d.pth' % (epoch_idx))
        
        # torch.save(optimizer.state_dict(), 'weights/optimizer.pth')

import traceback
if __name__ == '__main__':
    same_seeds(114514)
    try:
        train_detector(config.get_parser_for_training().parse_args())
    except Exception as e:
        fp = open('log.txt', 'a')
        traceback.print_exc(file=fp)
        fp.close()