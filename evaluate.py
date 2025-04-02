"""Evaluate directional marking point detector."""
import torch
import config
import util
import os
import yaml
from thop import profile
from data import get_predicted_points, match_marking_points, calc_point_squre_dist, calc_point_direction0_angle, calc_point_direction1_angle
from data import ParkingSlotDataset
from model import TeacherDetector, DirectionalPointDetector
from torch import nn
from model.effnetv2 import effnetv2_base
import math


def generate_objective(marking_points_batch, device, batch_size, is512 = True):
    """Get regression objective and gradient for directional point detector."""
    # batch_size = 24
    size = 16 if is512 else 15#30
    objective = torch.zeros(batch_size, config.NUM_FEATURE_MAP_CHANNEL,
                            size, size,
                            device=device)
    gradient = torch.zeros_like(objective)
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
    return objective, gradient

def is_gt_and_pred_matched(ground_truths, predictions, thresh):
    """Check if there is any false positive or false negative."""
    predictions = [pred for pred in predictions if pred[0] >= thresh]
    prediction_matched = [False] * len(predictions)
    for ground_truth in ground_truths:
        idx = util.match_gt_with_preds(ground_truth, predictions,
                                       match_marking_points)
        if idx < 0:
            return False
        prediction_matched[idx] = True
    if not all(prediction_matched):
        return False
    return True


def collect_error(ground_truths, predictions, thresh):
    """Collect errors for those correctly detected points."""
    dists = []
    angles = []
    predictions = [pred for pred in predictions if pred[0] >= thresh]
    for ground_truth in ground_truths:
        idx = util.match_gt_with_preds(ground_truth, predictions,
                                       match_marking_points)
        if idx >= 0:
            detected_point = predictions[idx][1]
            dists.append(calc_point_squre_dist(detected_point, ground_truth))
            angles.append(calc_point_direction0_angle(detected_point, ground_truth)
                          + calc_point_direction1_angle(detected_point, ground_truth))
        else:
            continue
    return dists, angles



def evaluate_detector(args, model_path=''):
    """Evaluate directional point detector."""
    args.cuda = not args.disable_cuda and torch.cuda.is_available()
    device = torch.device('cuda:' + str('0') if args.cuda else 'cpu')
    torch.set_grad_enabled(False)

    # dp_detector = effnetv2_base().to(device)
    dp_detector = TeacherDetector(
        3, args.depth_factor, config.NUM_FEATURE_MAP_CHANNEL).to(device)

    if args.detector_weights:
        dp_detector.load_state_dict(torch.load(args.detector_weights, map_location=torch.device('cpu')))
    else:
        dp_detector.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    dp_detector.eval()


    file_path = 'yaml/data_root.yaml'

    with open(file_path, 'r') as file:
        yaml_data = yaml.safe_load(file)

    data_root = yaml_data['data_root']
    point_test = yaml_data['point_test']
    path = data_root + point_test

    psdataset = ParkingSlotDataset(path)
    logger = util.Logger(enable_visdom=args.enable_visdom)

    total_loss = 0
    position_errors = []
    direction_errors = []
    ground_truths_list = []
    predictions_list = []

    for iter_idx, (image, marking_points) in enumerate(psdataset):

        ground_truths_list.append(marking_points)

        image = torch.unsqueeze(image, 0).to(device)
        prediction = dp_detector(image)
        # print(prediction.shape)
        objective, gradient = generate_objective([marking_points], device, 1, True)
        loss = (prediction - objective) ** 2
        total_loss += torch.sum(loss*gradient).item()

        pred_points = get_predicted_points(prediction[0], 0.01)
        predictions_list.append(pred_points)

        dists, angles = collect_error(marking_points, pred_points,
                                      config.CONFID_THRESH_FOR_POINT)
        position_errors += dists
        direction_errors += angles

        logger.log(iter=iter_idx, total_loss=total_loss)

    precisions, recalls, precision, recall, thresh, f1 = util.calc_precision_recall(
        ground_truths_list, predictions_list, match_marking_points)
    average_precision = util.calc_average_precision(precisions, recalls)
    if args.enable_visdom:
        logger.plot_curve(precisions, recalls)

    sample = torch.randn(1, 3, config.INPUT_IMAGE_SIZE,
                         config.INPUT_IMAGE_SIZE)
    flops, params = profile(dp_detector, inputs=(sample.to(device), ))
    logger.log(average_loss=total_loss / len(psdataset),
               average_precision=average_precision,
               flops=flops,
               params=params)
    
    
    return precision, recall, thresh, f1, average_precision


if __name__ == '__main__':

    args = config.get_parser_for_evaluation().parse_args()
    if args.eval_all:
        file_path = "eval/ev_noagu.txt"
        log_file = open(file_path, "a")
        log_file.write('----------------start eval -----------------\n')
        max_ap = -1
        max_pth = ''
        
        weights_folder = 'weights_noagu'
        pth_files = [file for file in os.listdir(weights_folder) if file.endswith('.pth')]
        pth_files = sorted(pth_files, key=lambda x: os.path.getmtime(os.path.join(weights_folder, x)))
        for pth_file in pth_files:
            model_path = os.path.join(weights_folder, pth_file)
            precision, recall, thresh, f1, ap= evaluate_detector(args, model_path)
            log_file.write(f'{pth_file} thresh: {thresh} precision: {precision} recall: {recall} f1: {f1} ap: {ap}\n')
            if ap > max_ap:
                max_ap = ap
                max_pth = pth_file
            log_file.flush()

        log_file.write('----------------end eval -----------------\n')
        log_file.write(f'max_pth:{max_pth} max_ap: {max_ap}\n')
        log_file.close()
    else:
        evaluate_detector(args)
