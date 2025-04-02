"""Evaluate directional marking point detector."""
import json
import os
import math
import cv2 as cv
import numpy as np
import torch
import yaml
import config
import util
from data import match_slots, Slot
from data.struct import direction_diff
from model import TeacherDetector, DirectionalPointDetector
from model.effnetv2 import effnetv2_base
from inference import detect_marking_points, inference_slots
import matplotlib.pyplot as plt

def plot_pr_curve(precisions, recalls, save_path=None):
    """Plot the PR curve given precision and recall values and save it as an image."""
    plt.plot(recalls, precisions, '-o')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def get_ground_truths(label):
    """Read label to get ground truth slot."""
    slots = np.array(label['slots'])
    if slots.size == 0:
        return []
    if len(slots.shape) < 2:
        slots = np.expand_dims(slots, axis=0)
    marks = np.array(label['marks'])
    if len(marks.shape) < 2:
        marks = np.expand_dims(marks, axis=0)
    ground_truths = []
    for slot in slots:
        mark_a = marks[int(slot[0]) - 1]
        mark_b = marks[int(slot[1]) - 1]
        angle = slot[3] * math.pi / 180
        coords = np.array([(mark_a[0]-0.5)/512, (mark_a[1]-0.5)/512, (mark_b[0]-0.5)/512, (mark_b[1]-0.5)/512, angle])
        ground_truths.append(Slot(*coords))
    return ground_truths


def psevaluate_detector(args, model_path=''):
    """Evaluate directional point detector."""
    args.cuda = not args.disable_cuda and torch.cuda.is_available()
    device = torch.device('cuda:' + str("2") if args.cuda else 'cpu')
    torch.set_grad_enabled(False)

    # dp_detector = effnetv2_base().to(device)

    dp_detector = TeacherDetector(
        3, args.depth_factor, config.NUM_FEATURE_MAP_CHANNEL).to(device)


    if args.detector_weights:
        dp_detector.load_state_dict(torch.load(args.detector_weights, map_location=torch.device('cpu')))
    else:
        dp_detector.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    dp_detector.eval()

    logger = util.Logger(enable_visdom=args.enable_visdom)

    ground_truths_list = []
    predictions_list = []

    file_path = 'yaml/data_root.yaml'

    with open(file_path, 'r') as file:
        yaml_data = yaml.safe_load(file)

    data_root = yaml_data['data_root']
    slot_test = yaml_data['slot_test']
    img_test = yaml_data['img_test']
    label_directory = data_root + slot_test
    image_directory = data_root + img_test

    for idx, label_file in enumerate(os.listdir(label_directory)):
        name = os.path.splitext(label_file)[0]
        print(idx, name)
        image = cv.imread(os.path.join(image_directory, name + '.jpg'))

        pred_points = detect_marking_points(
            dp_detector, image, config.CONFID_THRESH_FOR_POINT, device)
        

        slots = []
        if pred_points:
            marking_points = list(list(zip(*pred_points))[1])
        
            slots = inference_slots(marking_points)


        pred_slots = []
        for slot in slots:
            point_a = marking_points[slot[0]]
            point_b = marking_points[slot[1]]
            # 置信度
            prob = min((pred_points[slot[0]][0], pred_points[slot[1]][0]))
            vector = np.array([point_b.x - point_a.x, point_b.y - point_a.y])
            vec_direct = math.atan2(vector[1], vector[0])
            angle = direction_diff(vec_direct, slot[2])
            pred_slots.append(
                (prob, Slot(point_a.x, point_a.y, point_b.x, point_b.y, angle)))
        predictions_list.append(pred_slots)


        with open(os.path.join(label_directory, label_file), 'r') as file:
            ground_truths_list.append(get_ground_truths(json.load(file)))

    precisions, recalls, precision, recall, thresh, f1 = util.calc_precision_recall(
        ground_truths_list, predictions_list, match_slots)
    average_precision = util.calc_average_precision(precisions, recalls)

    # 绘制并保存 PR 曲线
    # save_path = 'pr_curve.png'
    # plot_pr_curve(precisions, recalls, save_path)

    if args.enable_visdom:
        logger.plot_curve(precisions, recalls)
    logger.log(average_precision=average_precision)

    return precision, recall, thresh, f1, average_precision

# import traceback
if __name__ == '__main__':
    args = config.get_parser_for_ps_evaluation().parse_args()
    if args.eval_all:
        file_path = "eval/ps_ep10.txt"
        log_file = open(file_path, "a")
        log_file.write('----------------start eval -----------------\n')
        max_ap = -1
        max_pth = ''
        
        weights_folder = 'weights_eps10'
        pth_files = [file for file in os.listdir(weights_folder) if file.endswith('.pth')]
        pth_files = sorted(pth_files, key=lambda x: os.path.getmtime(os.path.join(weights_folder, x)))
        for pth_file in pth_files:
            model_path = os.path.join(weights_folder, pth_file)
            precision, recall, thresh, f1, ap= psevaluate_detector(args, model_path)
            log_file.write(f'{pth_file} thresh: {thresh} precision: {precision} recall: {recall} f1: {f1} ap: {ap}\n')
            if ap > max_ap:
                max_ap = ap
                max_pth = pth_file
            log_file.flush()

        log_file.write('----------------end eval -----------------\n')
        log_file.write(f'max_pth:{max_pth} max_ap: {max_ap}\n')
        log_file.close()
    else:
        psevaluate_detector(args)

    # try:
    #     psevaluate_detector(config.get_parser_for_ps_evaluation().parse_args())
    # except Exception as e:
    #     fp = open('log.txt', 'a')
    #     traceback.print_exc(file=fp)
    #     fp.close()
