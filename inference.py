"""Inference demo of directional point detector."""
import math
import cv2 as cv
import numpy as np
import torch
from torchvision.transforms import ToTensor
import config
from data import get_predicted_points, pair_marking_points, calc_point_squre_dist, pass_through_third_point
from data.process import pair_marking_points_slant, pair_marking_points_vertical
from model import TeacherDetector
from model.effnetv2 import effnetv2_base
from util import Timer
from pathlib import Path

def plot_points(image, pred_points):
    """Plot marking points on the image."""
    if not pred_points:
        return
    height = image.shape[0]
    width = image.shape[1]
    image_size = max(height, width)
    for confidence, marking_point in pred_points:
        p0_x = image_size * marking_point.x - 0.5
        p0_y = image_size * marking_point.y - 0.5
        cos_val = math.cos(marking_point.direction0)
        sin_val = math.sin(marking_point.direction0)
        p1_x = p0_x + 50*cos_val
        p1_y = p0_y + 50*sin_val
        if marking_point.type < 0.5:
            p2_x = p0_x - 50*sin_val
            p2_y = p0_y + 50*cos_val
            p3_x = p0_x + 50*sin_val
            p3_y = p0_y - 50*cos_val
        else:
            cos_val = math.cos(marking_point.direction1)
            sin_val = math.sin(marking_point.direction1)
            p2_x = p0_x + 50*cos_val
            p2_y = p0_y + 50*sin_val
            p3_x = p0_x - 50*cos_val
            p3_y = p0_y - 50*sin_val
        p0_x = int(round(p0_x))
        p0_y = int(round(p0_y))
        p1_x = int(round(p1_x))
        p1_y = int(round(p1_y))
        p2_x = int(round(p2_x))
        p2_y = int(round(p2_y))
        cv.line(image, (p0_x, p0_y), (p1_x, p1_y), (0, 0, 255), 2)
        cv.putText(image, str(confidence), (p0_x, p0_y),
                   cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 0))
        if marking_point.shape > 0.5:
            cv.line(image, (p0_x, p0_y), (p2_x, p2_y), (0, 0, 255), 2)
        else:
            p3_x = int(round(p3_x))
            p3_y = int(round(p3_y))
            cv.line(image, (p2_x, p2_y), (p3_x, p3_y), (0, 0, 255), 2)


def plot_slots(image, pred_points, slots):
    """Plot parking slots on the image."""
    if not pred_points or not slots:
        return
    marking_points = list(list(zip(*pred_points))[1])
    height = image.shape[0]
    width = image.shape[1]
    image_size = max(height, width)
    for slot in slots:
        point_a = marking_points[slot[0]]
        point_b = marking_points[slot[1]]
        p0_x = image_size * point_a.x - 0.5
        p0_y = image_size * point_a.y - 0.5
        p1_x = image_size * point_b.x - 0.5
        p1_y = image_size * point_b.y - 0.5
        if point_a.type < 0.5:
            distance = calc_point_squre_dist(point_a, point_b)
            if distance <= config.VSLOT_MAX_DIST * config.SQUARED_RATIO:
                separating_length = config.LONG_SEPARATOR_LENGTH * config.RATIO
            else:
                separating_length = config.SHORT_SEPARATOR_LENGTH * config.RATIO
        else:
            separating_length = config.SLANT_SEPARATOR_LENGTH * config.RATIO

        cos_val = math.cos(slot[2])
        sin_val = math.sin(slot[2])

        p2_x = p0_x + image_size * separating_length * cos_val
        p2_y = p0_y + image_size * separating_length * sin_val
        p3_x = p1_x + image_size * separating_length * cos_val
        p3_y = p1_y + image_size * separating_length * sin_val

        p0_x = int(round(p0_x))
        p0_y = int(round(p0_y))
        p1_x = int(round(p1_x))
        p1_y = int(round(p1_y))
        p2_x = int(round(p2_x))
        p2_y = int(round(p2_y))
        p3_x = int(round(p3_x))
        p3_y = int(round(p3_y))
        cv.line(image, (p0_x, p0_y), (p1_x, p1_y), (255, 0, 0), 2)
        cv.line(image, (p0_x, p0_y), (p2_x, p2_y), (255, 0, 0), 2)
        cv.line(image, (p1_x, p1_y), (p3_x, p3_y), (255, 0, 0), 2)


def preprocess_image(image):
    """Preprocess numpy image to torch tensor."""
    # if image.shape[0] != 512 or image.shape[1] != 512:
    #     image = cv.resize(image, (512, 512))
    # image = cv.resize(image, (768, 416))
    # image = cv.resize(image, (608, 320))
    return torch.unsqueeze(ToTensor()(image), 0)


def detect_marking_points(detector, image, thresh, device):
    """Given image read from opencv, return detected marking points."""
    prediction = detector(preprocess_image(image).to(device))
    ss = prediction.detach().cpu().numpy()
    return get_predicted_points(prediction[0], thresh)


def inference_slots(marking_points):
    """Inference slots based on marking points."""
    num_detected = len(marking_points)
    slots = []
    for i in range(num_detected - 1):
        for j in range(i + 1, num_detected):
            useSlant = False
            useVertical = False
            point_i = marking_points[i]
            point_j = marking_points[j]
            # Step 1: length filtration.
            distance = calc_point_squre_dist(point_i, point_j)
            if point_i.type < 0.5 and point_j.type > 0.5:
                useSlant = True
                # continue
            if point_i.type > 0.5 and point_j.type < 0.5:
                useSlant = True
                # continue
            if useSlant and distance > config.SLANT_MAX_DIST * config.SQUARED_RATIO:
                useVertical = True
                useSlant = False
            if point_i.type < 0.5:
                if not (config.VSLOT_MIN_DIST * config.SQUARED_RATIO <= distance <= config.VSLOT_MAX_DIST * config.SQUARED_RATIO
                    or config.HSLOT_MIN_DIST * config.SQUARED_RATIO <= distance <= config.HSLOT_MAX_DIST * config.SQUARED_RATIO
                    or useVertical):
                    continue
            else:
                if not(config.SLANT_MIN_DIST * config.SQUARED_RATIO <= distance <= config.SLANT_MAX_DIST * config.SQUARED_RATIO or useVertical):
                    continue
            # Step 2: pass through filtration.
            if pass_through_third_point(marking_points, i, j):
                continue
            result = pair_marking_points(point_i, point_j)
            if useSlant:
                result = pair_marking_points_slant(point_i, point_j)
            if useVertical:
                result = pair_marking_points_vertical(point_i, point_j)
            if result[0] == 1:
                slots.append((i, j, result[1]))
            elif result[0] == -1:
                slots.append((j, i, result[1]))

    # slot
    return slots


def detect_video(detector, device, args):
    """Demo for detecting video."""
    timer = Timer()
    input_video = cv.VideoCapture(args.video)
    frame_width = int(input_video.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(input_video.get(cv.CAP_PROP_FRAME_HEIGHT))
    output_video = cv.VideoWriter()
    if args.save:
        output_video.open('record.avi', cv.VideoWriter_fourcc(*'XVID'),
                          input_video.get(cv.CAP_PROP_FPS),
                          (frame_width, frame_height), True)
    frame = np.empty([frame_height, frame_width, 3], dtype=np.uint8)
    while input_video.read(frame)[0]:
        timer.tic()
        pred_points = detect_marking_points(
            detector, frame, args.thresh, device)
        slots = None
        if pred_points and args.inference_slot:
            marking_points = list(list(zip(*pred_points))[1])
            slots = inference_slots(marking_points)
        timer.toc()
        plot_points(frame, pred_points)
        plot_slots(frame, pred_points, slots)
        cv.imshow('demo', frame)
        cv.waitKey(1)
        if args.save:
            output_video.write(frame)
    print("Average time: ", timer.calc_average_time(), "s.")
    input_video.release()
    output_video.release()


def detect_image(detector, device, args):
    """Demo for detecting images."""
    timer = Timer()
    image_dir = Path('/data/cylin/zzh/DMPR/VAE/output/')
    slot_num = 0
    slot_pic = 0
    for img_path in image_dir.glob('*.jpg'):
        image = cv.imread(str(img_path))
        timer.tic()
        pred_points = detect_marking_points(
            detector, image, 0.4, device)
        slots = None
        if pred_points and args.inference_slot:
            marking_points = list(list(zip(*pred_points))[1])
            slots = inference_slots(marking_points)
        timer.toc()
        if slots != None:
            if len(slots) > 0:
                slot_pic += 1
            slot_num += len(slots)

        plot_points(image, pred_points)
        plot_slots(image, pred_points, slots)
        save_path = 'test/' + img_path.stem + '.jpg'
        cv.imwrite(save_path, image, [int(cv.IMWRITE_JPEG_QUALITY), 100])
    print("Average time: ", timer.calc_average_time(), "s.")
    print(slot_pic)
    print(slot_num)

def inference_detector(args):
    """Inference demo of directional point detector."""
    args.cuda = not args.disable_cuda and torch.cuda.is_available()
    device = torch.device('cuda:' + str('4') if args.cuda else 'cpu')
    torch.set_grad_enabled(False)
    dp_detector = TeacherDetector(
        3, args.depth_factor, config.NUM_FEATURE_MAP_CHANNEL).to(device)
    # dp_detector = shufflenetv2(model_size= '0.5x').to(device)
    # dp_detector = effnetv2_base().to(device)
    # print(dp_detector)
    dp_detector.load_state_dict(torch.load(args.detector_weights, map_location=torch.device('cpu')))
    dp_detector.eval()
    # sample = torch.randn(1, 3, config.INPUT_IMAGE_SIZE,
    #                      config.INPUT_IMAGE_SIZE)
    # flops, params = profile(dp_detector, inputs=(sample.to(device),))
    # print(flops)
    if args.mode == "image":
        detect_image(dp_detector, device, args)
    elif args.mode == "video":
        detect_video(dp_detector, device, args)


if __name__ == '__main__':
    inference_detector(config.get_parser_for_inference().parse_args())
