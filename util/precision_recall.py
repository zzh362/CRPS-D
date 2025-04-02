"""Universal procedure of calculating precision and recall."""
import bisect
import numpy as np

def match_gt_with_preds(ground_truth, predictions, match_labels):
    """Match a ground truth with every predictions and return matched index."""
    max_confidence = 0.
    matched_idx = -1
    for i, pred in enumerate(predictions):
        if match_labels(ground_truth, pred[1]) and max_confidence < pred[0]:
            max_confidence = pred[0]
            matched_idx = i
    return matched_idx


def get_confidence_list(ground_truths_list, predictions_list, match_labels):
    """Generate a list of confidence of true positives and false positives."""
    assert len(ground_truths_list) == len(predictions_list)
    true_positive_list = []
    false_positive_list = []
    num_samples = len(ground_truths_list)
    for i in range(num_samples):
        ground_truths = ground_truths_list[i]
        predictions = predictions_list[i]
        prediction_matched = [False] * len(predictions)
        for ground_truth in ground_truths:
            idx = match_gt_with_preds(ground_truth, predictions, match_labels)
            if idx >= 0:
                prediction_matched[idx] = True
                true_positive_list.append(predictions[idx][0])
            else:
                true_positive_list.append(.0)
        for idx, pred_matched in enumerate(prediction_matched):
            if not pred_matched:
                false_positive_list.append(predictions[idx][0])
    return true_positive_list, false_positive_list


def calc_precision_recall(ground_truths_list, predictions_list, match_labels):
    """Adjust threshold to get mutiple precision recall sample."""
    true_positive_list, false_positive_list = get_confidence_list(
        ground_truths_list, predictions_list, match_labels)
    true_positive_list = sorted(true_positive_list)
    false_positive_list = sorted(false_positive_list)
    thresholds = sorted(list(set(true_positive_list)))
    recalls = [0.]
    precisions = [1.]

    # best_pr = 1e9
    best_thresh = 0
    best_precision = 0
    best_recall = 0
    max_f1 = -1
    for thresh in reversed(thresholds):
        if thresh == 0.:
            recalls.append(1.)
            precisions.append(0.)
            break
        false_negatives = bisect.bisect_left(true_positive_list, thresh)
        true_positives = len(true_positive_list) - false_negatives
        true_negatives = bisect.bisect_left(false_positive_list, thresh)
        false_positives = len(false_positive_list) - true_negatives
        recall=(true_positives / (true_positives + false_negatives))
        precision=(true_positives / (true_positives + false_positives))
        recalls.append(recall)
        precisions.append(precision)
        min_pr = abs(precision - recall)
        f1 = 2 * (precision * recall) / (precision + recall)
        if min_pr < 0.02 and f1 > max_f1:
            max_f1 = f1
            best_thresh = thresh
            best_precision = precision
            best_recall = recall
    print('thresh: %f precision: %f recall: %f f1: %f' %(best_thresh, best_precision, best_recall, max_f1))
    # print(max_f1, best_thresh)    

        

    thresh = 0.55
    false_negatives = bisect.bisect_left(true_positive_list, thresh)
    true_positives = len(true_positive_list) - false_negatives
    true_negatives = bisect.bisect_left(false_positive_list, thresh)
    false_positives = len(false_positive_list) - true_negatives
    recall=(true_positives / (true_positives + false_negatives))
    precision=(true_positives / (true_positives + false_positives))
    f1 = 2 * (precision * recall) / (precision + recall)
    print('thresh: %f precision: %f recall: %f f1: %f' %(thresh, precision, recall, f1))
    # print(thresholds[:10])


    return precisions, recalls, best_precision, best_recall, best_thresh, max_f1


# def calc_average_precision(precisions, recalls):
#     """Calculate average precision defined in VOC contest."""
#     total_precision = 0.
#     for i in range(11):
#         index = next(conf[0] for conf in enumerate(recalls) if conf[1] >= i/10)
#         total_precision += max(precisions[index:])
#     return total_precision / 11

def calc_average_precision(prec, rec):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.array(rec) 
    mpre = np.array(prec)

    # compute the precision envelope
    # 计算出precision的各个断点(折线点)
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]  #precision前后两个值不一样的点
    # print(mrec[1:], mrec[:-1])
    # print(i) #[0, 1, 3, 4, 5]

    # AP= AP1 + AP2+ AP3+ AP4
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap
