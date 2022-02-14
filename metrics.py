import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support


class ModelMetrics(object):
    def __init__(self):
        self.true_labels = []
        self.pred_labels = []

    def push_result(self, prediction, true_labels):
        prediction_ = torch.round(prediction)
        prediction_np = prediction_.cpu().detach().numpy()
        self.pred_labels += prediction_np.tolist()
        true_labels_np = true_labels.cpu().detach().numpy()
        self.true_labels += true_labels_np.tolist()

    def clear_stat(self):
        self.true_labels.clear()
        self.pred_labels.clear()

    def print_stat(self):
        report = classification_report(self.true_labels, self.pred_labels, zero_division=True)
        print(report)
        print(confusion_matrix(self.true_labels, self.pred_labels))
        _, _, f1, _ = precision_recall_fscore_support(self.true_labels, self.pred_labels)
        return np.mean(f1)


class IOUMetrics(object):
    def __init__(self):
        self.true_labels = []
        self.p_labels = []
        self.count = 0

    def push_result(self, p_labels, t_labels):
        p_labels_np = p_labels.squeeze(-1).cpu().detach().numpy()
        t_labels_np = t_labels.squeeze(-1).cpu().detach().numpy()
        self.p_labels += p_labels_np.tolist()
        self.true_labels += t_labels_np.tolist()

    def print_stat(self):
        negative_classes = []
        positive_classes = []
        true_positive_count = 0
        true_negative_count = 0
        total_true_count = 0
        total_false_count = 0
        total_pred_positive_count = 0
        total_pred_negative_count = 0
        false_positive_count = 0

        for i in range(len(self.true_labels)):
            pred_ = self.p_labels[i]
            true_ = self.true_labels[i]

            if pred_ == true_:
                total_true_count += 1
            else:
                total_false_count += 1

            if pred_ == 1:
                total_pred_positive_count += 1
                if pred_ != true_:
                    false_positive_count += 1
            else:
                total_pred_negative_count += 1

            if true_ == 1:
                positive_classes.append(true_)
                if pred_ == true_:
                    true_positive_count += 1


            else:
                negative_classes.append(true_)
                if pred_ == true_:
                    true_negative_count += 1
        positive_total = len(positive_classes) + total_pred_positive_count - true_positive_count
        negative_total = len(negative_classes) + total_pred_negative_count - true_negative_count
        iou_positive = true_positive_count / positive_total if positive_total > 0 else -1
        iou_negative = true_negative_count / negative_total if negative_total > 0 else -1
        print('Overall acc: %f' % (total_true_count / len(self.true_labels)))
        if total_pred_positive_count > 0:
            print(f'P for positive: {true_positive_count / total_pred_positive_count :4f}')
        print(f'R for positive: {true_positive_count / len(positive_classes) :4f}')
        print('Class_%d:  iou class is %f' % (1, iou_positive))
        print('Class_%d:  acc class is %f' % (0, iou_negative))
        mIOU = (iou_positive + iou_negative) / 2
        print('Mean iou is %f' % mIOU)
        return mIOU

    def clear_stat(self):
        self.true_labels.clear()
        self.p_labels.clear()

