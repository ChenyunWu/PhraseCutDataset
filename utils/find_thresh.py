import numpy as np


class ThreshBinSearcher:
    """
    input prediction scores, calculate the thresholds for given pos_percents
    """
    def __init__(self, pos_percents=(0.14, 0.16, 0.18, 0.20), early_stop_img_count=0, max_img_count=0,
                 acc=0.01, min_score=0.0, max_score=1.0, max_per_sample=20480):
        self.acc = acc
        self.min = min_score
        self.max_per_sample = max_per_sample

        num_bins = int((max_score - min_score) / acc)
        self.bins = np.zeros(num_bins)
        self.cum_count = None

        self.percents = pos_percents
        self.threshs = np.zeros(len(pos_percents))

        self.stable_count = 0
        self.img_count = 0
        self.early_stop_count = early_stop_img_count
        self.max_img_count = max_img_count

    def reset(self):
        self.bins = np.zeros_like(self.bins)
        self.cum_count = None
        self.stable_count = 0
        self.img_count = 0

    def is_finished(self):
        if self.stable_count >= self.early_stop_count > 0 or self.img_count >= self.max_img_count > 0:
            return True
        return False

    def update_single_img(self, img_pred_dict, pred_score_tag=None, verbose=False):
        for pred_scores in img_pred_dict.values():
            if pred_score_tag is not None:
                pred_scores = pred_scores[pred_score_tag]
            self._add_scores_to_bin(pred_scores)

        old_threshs = self.threshs.copy()
        for pi, pos_percent in enumerate(self.percents):
            self.threshs[pi] = self._get_threshold_by_percent(pos_percent)

        self.img_count += 1
        if np.max(np.abs(self.threshs - old_threshs)) < self.acc:
            self.stable_count += 1
        else:
            self.stable_count = 0

        if verbose:
            self.print_info()

        if self.is_finished():
            return self.threshs
        return None

    def print_info(self):
        print('stable_count %d/%d, img_count %d/%d, scores logged:'
              % (self.stable_count, self.early_stop_count, self.img_count, self.max_img_count), np.sum(self.bins))
        print('threshs: ', self.threshs)

    def _add_scores_to_bin(self, scores):
        self.cum_count = None
        sample_count = min(self.max_per_sample, scores.size)
        for s in np.random.choice(scores.flat, sample_count, replace=False):
            bi = int((s - self.min) // self.acc)
            self.bins[bi] += 1

        # if scores.size > self.max_per_sample:
        #     sample_rate = self.max_per_sample * 1.0 / scores.size
        #     for s in scores.flat:
        #         if random.random() < sample_rate:
        #             bi = int((s - self.min) // self.acc)
        #             self.bins[bi] += 1

    def _get_threshold_by_percent(self, top_percent):
        if self.cum_count is None:
            self.cum_count = np.cumsum(self.bins)
        target_count = self.cum_count[-1] * (1.0 - top_percent)
        upper_idx = np.argmax(self.cum_count >= target_count)
        if upper_idx == 0:
            thresh = self.min + self.acc * target_count / self.cum_count[0]
        else:
            target_idx = upper_idx - 1.0 * (self.cum_count[upper_idx] - target_count) / \
                         (self.cum_count[upper_idx] - self.cum_count[upper_idx - 1])
            thresh = self.min + (target_idx + 1) * self.acc
        # print(upper_idx, target_idx, top_percent, target_count)
        # print(self.cum_count)
        return thresh


def predict_with_thresh(predictions, threshold=0.5):
    for img_pred in predictions.values():
        for pred in img_pred.values():
            pred_scores = pred['pred_scores']
            pred_mask = pred_scores > threshold
            pred_mask = np.packbits(pred_mask)
            pred['pred_mask'] = pred_mask
    return predictions
