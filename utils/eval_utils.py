import random
import numpy as np


class BinThreshSearcher:
    """
    given prediction scores, calculate
    """
    def __init__(self, bin_size=0.01, min_score=0.0, max_score=1.0, sample_rate=0.005):
        self.bin_size = bin_size
        self.min = min_score
        self.max = max_score
        self.sample_rate = sample_rate
        self.num_bins = int((max_score - min_score) / bin_size)
        self.bins = np.zeros(self.num_bins)
        self.cum_count = None

    def reset(self):
        self.bins = np.zeros(self.num_bins)
        self.cum_count = None

    def add_scores_to_bin(self, scores):
        self.cum_count = None
        for s in scores.flat:
            if random.random() < self.sample_rate:
                bi = int((s - self.min) // self.bin_size)
                self.bins[bi] += 1

    def get_threshold_by_percent(self, top_percent):
        if self.cum_count is None:
            self.cum_count = np.cumsum(self.bins)
        target_count = self.cum_count[-1] * (1.0 - top_percent)
        upper_idx = np.argmax(self.cum_count >= target_count)
        if upper_idx == 0:
            thresh = self.min + self.bin_size * target_count / self.cum_count[0]
        else:
            target_idx = upper_idx - 1.0 * (self.cum_count[upper_idx] - target_count) / \
                         (self.cum_count[upper_idx] - self.cum_count[upper_idx - 1])
            thresh = self.min + (target_idx + 1) * self.bin_size
        # print(upper_idx, target_idx, top_percent, target_count)
        # print(self.cum_count)
        return thresh


def find_thresh_by_percents(predictions, pos_percents=(0.2, 0.3, 0.4, 0.5), acc=0.01,
                            early_stop_img=0, max_img_count=0):
    thresholds = np.zeros(len(pos_percents))
    thresholds_old = np.zeros(len(pos_percents))
    stable_img_count = 0
    thresh_searcher = BinThreshSearcher(bin_size=acc)

    img_ids = list(predictions.keys())
    random.shuffle(img_ids)
    for img_i, img_id in enumerate(img_ids):
        thresholds_old[:] = thresholds

        for pred in predictions[img_id].values():
            pred_scores = pred['pred_scores']
            thresh_searcher.add_scores_to_bin(pred_scores)

        if img_i > max_img_count > 0:
            break

        if early_stop_img > 0:
            for pi, pos_percent in enumerate(pos_percents):
                thresholds[pi] = thresh_searcher.get_threshold_by_percent(pos_percent)
            if np.max(np.abs(thresholds - thresholds_old)) < acc:
                stable_img_count += 1
            if stable_img_count > early_stop_img:
                break

    if np.sum(thresholds) == 0:
        for pi, pos_percent in enumerate(pos_percents):
            thresholds[pi] = thresh_searcher.get_threshold_by_percent(pos_percent)

    # print(thresh_searcher)
    print('score bins: ', thresh_searcher.bins)
    # print(thresholds)
    return thresholds


def predict_with_thresh(predictions, threshold=0.5):
    for img_pred in predictions.values():
        for pred in img_pred.values():
            pred_scores = pred['pred_scores']
            pred_mask = pred_scores > threshold
            pred_mask = np.packbits(pred_mask)
            pred['pred_mask'] = pred_mask
    return predictions
