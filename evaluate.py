import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

plt.switch_backend('agg')
current_path = os.path.realpath(__file__)
dataset_path = os.path.join(current_path, '..')
sys.path.append(os.path.abspath(dataset_path))

from utils.iou import iou_boxes, iou_polygons_masks
from utils.refvg_loader import RefVGLoader
from utils.predictor_examples import vg_gt_predictor, vg_rand_predictor, ins_rand_predictor
from utils import subset as subset_utils


class Evaluator:

    def __init__(self, refvg_loader=None, refvg_split='miniv', analyze_subset=True):
        """
        :param refvg_loader:
        :param refvg_split: only used when refvg_loader is None
        :param analyze_subset:
        """
        if refvg_loader is None:
            refvg_loader = RefVGLoader(split=refvg_split)
            refvg_loader.shuffle()
        else:
            refvg_split = '_'.join(refvg_loader.splits)
        self.refvg_loader = refvg_loader
        self.refvg_split = refvg_split

        # stats for each subset: correct_count, iou_box, iou_mask, i_mask, u_mask
        self.subset_stats = {'all': [0, [], [], [], []]}
        self.analyze_subset = analyze_subset
        if analyze_subset:
            for k in subset_utils.subsets:
                self.subset_stats[k] = [0, [], [], [], []]
        self.evaluated_img_ids = set()
        self.evaluated_task_count = 0

    def eval_single_img(self, img_id, im_pred_dict, pred_mask_tag='pred_mask', pred_boxes_tag=None,
                        correct_tag=None, verbose=False, mask_score_thresh=0, log_to_evaluator=True):
        if img_id not in self.refvg_loader.img_ids:
            print('WARNING: IMG %d is not in RefVG %s. Ignored.' % (img_id, self.refvg_split))
            return
        if img_id in self.evaluated_img_ids and log_to_evaluator:
            print('WARNING: IMG %d is already evaluated. Ignored.' % img_id)
            return
        if log_to_evaluator:
            self.evaluated_img_ids.add(img_id)

        img_data = self.refvg_loader.get_img_ref_data(img_id)
        img_box_ious = dict()
        img_mask_ious = dict()
        for task_i, task_id in enumerate(img_data['task_ids']):
            if task_id not in im_pred_dict:
                print('WARNING: no prediction on task: %s' % task_id)
                continue
            task_pred_dict = im_pred_dict[task_id]
            for t in [pred_mask_tag, pred_boxes_tag, correct_tag]:
                if t is not None:
                    assert t in task_pred_dict

            iou_box, iou_mask, i_mask, u_mask = 0.0, 0.0, 0.0, 0.0
            evaluated = False

            if pred_mask_tag is not None:
                gt_Polygons = img_data['gt_Polygons'][task_i]
                gt_polygons = list()
                for ps in gt_Polygons:
                    gt_polygons += ps
                pred_mask = task_pred_dict[pred_mask_tag]
                if len(pred_mask.shape) == 1:
                    pred_mask = np.unpackbits(pred_mask)[:img_data['height'] * img_data['width']] \
                        .reshape((img_data['height'], img_data['width']))
                elif mask_score_thresh > 0:
                    pred_mask = pred_mask > mask_score_thresh
                iou_mask, i_mask, u_mask = iou_polygons_masks(gt_polygons, [pred_mask], iandu=True)
                img_mask_ious[task_id] = iou_mask
                evaluated = True

            if pred_boxes_tag is not None:
                pred_boxes = task_pred_dict[pred_boxes_tag]
                iou_box = iou_boxes(pred_boxes, img_data['gt_boxes'][task_i])
                img_box_ious[task_id] = iou_box
                evaluated = True

            correct = 0
            if correct_tag is not None:
                correct = task_pred_dict[correct_tag]

            if log_to_evaluator:
                if evaluated:
                    self.evaluated_task_count += 1

                subsets = ['all']
                if self.analyze_subset:
                    subsets = self.refvg_loader.get_task_subset(img_id, task_id)
                for k in subsets:
                    self.subset_stats[k][0] += correct
                    self.subset_stats[k][1].append(float(iou_box))
                    self.subset_stats[k][2].append(float(iou_mask))
                    self.subset_stats[k][3].append(float(i_mask))
                    self.subset_stats[k][4].append(float(u_mask))

        if verbose:
            bi = np.mean(self.subset_stats['all'][1])
            mi = np.mean(self.subset_stats['all'][2])
            print('img|task[%d|%d] %s: %d phrases. Up till now: mean_iou_box %.3f, mean_iou_mask %.3f' %
                  (len(self.evaluated_img_ids), self.evaluated_task_count, img_id, len(img_data['task_ids']), bi, mi))

        return img_mask_ious, img_box_ious

    def analyze_stats(self, mask_box=('mask', 'box'), exp_name_in_summary=None, save_result_to_path=None):
        stats = self.subset_stats
        results = dict()
        result_f = None
        summary_mask = None
        summary_box = None
        summary_subset = None
        subset_summary_str = ''

        s = 'subsets:\n' + ','.join(subset_utils.subsets)
        print(s)
        if save_result_to_path is not None:
            if not os.path.exists(save_result_to_path):
                os.makedirs(save_result_to_path)
            result_f = open(os.path.join(save_result_to_path, 'results.txt'), 'w')
            result_f.write(s + '\n')
        if exp_name_in_summary is not None:
            if 'mask' in mask_box:
                summary_mask = open('output/eval_refvg/summary_mask.csv', 'a')
                summary_subset = open('output/eval_refvg/summary_subset.csv', 'a')
                subset_summary_str = exp_name_in_summary
            if 'box' in mask_box:
                summary_box = open('output/eval_refvg/summary_box.csv', 'a')

        task_num = self.evaluated_task_count

        for subset in subset_utils.subsets:
            if subset not in stats:
                continue
            stat = stats[subset]
            count = len(stat[1])
            if count == 0:
                s = '\n%s: count = 0' % subset
                print(s)
                if save_result_to_path is not None:
                    result_f.write(s + '\n')
                if exp_name_in_summary is not None:
                    subset_summary_str += ',0.0'
                continue

            subset_result = dict()
            result_str_head = '\n%s: count=%d(%.4f)' % (subset, count, count * 1.0 / task_num)

            pred_box_acc_str = ''
            if 'box' in mask_box:
                box_acc = stat[0] * 1.0 / count
                mean_box_iou = float(np.mean(stat[1]))
                result_str_head += ', box_acc=%.4f, mean_box_iou=%.4f' % (box_acc, mean_box_iou)

                box_threshs = [0.5, 0.6, 0.7, 0.8, 0.9]
                pred_box_acc = {}
                pred_box_acc_str = '\npred_box_acc: '
                box_sum_str = '%s,%.4f' % (exp_name_in_summary, mean_box_iou)

                for thresh in box_threshs:
                    pred_box_acc[thresh] = np.sum(np.array(stat[1]) > thresh) * 1.0 / count
                    pred_box_acc_str += 'acc%.1f = %.4f  ' % (thresh, pred_box_acc[thresh])
                    box_sum_str += ',%.4f' % pred_box_acc[thresh]

                if exp_name_in_summary is not None and subset == 'all':
                    summary_box.write(box_sum_str + '\n')

                subset_result['box_acc'] = box_acc
                subset_result['mean_box_iou'] = mean_box_iou
                subset_result['pred_box_acc'] = pred_box_acc

            pred_mask_acc_str = ''
            if 'mask' in mask_box:
                mean_mask_iou = np.mean(stat[2])
                cum_mask_iou = np.sum(stat[3]) * 1.0 / np.sum(stat[4])

                result_str_head += ', mean_mask_iou=%.4f, cum_mask_iou=%.4f' % (mean_mask_iou, cum_mask_iou)
                if exp_name_in_summary is not None:
                    subset_summary_str += ',%.4f' % mean_mask_iou

                mask_threshs = [0.5, 0.6, 0.7, 0.8, 0.9]
                pred_mask_acc = {}
                pred_mask_acc_str = '\npred_mask_acc: '
                mask_sum_str = '%s,%.4f,%.4f' % (exp_name_in_summary, mean_mask_iou, cum_mask_iou)
                for thresh in mask_threshs:
                    pred_mask_acc[thresh] = np.sum(np.array(stat[2]) > thresh) * 1.0 / count
                    pred_mask_acc_str += 'acc%.1f = %.4f  ' % (thresh, pred_mask_acc[thresh])
                    mask_sum_str += ',%.4f' % pred_mask_acc[thresh]

                if exp_name_in_summary is not None and subset == 'all':
                    summary_mask.write(mask_sum_str + '\n')

                subset_result['mean_mask_iou'] = mean_mask_iou
                subset_result['cum_mask_iou'] = cum_mask_iou
                subset_result['pred_mask_acc'] = pred_mask_acc

            result_str = result_str_head + pred_box_acc_str + pred_mask_acc_str
            print(result_str)
            if save_result_to_path is not None:
                result_f.write(result_str)

            results[subset] = subset_result

        if save_result_to_path is not None:
            result_f.close()
        if exp_name_in_summary is not None:
            if 'mask' in mask_box:
                summary_mask.close()
                summary_subset.write(subset_summary_str + '\n')
                summary_subset.close()
            if 'box' in mask_box:
                summary_box.close()
        return results


def evaluate_from_pred_dict(predictions, refvg_split, analyze_subset=True, exp_name_in_summary=None,
                            save_result_to_path=None, update_predictions=False, verbose=False):

    evaluator = Evaluator(refvg_split=refvg_split, analyze_subset=analyze_subset)
    mb = list()
    for img_id, im_preds in predictions.items():
        pred_mask_tag = None
        pred_boxes_tag = None
        correct_tag = None
        for task_id, task_pred in im_preds.items():
            if 'pred_mask' in task_pred:
                pred_mask_tag = 'pred_mask'
            if 'pred_boxes' in task_pred:
                pred_boxes_tag = 'pred_boxes'
            if 'correct' in task_pred:
                correct_tag = 'correct'
            break
        if len(mb) == 0:
            if pred_mask_tag is not None:
                mb.append('mask')
            if pred_boxes_tag is not None:
                mb.append('box')
        im, ib = evaluator.eval_single_img(img_id=img_id, im_pred_dict=im_preds, pred_mask_tag=pred_mask_tag,
                                           pred_boxes_tag=pred_boxes_tag, correct_tag=correct_tag, verbose=verbose)
        if update_predictions:
            for task_id, task_pred in im_preds.items():
                if task_id in im:
                    task_pred['mask_iou'] = im[task_id]
                if task_id in ib:
                    task_pred['box_iou'] = ib[task_id]

    evaluator.analyze_stats(mask_box=mb, exp_name_in_summary=exp_name_in_summary,
                            save_result_to_path=save_result_to_path)

    return predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--pred_name', type=str, default='vg_pred', help='name of the predictor to be evaluated:'
                                                                               'vg_gt, vg_rand, ins_rand')
    parser.add_argument('-p', '--pred_path', type=str, default=None,
                        help='path to the prediction results. If empty, run the predictor.')
    parser.add_argument('-s', '--split', type=str, default='miniv',
                        help='dataset split to evaluate: val, miniv, test, train, val_miniv, etc')
    parser.add_argument('-a', '--analyze_subset', type=int, default=1, help='whether to enable subset analysis')
    parser.add_argument('-f', '--save_pred', type=int, default=1, help='whether to save evaluated predictions to file.')
    parser.add_argument('-l', '--log_to_summary', type=int, default=1, help='whether to log results to the summary.')
    args = parser.parse_args()

    out_path = None
    if args.save_pred:
        if args.pred_path is not None:
            out_path = os.path.dirname(args.pred_path)
        else:
            out_path = os.path.join('output/eval_refvg', args.pred_name)

    if not args.pred_path or not os.path.exists(args.pred_path):
        if args.pred_name.starts_with('vg_gt'):
            predictions = vg_gt_predictor(split=args.split, eval_img_count=-1, out_path=out_path)
        elif args.pred_name.starts_with('vg_rand'):
            predictions = vg_rand_predictor(split=args.split, eval_img_count=-1, out_path=out_path)
        elif args.pred_name.starts_with('ins_rand'):
            predictions = ins_rand_predictor(split=args.split, eval_img_count=-1, out_path=out_path)
        else:
            raise NotImplementedError
    else:
        predictions = np.load(args.pred_path).item()

    predictions = evaluate_from_pred_dict(predictions=predictions, refvg_split=args.split,
                                          analyze_subset=args.analyze_subset, exp_name_in_summary=args.pred_name,
                                          save_result_to_path=out_path, update_predictions=args.save_pred, verbose=True)
    if args.save_pred:
        print('saving %s to %s' % (args.pred_name, out_path))
        pred_path = os.path.join(out_path, 'pred_eval.npy')
        np.save(pred_path, predictions)

    return


if __name__ == '__main__':
    main()
