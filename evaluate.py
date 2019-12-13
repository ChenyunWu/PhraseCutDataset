import argparse
import os
import numpy as np

from utils.evaluator import Evaluator
from utils.predictor_examples import vg_gt_predictor, vg_rand_predictor, ins_rand_predictor
from utils.file_paths import output_path


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
            if 'pred_boxlist' in task_pred:
                pred_boxes_tag = 'pred_boxlist'
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
    parser.add_argument('-n', '--pred_name', type=str, default='ins_rand',
                        help='name of the predictor to be evaluated: vg_gt, vg_rand, ins_rand')
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
            out_path = os.path.join(output_path, 'baselines', args.pred_name, args.split)

    if not args.pred_path or not os.path.exists(args.pred_path):
        if args.pred_name.startswith('vg_gt'):
            predictions = vg_gt_predictor(split=args.split)
        elif args.pred_name.startswith('vg_rand'):
            predictions = vg_rand_predictor(split=args.split)
        elif args.pred_name.startswith('ins_rand'):
            predictions = ins_rand_predictor(split=args.split)
        else:
            raise NotImplementedError
    else:
        predictions = np.load(args.pred_path).item()

    if not args.log_to_summary:
        exp_name = None
    else:
        exp_name = args.pred_name
    predictions = evaluate_from_pred_dict(predictions=predictions, refvg_split=args.split,
                                          analyze_subset=args.analyze_subset, exp_name_in_summary=exp_name,
                                          save_result_to_path=out_path, update_predictions=args.save_pred, verbose=True)
    if args.save_pred:
        print('saving %s to %s' % (args.pred_name, out_path))
        pred_path = os.path.join(out_path, 'pred_eval.npy')
        np.save(pred_path, predictions)

    return


if __name__ == '__main__':
    main()
