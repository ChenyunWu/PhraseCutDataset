import argparse
import os
import numpy as np
from PIL import Image

from utils.evaluator import Evaluator
from utils.predictor_examples import vg_gt_predictor, vg_rand_predictor, ins_rand_predictor, box_rand_predictor
from utils.file_paths import output_path


def evaluate_from_pred_folder(pred_folder, refvg_split, analyze_subset=True, exp_name_in_summary=None,
                              save_result_to_path=None, verbose=False):

    evaluator = Evaluator(refvg_split=refvg_split, analyze_subset=analyze_subset)
    fnames = os.listdir(pred_folder)
    fnames.sort()
    img_preds = dict()
    cur_img_id = ''
    for fname in fnames:
        if not fname.endswith('.png'):
            continue
        task_id = fname.split('.')[0]
        img_id = task_id.split('__')[0]
        im_frame = Image.open(os.path.join(pred_folder, fname))
        w, h = im_frame.size
        np_frame = np.array(im_frame.getdata()).reshape((h, w))

        if img_id == cur_img_id:
            img_preds[task_id] = {'pred_mask': np_frame}

        else:
            if len(img_preds) > 0:
                # print(img_preds)
                im, _ = evaluator.eval_single_img(img_id=int(cur_img_id), im_pred_dict=img_preds,
                                                  pred_mask_tag='pred_mask', verbose=verbose)
            cur_img_id = img_id
            img_preds = dict()
            img_preds[task_id] = {'pred_mask': np_frame}

    evaluator.analyze_stats(mask_box=['mask'], exp_name_in_summary=exp_name_in_summary,
                            save_result_to_path=save_result_to_path)

    return


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
    parser.add_argument('-n', '--pred_name', type=str, default='box_rand',
                        help='name of the predictor: box_rand, vg_gt, vg_rand, ins_rand, or your own name.')
    parser.add_argument('-p', '--pred_path', type=str, default=None,
                        help='path to the folder containing prediction results. If empty, check pred_dict.')
    parser.add_argument('-d', '--pred_dict', type=str, default=None,
                        help='path to the .npy file of prediction results. If empty, run the predictor.')
    parser.add_argument('-s', '--split', type=str, default='miniv',
                        help='dataset split to evaluate: val, miniv, test, train, val_miniv, etc')
    parser.add_argument('-a', '--analyze_subset', type=int, default=1, help='whether to enable subset analysis')
    parser.add_argument('-f', '--save_pred', type=int, default=1, help='whether to save pred numpy to file.')
    parser.add_argument('-l', '--log_to_summary', type=int, default=1, help='whether to log results to the summary.')
    args = parser.parse_args()

    # out_path
    out_path = None
    if args.save_pred:
        if args.pred_path is not None:
            out_path = os.path.dirname(args.pred_path)
        elif args.pred_dict is not None:
            out_path = os.path.dirname(args.pred_dict)
        else:
            out_path = os.path.join(output_path, 'baselines', args.pred_name, args.split)

    # exp_name for summary
    if not args.log_to_summary:
        exp_name = None
    else:
        exp_name = args.pred_name

    # evaluate
    if args.pred_path is None and args.pred_name.startswith('box_rand'):
        pred_folder = os.path.join(out_path, 'predictions')
        box_rand_predictor(split=args.split, out_path=pred_folder)
        evaluate_from_pred_folder(pred_folder=pred_folder, refvg_split=args.split,
                                  analyze_subset=args.analyze_subset, exp_name_in_summary=exp_name,
                                  save_result_to_path=out_path, verbose=True)
    elif args.pred_path is not None and os.path.exists(args.pred_path):
        evaluate_from_pred_folder(pred_folder=args.pred_folder, refvg_split=args.split,
                                  analyze_subset=args.analyze_subset, exp_name_in_summary=exp_name,
                                  save_result_to_path=out_path, verbose=True)
    else:  # cases to evaluate from pred_dict
        if args.pred_dict is not None and os.path.exists(args.pred_dict):
            predictions = np.load(args.pred_path).item()
        else:
            if args.pred_name.startswith('vg_gt'):
                predictions = vg_gt_predictor(split=args.split)
            elif args.pred_name.startswith('vg_rand'):
                predictions = vg_rand_predictor(split=args.split)
            elif args.pred_name.startswith('ins_rand'):
                predictions = ins_rand_predictor(split=args.split)
            else:
                raise NotImplementedError

        predictions = evaluate_from_pred_dict(predictions=predictions, refvg_split=args.split,
                                              analyze_subset=args.analyze_subset, exp_name_in_summary=exp_name,
                                              save_result_to_path=out_path, update_predictions=args.save_pred,
                                              verbose=True)

        if args.save_pred:
            print('saving %s to %s' % (args.pred_name, out_path))
            pred_path = os.path.join(out_path, 'pred_eval.npy')
            np.save(pred_path, predictions)

    return


if __name__ == '__main__':
    main()
