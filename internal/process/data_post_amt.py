import os
import json
import csv

import matplotlib.pyplot as plt

plt.switch_backend('agg')


def analyze_turkers(mask_json, csv_folder):
    turkers = {}
    print('loading mask data')
    with open(mask_json) as f:
        mask_data = json.load(f)

    print('analyzing iou/iob/iop')
    for entry in mask_data:
        turk_id = entry['turk_id']
        if turk_id not in turkers:
            turkers[turk_id] = {'iou_sum': entry['iou'], 'iob_sum': entry['iob'], 'iop_sum': entry['iop'],
                                'count': 1}
        else:
            turkers[turk_id]['iou_sum'] += entry['iou']
            turkers[turk_id]['iob_sum'] += entry['iob']
            turkers[turk_id]['iop_sum'] += entry['iop']
            turkers[turk_id]['count'] += 1

    for t_id in turkers.keys():
        e = turkers[t_id]
        count = e['count']
        iou = e['iou_sum'] / count
        iob = e['iob_sum'] / count
        iop = e['iop_sum'] / count
        score = iop + 0.8 * iou
        if count > 50:
            thresh = 0.7
        else:
            thresh = 0.95 - 0.005 * count
        if count > 10 and score > thresh:
            trust = 1
        else:
            trust = 0
        turkers[t_id] = {'count': count, 'iou': iou, 'iob': iob, 'iop': iop, 'score': score, 'thresh': thresh,
                         'trust': trust, 'turk_id': t_id, 'batches': set(), 'hit': 0}

    print('loading amt_collect csv files')
    fake_turkers = {}
    for csv_f in os.listdir(csv_folder):
        if csv_f.endswith(".csv"):
            batch = csv_f.split('.')[0].split('_')[1]
            with open(os.path.join(csv_folder, csv_f)) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    turk_id = row['WorkerId']
                    if turk_id in turkers:
                        turkers[turk_id]['batches'].add(batch)
                        turkers[turk_id]['hit'] += 1
                    else:
                        if turk_id in fake_turkers:
                            fake_turkers[turk_id]['batches'].add(batch)
                            fake_turkers[turk_id]['hit'] += 1
                        else:
                            fake_turkers[turk_id] = {'batches': set([batch]), 'hit': 1}

    print(len(turkers))
    print(len([t for t in turkers.values() if t['trust']]))
    print('save to csv and json file')
    with open('data/refvg/turker_result/turkers_v2.csv', 'w') as f:
        init = True
        for entry in turkers.values():
            entry['turk_id'] = entry['turk_id'].encode('utf8')
            entry['batches'] = list(entry['batches'])
            if init:
                w = csv.DictWriter(f, entry.keys())
                w.writeheader()
                init = False
            w.writerow(entry)

    with open('data/refvg/turker_result/turkers_v2.json', 'w') as f:
        json.dump(turkers, f)

    for turk in turkers.values():
        if turk['hit'] * 20 < 0.9 * turk['count']:
            print(turk)
    print('\n')

    for turk in turkers.values():
        if not turk['trust']:
            print(turk)
    print('\n')

    # bs = set(['3406990', '3406438', '3404025'])
    # for turk in turkers.values():
    #     if turk['hit'] * 20 * 0.9 > turk['count'] and bs.intersection(turk['batches']):
    #         print(turk)
    # print('\n')
    #
    # for turk in turkers.values():
    #     if not turk['trust'] and bs.intersection(turk['batches']):
    #         print(turk)
    # print('\n')

    for t in fake_turkers.values():
        print(t)
    print('\n')

    # for i, t in fake_turkers.items():
    #     if bs.intersection(t['batches']):
    #         print(i, t)

    return


def filter_refer(version='v2'):
    with open('data/refvg/turker_result/turkers_%s.json' % version) as f:
        turkers = json.load(f)

    with open('data/refvg/turker_result/refer_%s.json' % version) as f:
        ref_d = json.load(f)

    f_d = []
    acc = 0
    rej = 0
    print([t for t in turkers.values() if not t['trust']])
    for d in ref_d:
        if d['turk_id'] not in turkers:
            print(d['turk_id'])
            continue
        turker = turkers[d['turk_id']]
        score = d['iop'] + 0.8 * d['iou']
        if turker['trust'] or score > turker['thresh'] / (turker['score'] + 1.0e-8):
            f_d.append(d)
            acc += 1
        else:
            rej += 1
            # print(turker,score)

    print('%s: acc %.2f; rej %d' % (version, acc * 1.0 / len(ref_d), rej))

    with open('data/refvg/turker_result/refer_filtered_%s.json' % version, 'w') as f:
        json.dump(f_d, f)

    # f_slim = []
    # f_slim_keys = ['image_id', 'phrase', 'polygons']
    # for entry in f_d:
    #     es = dict()
    #     for k in f_slim_keys:
    #         es[k] = entry[k]
    #     f_slim.append(es)
    # with open('data/refvg/turker_result/refer_filtered_slim_%s.json' % split, 'w') as f:
    #     json.dump(f_slim, f)
    return


if __name__ == '__main__':
    # analyze_collected_data(mode='mask', start_id=0, end_id=-1, output_tag='_train', max_output_img=10)
    analyze_turkers('data/refvg/turker_result/refer_v2.json', 'data/refvg/turker_result/v2_batches')
    filter_refer()


# Obsolete
# def analyze_collected_data(mode, record_fpath=None, start_id=0, end_id=-1, max_output_img=10, count_per_img=60,
#                            output_tag=''):
#     # load data of masks / skipped
#     if not record_fpath:
#         # record_dataset = get_data_from_db(mode, start_id, end_id)
#         pass
#     else:
#         with open(record_fpath) as f:
#             record_dataset = json.load(f)
#         record_dataset = {task_id: record for task_id, record in record_dataset.items()[start_id:end_id]}
#
#     turker_dict = {}
#     involved_obj_ids = set()
#     involved_img_ids = set()
#     max_record_id = 0
#     for task_id, record in record_dataset.items():
#         ids = task_id.split('__')
#         image_id = int(ids[0])
#         object_ids = [int(i) for i in ids[1].split('_')]
#         record['image_id'] = image_id
#         record['object_ids'] = object_ids
#         involved_img_ids.add(image_id)
#         involved_obj_ids |= set(object_ids)
#         turk_id = record['turk_id']
#         max_record_id = max(max_record_id, record['record_id'])
#         if 'mask' in mode:
#             polygons = json.loads(record['polygons_str'])
#             # data = {'record_id': record['record_id'], 'image_id': image_id, 'object_ids': object_ids,
#             #         'turk_id': record['turk_id'], 'polygon_str': polygons}
#             record['polygon_str'] = polygons
#
#             if turk_id in turker_dict:
#                 turker_dict[turk_id]['mask_count'] += 1
#             else:
#                 turker_dict[turk_id] = {'mask_count': 1, 'pass_count': 0, 'iou': [], 'iob': [], 'iop': [], 'score': []}
#
#         else:  # if 'skip' in mode:
#             if turk_id in turker_dict:
#                 turker_dict[turk_id]['skip_count'] += 1
#             else:
#                 turker_dict[turk_id] = {'skip_count': 1, 'reasons': {}}
#             reason = record['reason']
#             turker_dict[turk_id]['reasons'][reason] = turker_dict[turk_id]['reasons'].get(reason, 0) + 1
#         record_dataset[task_id] = record
#         # print(task_id)
#
#     # load VG img_info data
#     with open('data/refvg/image_data_split1000.json', 'r') as f:
#         img_info = json.load(f)
#         info_dict = {}
#         for img in img_info:
#             if img['image_id'] in involved_img_ids:
#                 info_dict[img['image_id']] = [img['width'], img['height'], img['url']]
#
#     # load VG object data
#     with open('data/refvg/objects.json', 'r') as f:
#         imgs = json.load(f)
#         object_dict = {}
#         for img in imgs:
#             for obj in img['objects']:
#                 if obj['object_id'] in involved_obj_ids:
#                     object_dict[obj['object_id']] = obj
#
#     for record in record_dataset.values():
#         record['boxes'] = []
#         record['width'], record['height'], record['image_url'] = info_dict[record['image_id']]
#         for obj_id in record['object_ids']:
#             # x, y, w, h
#             obj = object_dict[obj_id]
#             x = obj['x']
#             y = obj['y']
#             w = obj['w']
#             h = obj['h']
#             box = [x, y, w, h]
#             record['boxes'].append(box)
#
#     # visualize
#     if max_output_img > 0:
#         fig, axes = plt.subplots(count_per_img / 2, 2, figsize=(6, 1.5 * count_per_img))
#     out_img_count = 0
#     # sorted_dataset = sorted(record_dataset.values(), key=lambda d: d['turk_id'])
#     sorted_dataset = record_dataset.values()
#     random.shuffle(sorted_dataset)
#     for i, record in enumerate(sorted_dataset):
#         if 'mask' in mode:
#             record['Polygons'] = []
#             names = []
#             for polygon in record['polygon_str']:
#                 if len(polygon['points']) < 3:
#                     print("WARNING: invalid polygon")
#                     continue
#                 points = np.zeros((len(polygon['points']), 2))
#                 for pi, p_str in enumerate(polygon['points']):
#                     xy_str = p_str.split(',')
#                     points[pi, 0] = int(xy_str[0]) / 600.0 * record['width']
#                     points[pi, 1] = int(xy_str[1]) / 600.0 * record['width']
#                 try:
#                     instance_id = names.index(polygon['name'])
#                 except ValueError:
#                     instance_id = len(names)
#                     names.append(polygon['name'])
#                 record['Polygons'].append({'instance_id': instance_id, 'points': points})
#
#             iou, iob, iop = iou_boxes_polygons(record['boxes'], record['polygons'], record['width'], record['height'],
#                                                xywh=True, ioubp=True)
#             score = iop + 0.8 * iou
#             # if iob >= 0.3 or iop >= 0.6:
#             if score > 0.5:
#                 passed = True
#             else:
#                 passed = False
#             record['iou'] = iou
#             record['iob'] = iob
#             record['iop'] = iop
#             record['score'] = score
#             title = '[%d] %s\niou=%.2f,iob=%.2f,iop=%.2f, score=%.2f \nturk=%s' \
#                     % (record['record_id'], record['phrase'], iou, iob, iop, score, record['turk_id'])
#             turker_dict[record['turk_id']]['iou'].append(iou)
#             turker_dict[record['turk_id']]['iob'].append(iob)
#             turker_dict[record['turk_id']]['iop'].append(iop)
#             turker_dict[record['turk_id']]['score'].append(score)
#             if passed:
#                 turker_dict[record['turk_id']]['pass_count'] += 1
#
#         else:  # 'skip' in mode
#             title = '[%d] %s\n%s\nturk=%s' \
#                     % (record['record_id'], record['phrase'], record['reason'], record['turk_id'])
#
#         print(title)
#         if out_img_count < max_output_img:
#             ax = axes.flatten()[i]
#             img_url = record.get('image_url', None)
#             Polygons = None
#             t_color = 'black'
#             if 'mask' in mode:
#                 Polygons = record['Polygons']
#                 if not passed: t_color = 'red'
#
#             visualize_refvg(ax, title=title, img_url=img_url, gt_Polygons=Polygons, vg_boxes=record['boxes'],
#                             set_colors={'gt_polygons': 'colorful', 'title': t_color})
#
#             print('processed img %d' % i)
#             if i % count_per_img == count_per_img - 1 or i == len(sorted_dataset) - 1:
#                 if 'mask' in mode:
#                     mode_tag = 'masks'
#                 else:
#                     mode_tag = 'skipped'
#                 fname = 'visualize_%s%s_%d_%d_%d.pdf' % (mode_tag, output_tag, start_id, max_record_id, out_img_count)
#                 plt.subplots_adjust(left=0.1, right=0.9, top=0.99, bottom=0.01)
#                 fig.tight_layout()
#                 fig.savefig(os.path.join('collect_data_visualize', fname))
#                 print('\nsaved to %s' % fname)
#                 plt.close(fig)
#                 if i != len(sorted_dataset) - 1:
#                     fig, axes = plt.subplots(30, 2, figsize=(6, 3 * 30))
#                 out_img_count += 1
#
#     if 'mask' in mode:
#         with open('collect_data_visualize/turkers_mask%s_%d_%d.txt' % (output_tag, start_id, max_record_id), 'w') as f:
#             for t_id, turker in sorted(turker_dict.items()):
#                 acc = turker['pass_count'] * 1.0 / turker['mask_count']
#                 iou = np.sum(turker['iou']) / turker['mask_count']
#                 iob = np.sum(turker['iob']) / turker['mask_count']
#                 iop = np.sum(turker['iop']) / turker['mask_count']
#                 score = np.sum(turker['score']) / turker['mask_count']
#                 s = '%s: total:%d, acc=%.2f, iou=%.2f, iob=%.2f, iop=%.2f, score=%.2f' \
#                     % (t_id, turker['mask_count'], acc, iou, iob, iop, score)
#                 print(s)
#                 f.write(s + '\n')
#
#     else:
#         with open('collect_data_visualize/turkers_skip%s_%d_%d.txt' % (output_tag, start_id, max_record_id), 'w') as f:
#             for t_id, turker in sorted(turker_dict.items()):
#                 s = '%s: total: %d reasons: %s' % (t_id, turker['skip_count'], turker['reasons'])
#                 print(s)
#                 f.write(s + '\n')


# # one time use. remix test/val
# def remix():
#     t_refer = json.load(open('data/refvg/turker_result/refer_test.json'))
#     v_refer = json.load(open('data/refvg/turker_result/refer_val.json'))
#     t_skip = json.load(open('data/refvg/turker_result/skip_test.json'))
#     v_skip = json.load(open('data/refvg/turker_result/skip_val.json'))
#     img_ids = [r['image_id'] for r in t_refer + v_refer + t_skip + v_skip]
#     img_ids = list(set(img_ids))
#     random.shuffle(img_ids)
#     test_img_count = len(img_ids) / 2
#     print('total imgs %d. test %d' % (len(img_ids), len(img_ids) / 2))
#     test_img_ids = img_ids[: test_img_count]
#     # val_img_ids = img_ids[test_img_count:]
#     test_refer = []
#     val_refer = []
#     tr_old_count = 0
#     vr_old_count = 0
#     for r in t_refer + v_refer:
#         if 'phrase_structure' not in r:
#             r['phrase_structure'] = None
#         if r['image_id'] in test_img_ids:
#             test_refer.append(r)
#             if not r['phrase_structure']:
#                 tr_old_count += 1
#         else:
#             val_refer.append(r)
#             if not r['phrase_structure']:
#                 vr_old_count += 1
#     print('test_refer %d old %d' % (len(test_refer), tr_old_count))
#     print('val_refer %d old %d' % (len(val_refer), vr_old_count))
#
#     test_skip = []
#     val_skip = []
#     ts_old_count = 0
#     vs_old_count = 0
#     for r in t_skip + v_skip:
#         if 'phrase_structure' not in r:
#             r['phrase_structure'] = None
#         if r['image_id'] in test_img_ids:
#             test_skip.append(r)
#             if not r['phrase_structure']:
#                 ts_old_count += 1
#         else:
#             val_skip.append(r)
#             if not r['phrase_structure']:
#                 vs_old_count += 1
#     print('test_skip %d old %d' % (len(test_skip), ts_old_count))
#     print('val_skip %d old %d' % (len(val_skip), vs_old_count))
#
#     with open('data/refvg/turker_result_remix/refer_test.json', 'w') as f:
#         json.dump(test_refer, f)
#     with open('data/refvg/turker_result_remix/refer_val.json', 'w') as f:
#         json.dump(val_refer, f)
#     with open('data/refvg/turker_result_remix/skip_test.json', 'w') as f:
#         json.dump(test_skip, f)
#     with open('data/refvg/turker_result_remix/skip_val.json', 'w') as f:
#         json.dump(val_skip, f)
#     return
#
