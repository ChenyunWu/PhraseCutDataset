import json
import random

import matplotlib.pyplot as plt

plt.switch_backend('agg')

from _dataset.utils.data_transfer import *
from _dataset.utils.iou import *


def get_instance_boxes(vg_boxes, phrase, name, polygons=None, Polygons=None, xywh=True, verbose=False):
    def merge_boxes(boxes):
        x1 = 1e8
        y1 = 1e8
        x2 = y2 = 0
        for b in boxes:
            x1 = b[0] if b[0] < x1 else x1
            y1 = b[1] if b[1] < y1 else y1
            bx2 = b[0] + b[2] + 1
            by2 = b[1] + b[3] + 1
            x2 = bx2 if bx2 > x2 else x2
            y2 = by2 if by2 > y2 else y2
        w = x2 - x1 + 1
        h = y2 - y1 + 1
        return [x1, y1, w, h]

    def plural_phrase(phrase1, name1):
        if not phrase1:
            return False
        plural = False
        if not name1:
            name1 = phrase1
        if name1[-1] == 's':
            plural = True
            for w in ['grass', 'glass', 'bus', 'jeans', 'pants', 'dress', 'moss', 'harness']:
                if w in name1:
                    plural = False
                    break

        for w in ['people','children', 'men', 'women',
                  'many', 'several', 'multiple', 'two', 'three', 'four', 'five', 'six']:
            if w in phrase1:
                plural = True
        return plural

    if not xywh:
        vg_boxes = xyxy_to_xywh(vg_boxes)

    # all boxes from polygons
    ins_boxes = []
    if Polygons:
        polygons = []
        ins_polygons = Polygons
        for ins_pos in Polygons:
            polygons += ins_pos
            xyxy = boxes_region([polygon_to_box(p) for p in ins_pos])
            ins_boxes += xyxy_to_xywh([xyxy]).tolist()
    else: # polygons
        for polygon in polygons:
            ins_boxes.append(polygon_to_box(polygon))
        ins_polygons = [[p] for p in polygons]

    merge_info = ''
    list_str = lambda l: str(np.array(l).astype(int))

    if verbose:
        print('Start "%s" : VG_boxes %d, polygons %d' % (phrase, len(vg_boxes), len(polygons)))
    # STEP 1: merge by VG boxes: an occluded instance (one vg_box) split into multiple polygons
    if not plural_phrase(phrase, name):
        # correspondence between VG boxes and ins boxes
        vg_boxes.sort(key=lambda b: b[2] * b[3])  # small to large
        match_table = np.zeros((len(vg_boxes), len(ins_boxes)))
        merged_ins_idxs = []
        merged_boxes = []
        merged_polygons = []
        for vg_i, vg_b in enumerate(vg_boxes):
            for ins_i, (ins_b, ins_ps) in enumerate(zip(ins_boxes, ins_polygons)):
                s_ins = ins_b[2] * ins_b[3]
                s_vg = vg_b[2] * vg_b[3]
                ratio = s_ins * 1.0 / s_vg
                iou, io_vg, io_ins = iou_boxes_polygons([vg_b], ins_ps, xywh=True, ioubp=True)
                if verbose and iou > 0:
                    print('1. %s vs. %s: iou=%.2f io_ins=%.2f ratio=%.2f'
                          % (list_str(vg_b), list_str(ins_b), iou, io_ins, ratio))
                if io_ins > 0.9 or iou > 0.4 or (io_ins > 0.4 + ratio):
                    match_table[vg_i, ins_i] = 1
        # merge ins boxes within each vg box
        for vg_i, vg_b in enumerate(vg_boxes):
            to_merge_ids = [i for i in range(len(ins_boxes)) if match_table[vg_i, i] > 0]
            to_merge_boxes = [ins_boxes[i] for i in to_merge_ids]
            if len(to_merge_boxes) < 2:
                continue
            merged_b = merge_boxes(to_merge_boxes)
            iou = iou_box(merged_b, vg_b, xywh=True, ioubp=False)
            if verbose:
                print('1. %s merged_ins_b: iou=%.2f | 0.5' % (list_str(vg_b), iou))
            if iou > 0.5:
                merged_boxes.append(merged_b)
                merged_ins_p = []
                for i in to_merge_ids:
                    merged_ins_p += ins_polygons[i]
                merged_polygons.append(merged_ins_p)
                merged_ins_idxs += to_merge_ids
                merge_info += 'VG_merge: iou=%.2f %s\n' % (iou, list_str(to_merge_boxes))
                for i in to_merge_ids:
                    match_table[:, i] = 0
        # collect result
        remain_boxes = merged_boxes
        remain_polygons = merged_polygons
        for ins_i, ins_b in enumerate(ins_boxes):
            if ins_i not in merged_ins_idxs:
                remain_boxes.append(ins_b)
                remain_polygons.append(ins_polygons[ins_i])
        ins_boxes = remain_boxes
        ins_polygons = remain_polygons

    # STEP 2: self-merge: used multiple polygons to cover one instance
    filtered_boxes = []
    filtered_polygons = []
    merged = True
    pass_count = 0
    while merged and pass_count < 3:
        zip_ins = zip(ins_boxes, ins_polygons)
        zip_ins.sort(key=lambda ins: - ins[0][2] * ins[0][3])
        # ins_boxes.sort(key=lambda b: -b[2] * b[3])  # large to small
        merged = False
        for cur_b, cur_ps in zip_ins:
            valid_box = True
            for fi, (fb, fps) in enumerate(zip(filtered_boxes, filtered_polygons)):
                s_cur = cur_b[2] * cur_b[3]
                s_f = fb[2] * fb[3]
                iou, io_cur, iof = iou_box(cur_b, fb, ioubp=True)
                iou_p = iou_boxes_polygons([cur_b], fps, ioubp=False)
                ratio = s_cur / s_f
                if verbose and iou > 0:
                    print('2. io_cur=%.2f ratio=%.2f %s vs. %s' % (io_cur, ratio, list_str(cur_b), list_str(fb)))
                # if (io_cur > 0.1 and ratio < 0.05) or (io_cur > 0.6 and ratio < 0.2) \
                if (io_cur > 0.3 + ratio * 2  or io_cur > 0.9) and iou_p > 0:
                    valid_box = False
                    merged = True
                    filtered_boxes[fi] = merge_boxes([fb, cur_b])
                    filtered_polygons[fi] += cur_ps
                    merge_info += 'Self_merge: io_cur=%.2f ratio=%.2f %s --> %s\n' % (io_cur, ratio, list_str(cur_b),
                                                                                      list_str(fb))
                    break
            if valid_box:
                filtered_boxes.append(cur_b)
                filtered_polygons.append(cur_ps)
        ins_boxes = filtered_boxes
        ins_polygons = filtered_polygons
        filtered_boxes = list()
        filtered_polygons = list()
        pass_count += 1

    splited = False
    # STEP 3: split: one polygon covers multi instances (multi vg boxes)
    if len(ins_boxes) < 10:
        split_ins_idxs = []
        new_ins_boxes = []
        new_ins_polygons = []
        zipped = zip(ins_boxes, ins_polygons)
        zipped.sort(key=lambda z: -z[0][2] * z[0][3] )# large to small
        for ins_i, (ins_b, ins_ps) in enumerate(zipped):
            # vg boxs mostly inside the ins box
            anchor_boxes = []
            for vg_i, vg_b in enumerate(vg_boxes):
                iou, io_vg, io_ins = iou_box(vg_b, ins_b, ioubp=True)
                if verbose and iou > 0:
                    print('3. vg_box in ins box: io_vg=%.2f | 0.7, %s in %s' % (io_vg, list_str(vg_b), list_str(ins_b)))
                if io_vg > 0.7:
                    anchor_boxes.append(vg_b)
            if len(anchor_boxes) < 2:
                continue

            # consider multi anchor boxes as one instance if they overlap too much
            # ignore anchor boxes with too few polygon mask inside it
            if verbose:
                print('3. # anchor_boxes = %d' % len(anchor_boxes))
            anchor_boxes.sort(key=lambda b: -b[2] * b[3])
            v_anchor_boxes = []
            for a_b in anchor_boxes:
                iou, iob, iop = iou_boxes_polygons([a_b], ins_ps, ioubp=True)
                if verbose:
                    print('%s iob=%.2f | 0.4, iop=%.2f | 0.1' % (list_str(a_b), iob, iop))
                if iob < 0.4 or iop < 0.1:
                    continue
                is_valid = True
                for v_i, v_b in enumerate(v_anchor_boxes):
                    iou, ioa, iov = iou_box(a_b, v_b, ioubp=True)
                    if verbose:
                        print('%s vs %s iou=%.2f | 0.4, ioa=%.2f | 0.8' % (list_str(a_b), list_str(v_b), iou, ioa))
                    if iou > 0.4 or ioa > 0.8:
                        is_valid = False
                        v_anchor_boxes[v_i] = merge_boxes([v_b, a_b])
                        break
                if is_valid:
                    v_anchor_boxes.append(a_b)
            if verbose:
                print('3. # v_anchor_boxes = %d' % len(v_anchor_boxes))
            if len(v_anchor_boxes) < 2:
                continue

            # the mask from polygons should be almost covered by the vg boxes
            iou, iob, iop = iou_boxes_polygons(v_anchor_boxes, ins_ps, ioubp=True)
            if verbose and iou > 0:
                print('3. polygons covered: iop=%.2f | 0.8, %s' % (iop, list_str(ins_b)))
            if iop < 0.8:
                continue

            # split using anchor boxes on the mask
            split_ins_idxs.append(ins_i)
            merge_info += 'Split: %s --> ' % list_str(ins_b)

            p_boxes = [polygon_to_box(p) for p in ins_ps]
            region = boxes_region(p_boxes + v_anchor_boxes)
            # w = int(region[2] - region[0] + 1)
            # h = int(region[3] - region[1] + 1)
            pw = int(region[2] + 1)
            ph = int(region[3] + 1)
            mask_ins = polygons_to_mask(ins_ps, pw, ph)
            mask_dis = (mask_ins * (pw**2 + ph**2)).astype(float)
            mask_i = mask_ins * -1
            anchor_xyxy_boxes = xywh_to_xyxy(v_anchor_boxes)
            for ai, xyxy in enumerate(anchor_xyxy_boxes):
                for px in range(pw):
                    for py in range(ph):
                        if not mask_ins[px, py]:
                            continue
                        dx = max(xyxy[0] - px, 0, px - xyxy[2])
                        dy = max(xyxy[1] - py, 0, py - xyxy[3])
                        dxy = np.sqrt(dx ** 2 + dy ** 2) + random.random()
                        if mask_dis[px, py] > dxy:
                            mask_dis[px, py] = dxy
                            mask_i[px, py] = ai + 1
            # if verbose:
            #     print('mask_i', set(mask_i.flatten()))
            #     print('mask_dis', np.min(mask_dis), np.max(mask_dis))
            new_xyxy = []
            for ai, ab in enumerate(v_anchor_boxes):
                if np.sum(mask_i == ai + 1) == 0:
                    print('WARNING: invalid box', ab)
                    continue
                xsum = np.sum(mask_i == ai + 1, axis=1)
                xrange = np.nonzero(xsum > 0)[0]
                x0 = xrange[0]
                x1 = xrange[-1]
                ysum = np.sum(mask_i == ai + 1, axis=0)
                yrange = np.nonzero(ysum > 0)[0]
                y0 = yrange[0]
                y1 = yrange[-1]
                if verbose:
                    print('3. split boxes')
                    # print(xsum,xrange, x0, x1)
                    # print(ysum,yrange, y0, y1)
                    splited = True
                new_xyxy.append([x0, y0, x1, y1])

            # polygons cropped by the boxes
            for xyxy in new_xyxy:
                new_ps = []
                for p in ins_ps:
                    new_p = polygon_in_box(p, xyxy, xywh=False)
                    if new_p:
                        new_ps.append(new_p)
                        # if len(new_p) != len(p):
                        #     print('polygon_in_box: ', xyxy)
                        #     print('old polygon: ' + list_str(p))
                        #     print('new polygon: ' + list_str(new_p))
                if new_ps:
                    new_ins_polygons.append(new_ps)
                    # ins_box may be smaller by polygons
                    ins_box_xyxy = boxes_region([polygon_to_box(p) for p in new_ps])
                    new_ins_boxes += xyxy_to_xywh([ins_box_xyxy]).tolist()
            merge_info += list_str(new_ins_boxes)

        # collect result
        merge_info += '\n'
        for i in range(len(ins_boxes)):
            if i not in split_ins_idxs:
                new_ins_boxes.append(ins_boxes[i])
                new_ins_polygons.append(ins_polygons[i])
        ins_boxes = new_ins_boxes
        ins_polygons = new_ins_polygons
    if verbose:
        print(splited)
        print(merge_info)

    return ins_boxes, ins_polygons, splited


def load_data(split=None):
    fpath = 'data/refvg/turker_result/refer_filtered.json'
    with open(fpath) as f:
        records = json.load(f)

    print('Loader loading image_data_split3000.json')
    with open('data/refvg/image_data_split3000.json', 'r') as f:
        imgs_info = json.load(f)
        info_dict = {img['image_id']: img for img in imgs_info}

    if split:
        records = [r for r in records if info_dict[r['image_id']]['split'] == split]

    # load VG scene_graphs data
    print('loading scene_graphs data...')
    if not split:
        with open('data/refvg/scene_graphs_pp.json', 'r') as f:
            imgs = json.load(f)
            print(len(imgs), 'images')
    else:
        with open('data/refvg/scene_graphs_pp_%s.json' % split, 'r') as f:
            imgs = json.load(f)
            print(split, len(imgs), 'images')

    print('Building obj_dict...')
    object_dict = dict()
    for img in imgs:
        for obj in img['objects']:
            object_dict[obj['object_id']] = obj

    print('data is ready.')
    return records, object_dict


def instance_visualize(split='val', img_count=10):
    records, object_dict = load_data(split)

    # records = [r for r in records if len(r['polygons']) > 1 or len(r['ann_ids']) > 1]
    # records = [r for r in records if len(r['ann_ids']) > 1]
    records = random.sample(records, img_count)
    fig, axes = plt.subplots(img_count / 2, 2, figsize=(6, 1.2 * img_count))

    vi = 0
    for i, r in enumerate(records):
        print('img %d/%d: %d' % (i, img_count, r['image_id']))
        vg_objs = []
        for ann_id in r['ann_ids']:
            if ann_id not in object_dict:
                print('missing obj:', r['image_id'], ann_id)
                continue
            vg_objs.append(object_dict[ann_id])
        vg_boxes = [[obj['x'], obj['y'], obj['w'], obj['h']] for obj in vg_objs]
        name = None
        if r['phrase_structure']:
            name = r['phrase_structure']['name']

        title = '%d: %s' % (r['image_id'], r['phrase']) # + ':\n' + str(r['phrase_structure'].values())

        if 'Polygons' in r:
            ins_boxes, ins_polygons, step3 = get_instance_boxes(vg_boxes, r['phrase'], name, Polygons=r['Polygons'])
        else:
            ins_boxes, ins_polygons, step3 = get_instance_boxes(vg_boxes, r['phrase'], name, polygons=r['polygons'],
                                                        verbose=True)
        # if step3:
        if True:
            visualize_refvg(ax=axes.flatten()[vi], title=title, img_id=r['image_id'],gt_Polygons=ins_polygons,
                            gt_boxes=ins_boxes, vg_boxes=vg_boxes, set_colors={'gt_polygons': 'colorful'})
            vi += 1
            if vi == img_count:
                break

    # plt.subplots_adjust(left=0.1, right=0.9, top=0.99, bottom=0.01)
    fig.tight_layout()
    fig.savefig('collect_data_visualize/visualize_instance_%s.pdf' % split, dpi=300)
    print('\nsaved to file.')
    plt.close(fig)
    return


def add_instance_boxes():
    """
    Original record: {'task_id', 'image_id', 'ann_ids', 'phrase', 'phrase_structure',
                      'Polygons'(only for train) or 'polygons', (not filtered: 'turk_id', 'iou', 'iob', 'iop')}
    This function: add 'instance_boxes' to each record
    :param split:
    :return:
    """
    records, object_dict = load_data(None)
    n_records = []
    c = 0
    for i, r in enumerate(records):
        if i % 500 == 0:
            print('task %d/%d: %d' % (i, len(records), r['image_id']))
        vg_objs = []
        for ann_id in r['ann_ids']:
            if ann_id not in object_dict:
                print('missing obj:', r['image_id'], ann_id)
                continue
            vg_objs.append(object_dict[ann_id])
        vg_boxes = [[obj['x'], obj['y'], obj['w'], obj['h']] for obj in vg_objs]
        name = None
        if r['phrase_structure']:
            name = r['phrase_structure']['name']
        if 'Polygons' in r:
            if len([p for p in r['Polygons']]) == 0:
                c += 1
                print(c, "P-WARNING:", r['task_id'])
                continue
            ins_boxes, ins_polygons, _ = get_instance_boxes(vg_boxes, r['phrase'], name, Polygons=r['Polygons'])
        else:
            if len(r['polygons']) == 0:
                c += 1
                print(c, "p-WARNING:", r['task_id'])
                continue
            ins_boxes, ins_polygons, _ = get_instance_boxes(vg_boxes, r['phrase'], name, polygons=r['polygons'])
        r['instance_boxes'] = ins_boxes
        r.pop('Polygons', None)
        r.pop('polygons', None)
        r['Polygons'] = ins_polygons
        n_records.append(r)
    print(c)
    print(len(n_records))
    with open('data/refvg/turker_result/refer_filtered_instance.json', 'w') as f:
        json.dump(n_records, f)
    print('saved to refer_filtered_instance.json')
    return

if __name__ == '__main__':
    add_instance_boxes()
    # instance_visualize(split='val', img_count=20)
    # instance_visualize(split='test', img_count=20)

