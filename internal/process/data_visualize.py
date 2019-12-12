import json
import random

import matplotlib.pyplot as plt

plt.switch_backend('agg')

from _dataset.utils.visualize import visualize_refvg


def load_data(split=None):
    if split:
        fpath = 'data/refvg/turker_result/refer_filtered_instance_%s.json' % split
    else:
        fpath = 'data/refvg/turker_result/refer_filtered_instance.json'
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


def ref_visualize(split='val', img_count=10):
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

        title = '%d: %s' % (r['image_id'], r['phrase']) # + ':\n' + str(r['phrase_structure'].values())

        ins_boxes = r['instance_boxes']
        ins_polygons = r['Polygons']

        visualize_refvg(ax=axes.flatten()[vi], title=title, img_id=r['image_id'], gt_Polygons=ins_polygons,
                        gt_boxes=ins_boxes, vg_boxes=vg_boxes)  # , set_colors={'gt_polygons': 'colorful'})
        vi += 1
        if vi == img_count:
            break

    # plt.subplots_adjust(left=0.1, right=0.9, top=0.99, bottom=0.01)
    fig.tight_layout()
    fig.savefig('collect_data_visualize/visualize_%s.pdf' % split, dpi=300)
    print('\nsaved to file.')
    plt.close(fig)
    return


if __name__ == '__main__':
    # instance_visualize(split='val', img_count=20)
    ref_visualize(split='val', img_count=60)

