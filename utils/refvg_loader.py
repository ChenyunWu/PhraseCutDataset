from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import random

from .vg_loader import VGLoader as VGLoader


class RefVGLoader:

    def __init__(self, split=None, rebalance=False, vg_loader=None, obj_filter=False, allow_no_structure=False,
                 word_embed=None):
        # parent loader instance
        self.vg_loader = vg_loader
        if not vg_loader:
            self.vg_loader = VGLoader(split=split, obj_filter=obj_filter, word_embed=word_embed)

        ref_tasks = []
        if not split:
            ss = ['miniv', 'val', 'test', 'train']
        else:
            ss = split.split('_')

        for s in ss:
            if rebalance:
                print('RefVGLoader loading refer_filtered_instance_rebalance_%s.json' % s)
                with open('data/refvg/amt_result/refer_filtered_instance_rebalance_%s.json' % s, 'r') as f:
                    ref_tasks += json.load(f)
            else:
                print('RefVGLoader loading refer_filtered_instance_%s.json' % s)
                with open('data/refvg/amt_result/refer_filtered_instance_%s.json' % s, 'r') as f:
                    ref_tasks += json.load(f)

        print('RefVGLoader preparing data')
        self.ImgInsBoxes = {}
        self.ImgInsPolygons = {}
        self.ImgReferTasks = {}
        loaded_img_ids = []
        for task in ref_tasks:
            if not allow_no_structure and not task['phrase_structure']:
                continue
            img_id = task['image_id']
            if 'relations' in task['phrase_structure']:
                task['phrase_structure']['rel_descriptions'] = self.get_rel_descriptions(task['phrase'],
                                                                                         task['phrase_structure'])
            if img_id in self.ImgReferTasks.keys():
                self.ImgReferTasks[img_id].append(task)
                self.ImgInsBoxes[img_id] += task['instance_boxes']
                self.ImgInsPolygons[img_id] += task['Polygons']
                task['ins_box_ixs'] = range(len(self.ImgInsBoxes[img_id]) - len(task['instance_boxes']),
                                            len(self.ImgInsBoxes[img_id]))
            else:
                self.ImgReferTasks[img_id] = [task]
                self.ImgInsBoxes[img_id] = task['instance_boxes'][:]
                self.ImgInsPolygons[img_id] = task['Polygons'][:]
                loaded_img_ids.append(img_id)
                task['ins_box_ixs'] = range(len(task['instance_boxes']))

        print('RefVGLoader spliting img_ids')
        self.img_ids = [img_id for img_id, img in self.vg_loader.images.items()
                        if img['split'] in ss and img_id in loaded_img_ids]
        self.img_ids.sort()
        self.iterator = 0
        print('RefVGLoader ready.')

    def shuffle(self):
        random.shuffle(self.img_ids)

    def get_rel_descriptions(self, phrase, p_struct):
        predicates = [rel['predicate'] for rel in p_struct['relations']]
        if len(predicates) == 0:
            return []
        ph_segs = []
        good_p = []
        for p in predicates:
            if p in phrase:
                segs = phrase.split(p, 1)
                ph_segs.append(segs[0])
                good_p.append(p)
                phrase = segs[1]
        ph_segs.append(phrase)
        ds = []
        for i in range(len(good_p)):
            d = good_p[i] + ' ' + ph_segs[i + 1]
            d = ' '.join(d.split())
            ds.append(d)
        if len(ds) == 0:
            w_na = p_struct['name'].split() + [a.split() for a in p_struct['attributes']]
            w_p = phrase.split()
            w_r = w_p[len(w_na):]
            rd = ' '.join(w_r)
            ds = [rd]
            # print('get_rel_descriptions:', phrase, '-->', rd, p_struct)
        return ds

    def get_img_ref_data(self, img_id=-1):
        """
        get a batch with one image and all refer data on that image
        """
        # Fetch feats according to the image_split_ix
        wrapped = False
        max_index = len(self.img_ids) - 1

        if img_id < 0:
            ri = self.iterator
            ri_next = ri + 1
            if ri_next > max_index:
                ri_next = 0
                wrapped = True
            self.iterator = ri_next
            img_id = self.img_ids[ri]

        vg_img = self.vg_loader.images[img_id]
        vg_ann_id_set = set()
        vg_ann_ids = []
        vg_boxes = []

        img_ann_ids = vg_img['ann_ids']
        img_vg_boxes = [self.vg_loader.objects[ann_id]['box'] for ann_id in img_ann_ids]
        img_ins_boxes = self.ImgInsBoxes[img_id]
        img_ins_Polygons = self.ImgInsPolygons[img_id]
        for task in self.ImgReferTasks[img_id]:
            # for i in task['ann_ids']:
                # if i not in img_ann_ids:
                #     print('ann_id not in vg_pp: %s' % i)
            task_ann_ids = [i for i in task['ann_ids'] if i in img_ann_ids]
            vg_ann_ids.append(task_ann_ids)
            vg_ann_id_set.update(task_ann_ids)
            vg_boxes.append([self.vg_loader.objects[ann_id]['box'] for ann_id in task_ann_ids])

        phrases = []
        task_ids = []
        p_structures = []
        gt_Polygons = []
        gt_boxes = []
        img_ins_names = []
        img_ins_atts = []
        # img_ins_rels = []
        for task in self.ImgReferTasks[img_id]:
            phrases.append(task['phrase'])
            task_ids.append(task['task_id'])
            gt_boxes.append(task['instance_boxes'])
            p_structures.append(task['phrase_structure'])
            gt_Polygons.append(task['Polygons'])
            img_ins_names += [task['phrase_structure']['name']] * len(task['instance_boxes'])
            img_ins_atts += [task['phrase_structure']['attributes']] * len(task['instance_boxes'])
            # img_ins_rels += [task['phrase_structure']['relations']] * len(task['instance_boxes'])

        # return data
        data = dict()
        data['image_id'] = img_id
        data['width'] = vg_img['width']
        data['height'] = vg_img['height']
        data['split'] = vg_img['split']

        data['task_ids'] = task_ids
        data['phrases'] = phrases
        data['p_structures'] = p_structures

        data['img_vg_boxes'] = img_vg_boxes
        data['img_ins_boxes'] = img_ins_boxes
        data['img_ins_Polygons'] = img_ins_Polygons
        data['img_ins_names'] = img_ins_names
        data['img_ins_atts'] = img_ins_atts
        # data['img_ins_rels'] = img_ins_rels
        data['gt_Polygons'] = gt_Polygons
        data['gt_boxes'] = gt_boxes
        data['vg_boxes'] = vg_boxes
        data['vg_ann_ids'] = vg_ann_ids

        data['bounds'] = {'it_pos_now': self.iterator, 'it_max': max_index, 'wrapped': wrapped}

        return data

