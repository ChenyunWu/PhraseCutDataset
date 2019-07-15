"""
scene_graphs.json:
list of scene graphs (one for each image, dict[u'relationships', u'image_id', u'objects'])
image_id: not continuous or increasing! min=1, max=2417997. 108077 images in total
objects: a list of dicts: object_id, x,y,w,h, attributes(list of str), names(list of str), synsets. (0 ~ 207 objects)
relationships: a list of dicts: subject_id, object_id, relationship_id, predicate, synsets (0 ~ 800 relationships)

img_data.json:
list of img info (dict: image_id, coco_id, flickr_id, width, height, url)
img_data_split.json:
add split: 'train'/'val'/'test' to img info

"""
# from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import random

from .iou import iou_box


class VGLoader(object):

    def __init__(self, word_embed=None, split=None, obj_filter=True, phrase_length=10, name_count_top=1272,
                 att_count_top=593, rel_count_top=126, obj_size_thresh=0.005, iou_st_thresh=0.5, iou_dt_thresh=0.8):

        # self.word_to_ix = self.info['word_to_ix']
        # self.ix_to_word = {ix: wd for wd, ix in self.word_to_ix.items()}
        # print('vocab size is ', self.vocab_size)
        # self.cat_to_ix = self.info['cat_to_ix']
        # self.ix_to_cat = {ix: cat for cat, ix in self.cat_to_ix.items()}
        # print('object cateogry size is ', len(self.ix_to_cat))

        self.word_embed = word_embed
        self.phrase_length = phrase_length
        self.obj_size_thresh = obj_size_thresh
        self.iou_st_thresh = iou_st_thresh
        self.iou_dt_thresh = iou_dt_thresh
        self.phrase_type_stat = {'name': 0, 'attribute': 0, 'relation': 0, 'location': 0, 'verbose': 0}

        # load the json files which contains info about the dataset
        print('Loader loading image_data_split3000.json')
        with open('data/refvg/image_data_split3000.json', 'r') as f:
            imgs_info = json.load(f)
            self.info_dict = {img['image_id']: img for img in imgs_info}

        print('Loader loading name_att_rel_count_amt.json')
        with open('data/refvg/amt_result/name_att_rel_count_amt.json', 'r') as f:

            count_info = json.load(f)
            name_count = count_info['name']
            att_count = count_info['att']
            rel_count = count_info['rel']

        name_count_sorted = name_count.values()
        name_count_sorted.sort()
        self.name_count_thresh = name_count_sorted[-name_count_top]
        self.name_num = len(name_count_sorted) - name_count_sorted.index(self.name_count_thresh)
        print('Number of categories: %d / %d, frequency thresh: %d'
              % (self.name_num, len(name_count), self.name_count_thresh))
        self.name_to_cnt = {k: v for k, v in name_count.items() if v >= self.name_count_thresh}
        self.ix_to_name = self.name_to_cnt.keys()
        self.name_to_ix = {name: ix for ix, name in enumerate(self.ix_to_name)}

        # prepare attributes
        att_count_sorted = att_count.values()
        att_count_sorted.sort()
        self.att_count_thresh = att_count_sorted[-att_count_top]
        self.att_num = len(att_count_sorted) - att_count_sorted.index(self.att_count_thresh)
        print('Number of valid unique attributes: %d / %d, frequency thresh: %d'
              % (self.att_num, len(att_count), self.att_count_thresh))
        self.att_to_cnt = {k: v for k, v in att_count.items() if v >= self.att_count_thresh}
        self.ix_to_att = self.att_to_cnt.keys()
        self.att_to_ix = {att: ix for ix, att in enumerate(self.ix_to_att)}

        rel_count_sorted = rel_count.values()
        rel_count_sorted.sort()
        self.rel_count_thresh = rel_count_sorted[-rel_count_top]
        self.rel_num = len(rel_count_sorted) - rel_count_sorted.index(self.rel_count_thresh)
        print('Number of valid unique relation predicates: %d / %d, frequency thresh: %d'
              % (self.rel_num, len(rel_count), self.rel_count_thresh))

        if not split:
            split = 'train_val_test_miniv'
        
        print('split: %s' % split)
        
        # print('Loader loading scene_graphs_pp.json')
        # with open('data/refvg/scene_graphs_pp.json', 'r') as f:
        #     self.images = json.load(f)
        # else:
        ss = split.split('_')
        self.images = []
        for s in ss:
            print('Loader loading scene_graphs_pp_%s.json' % s)
            with open('data/refvg/scene_graphs_pp_%s.json' % s, 'r') as f:
                self.images += json.load(f)
                
        self.relations = []
        self.Anns = dict()
        self.Images = dict()
        self.obj_ids = []
        for img in self.images:
            img['file_name'] = '%d.jpg' % img['image_id']
            img_info = self.info_dict[img['image_id']]
            img['width'] = img_info['width']
            img['height'] = img_info['height']
            img['split'] = img_info['split']
            self.Images[img['image_id']] = img
            self.obj_ids += [obj['object_id'] for obj in img['objects']]

            # to match the I/O for refCOCOx
            if obj_filter:
                f_objects, f_relations = self.filter_objects(img_info, img['objects'], img['relationships'])
            else:
                f_objects = img['objects']
                f_relations = img['relationships']
            img['ann_ids'] = [obj['object_id'] for obj in f_objects]
            self.relations += f_relations
            for obj in f_objects:
                # COCO anns: list of {ann_id, category_id, image_id, box, h5_id}
                attributes = []
                if 'attributes' in obj:
                    attributes = obj['attributes']
                self.Anns[obj['object_id']] = {'ann_id': obj['object_id'], 'image_id': img['image_id'],
                                               'box': [obj['x'], obj['y'], obj['w'], obj['h']],
                                               'names': obj['names'],
                                               'attributes': attributes,
                                               'synsets': obj['synsets'],
                                               'relations': []}

            for rel in f_relations:
                self.Anns[rel['subject_id']]['relations'].append(rel)

        self.obj_ids.sort()
        for h_id, ann_id in enumerate(self.obj_ids):
            if self.Anns.has_key(ann_id):
                self.Anns[ann_id]['h5_id'] = h_id

        print('we have %s images.' % len(self.images))
        print('we have %s (filtered) objects, %.1f per image.'
              % (len(self.Anns), len(self.Anns) * 1.0 / len(self.images)))
        print('we have %s relations, %.1f per image.'
              % (len(self.relations), len(self.relations) * 1.0 / len(self.images)))

    @property
    def vocab_size(self):
        return len(self.word_embed.word_to_ix)

    # @property
    # def label_length(self):
    #     return self.info['label_length']

    def is_same_type(self, ann1, ann2):
      names1 = ann1['names']
      names2 = ann2['names']
      if len(set(names1).intersection(names2)) == 0:
        return False
      return True

    def is_same_type_relation(self, rel1, rel2):
        if rel1['predicate'].strip().lower() != rel2['predicate'].strip().lower():
            return False
        sbj1 = self.Anns[rel1['subject_id']]
        obj1 = self.Anns[rel1['object_id']]
        sbj2 = self.Anns[rel2['subject_id']]
        obj2 = self.Anns[rel2['object_id']]
        if self.is_same_type(sbj1, sbj2) and self.is_same_type(obj1, obj2):
            return True
        return False

    def gen_language_clue_str(self, ann):
        str = '[' + '|'.join(ann['names']) + ']'
        if len(ann['attributes']) > 0:
            str = '{' + '|'.join(ann['attributes']) + '}' + str
        if len(ann['relations']) > 0:
            str += '{'
            for rel in ann['relations']:
                obj_name = random.choice(self.Anns[rel['object_id']]['names'])
                rel_str = rel['predicate'] + ' ' + obj_name
                str += rel_str + '|'
            str += '}'
        return str

    def gen_phrase(self, ref_ann_id, ann_ids):
        def att_name_phrase(att, name):
            att_words = att.split()
            ph = att + ' ' + name
            if len(att_words) > 1:
                if att_words[0] in ['in', 'on', 'for', 'of', 'with', 'made', 'to', 'not', 'turned', 'off', 'from'] or\
                        (att_words[0][-3:] == 'ing' and att_words[0] not in ['living', 'king', 'ping', 'ceiling']):
                    ph = name + ' ' + att
            return ph

        ref_ann = self.Anns[ref_ann_id]
        if not ref_ann['names']:
            ref_ann['names'] = ['bad_name']
        anns = [self.Anns[ann_id] for ann_id in ann_ids]
        unique_type = True
        st_anns = []
        st_atts = []
        st_rels = []
        phrase_structure = {'type': '', 'name': '', 'attributes': [], 'relations': []}
        for ann in anns:
            if ann['ann_id'] == ref_ann['ann_id']:
                continue
            if self.is_same_type(ref_ann, ann):
                unique_type = False
                st_anns.append(ann)
                st_atts += ann['attributes']
                st_rels += ann['relations']

        if unique_type:
            self.phrase_type_stat['name'] += 1
            name = random.choice(ref_ann['names'])
            phrase = name
            phrase_structure['type'] = 'name'
            phrase_structure['name'] = name
            if len(ref_ann['attributes']) > 0:
                att = random.choice(ref_ann['attributes'])
                phrase_structure['attributes'].append(att)
                phrase = att_name_phrase(att, phrase)
            elif len(ref_ann['relations']) > 0:
                rels = [r for r in ref_ann['relations'] if len(self.Anns[r['object_id']]['names']) > 0]
                if len(rels) > 0:
                    rel = random.choice(rels)
                    phrase_structure['relations'].append(rel)
                    obj_names = self.Anns[rel['object_id']]['names']
                    obj_name = random.choice(obj_names)
                    phrase += ' ' + rel['predicate'] + ' ' + obj_name
            return phrase, phrase_structure

        # look for unique attribute
        for att in ref_ann['attributes']:
            if att not in st_atts:
                name = random.choice(ref_ann['names'])
                phrase = att_name_phrase(att, name)
                self.phrase_type_stat['attribute'] += 1
                phrase_structure['type'] = 'attribute'
                phrase_structure['name'] = name
                phrase_structure['attributes'].append(att)
                return phrase, phrase_structure

        # look for unique relation
        for ref_rel in ref_ann['relations']:
            for rel in st_rels:
                if not self.is_same_type_relation(ref_rel, rel):
                    name = random.choice(ref_ann['names'])
                    if len(self.Anns[ref_rel['object_id']]['names']) > 0:
                        obj_name = random.choice(self.Anns[ref_rel['object_id']]['names'])
                        phrase = name + ' ' + ref_rel['predicate'] + ' ' + obj_name
                        self.phrase_type_stat['relation'] += 1
                        phrase_structure['type'] = 'relation'
                        phrase_structure['name'] = name
                        phrase_structure['relations'].append(rel)
                        return phrase, phrase_structure

        # most verbose phrase
        phrase_str = random.choice(ref_ann['names'])
        phrase_structure['type'] = 'verbose'
        phrase_structure['name'] = phrase_str
        phrase_structure['attributes'] = ref_ann['attributes']
        phrase_structure['relations'] = ref_ann['relations']
        for att in ref_ann['attributes']:
            phrase_str = att_name_phrase(att, phrase_str)
        added_rel = []
        for ref_rel in ref_ann['relations']:
            if len(self.Anns[ref_rel['object_id']]['names']) > 0:
                obj_name = random.choice(self.Anns[ref_rel['object_id']]['names'])
                rel_str = ref_rel['predicate'] + ' ' + obj_name
                if rel_str not in added_rel:
                    phrase_str += ' ' + rel_str
                    added_rel.append(rel_str)
        self.phrase_type_stat['verbose'] += 1
        return phrase_str, phrase_structure

    def encode_labels(self, sent_str_list):
        return self.word_embed.encode_labels(sent_str_list, self.phrase_length)

    def decode_labels(self, labels):
        return self.word_embed.decode_labels(labels)

    def filter_objects(self, img_info, objects, relations):
        img_w = img_info['width']
        img_h = img_info['height']
        size_thresh = img_w * img_h * self.obj_size_thresh

        f_objs = []
        f_obj_ids = []
        random.shuffle(objects)
        for obj in objects:
            w = obj['w']
            h = obj['h']
            if w * h < size_thresh:
                continue

            valid_name = False
            for name in obj['names']:
                # if self.name_count.get(name, 0) >= self.name_count_thresh:
                if name in self.ix_to_name:
                    valid_name = True
                    break
            if not valid_name:
                continue

            box = [obj['x'], obj['y'], obj['w'], obj['h']]
            duplicate = False
            for f_obj in f_objs:
                f_box = [f_obj['x'], f_obj['y'], f_obj['w'], f_obj['h']]
                iou = iou_box(box, f_box)
                if iou > self.iou_dt_thresh:
                    duplicate = True
                    break
                elif iou > self.iou_st_thresh:
                    if self.is_same_type(obj, f_obj):
                        duplicate = True
                        break

            if not duplicate:
                f_objs.append(obj)
                f_obj_ids.append(obj['object_id'])

        f_relations = []
        for rel in relations:
            s_id = rel['subject_id']
            o_id = rel['object_id']
            if s_id in f_obj_ids and o_id in f_obj_ids:
                f_relations.append(rel)
        return f_objs, f_relations
