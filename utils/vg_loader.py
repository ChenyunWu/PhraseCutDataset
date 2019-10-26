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
import json
import random

from .iou import iou_box


class VGLoader(object):

    def __init__(self, split=None, word_embed=None, phrase_length=10,
                 cat_count_thresh=21, att_count_thresh=21, rel_count_thresh=21,
                 obj_filter=False, obj_size_thresh=0.005, iou_st_thresh=0.5, iou_dt_thresh=0.8,
                 nar_count_file='data/refvg/amt_result/name_att_rel_count_amt.json',
                 data_split_file='data/refvg/image_data_split3000.json'):

        self.word_embed = word_embed
        self.phrase_length = phrase_length
        # used in filtering objects (if obj_filter=True)
        self.obj_size_thresh = obj_size_thresh
        self.iou_st_thresh = iou_st_thresh
        self.iou_dt_thresh = iou_dt_thresh

        # load cat/att/rel count data
        print('Loader loading nar_count: %s' % nar_count_file)
        with open(nar_count_file, 'r') as f:
            count_info = json.load(f)
            self.cat_count_list = count_info['cat']  # list of (cat_name, count), count from high to low
            self.att_count_list = count_info['att']
            self.rel_count_list = count_info['rel']

        # prepare category
        self.label_to_cat = ['[INV]'] + [k for (k, c) in self.cat_count_list if c >= cat_count_thresh] + ['[UNK]']
        self.cat_to_label = {cat: l for l, cat in enumerate(self.label_to_cat)}
        print('Number of categories: %d / %d, frequency thresh: %d (excluding [INV] [UNK])'
              % (len(self.label_to_cat) - 2, len(self.cat_count_list), cat_count_thresh))

        # prepare attributes
        self.att_to_count = {k: c for (k, c) in self.att_count_list}
        self.att_to_count['[INV]'] = 0
        self.att_to_count['[UNK]'] = 0
        self.label_to_att = ['[INV]'] + [k for (k, c) in self.att_count_list if c >= att_count_thresh] + ['[UNK]']
        self.att_to_label = {att: l for l, att in enumerate(self.label_to_att)}
        print('Number of attributes: %d / %d, frequency thresh: %d (excluding [INV] [UNK])'
              % (len(self.label_to_att) - 2, len(self.att_count_list), att_count_thresh))

        # prepare relationships
        self.label_to_rel = ['[INV]'] + [k for (k, c) in self.rel_count_list if c >= rel_count_thresh] + ['[UNK]']
        self.rel_to_label = {rel: l for l, rel in enumerate(self.label_to_rel)}
        print('Number of relationships: %d / %d, frequency thresh: %d (excluding [INV] [UNK])'
              % (len(self.label_to_rel) - 2, len(self.rel_count_list), rel_count_thresh))

        # load the image split data
        print('Loader loading data_split: %s' % data_split_file)
        with open(data_split_file, 'r') as f:
            imgs_info = json.load(f)
            self.info_dict = {img['image_id']: img for img in imgs_info}

        # load scene graph data
        if not split:
            split = 'train_val_test_miniv'
        print('split: %s' % split)
        ss = split.split('_')
        images = []
        for s in ss:
            print('Loader loading scene_graphs_pp_%s.json' % s)
            with open('data/refvg/scene_graphs_pp_%s.json' % s, 'r') as f:
                images += json.load(f)
                
        # organize loaded img/obj data
        self.relations = []
        self.objects = dict()
        self.images = dict()
        # obj_ids = []  # deprecated. Used when we cache all object features into h5 files
        for img in images:
            img['file_name'] = '%d.jpg' % img['image_id']
            img_info = self.info_dict[img['image_id']]
            img['width'] = img_info['width']
            img['height'] = img_info['height']
            img['split'] = img_info['split']
            self.images[img['image_id']] = img
            # obj_ids += [obj['object_id'] for obj in img['objects']]

            # filter objects
            if obj_filter:
                f_objects, f_relations = self.filter_objects(img_info, img['objects'], img['relationships'])
                img['objects'] = f_objects
                img['relationships'] = f_relations

            img['obj_ids'] = [obj['object_id'] for obj in img['objects']]
            self.relations += img['relationships']
            for obj in img['objects']:
                attributes = []
                if 'attributes' in obj:
                    attributes = obj['attributes']
                self.objects[obj['object_id']] = {'obj_id': obj['object_id'], 'image_id': img['image_id'],
                                                  'box': [obj['x'], obj['y'], obj['w'], obj['h']],
                                                  'names': obj['names'],
                                                  'attributes': attributes,
                                                  # 'synsets': obj['synsets'],
                                                  'relations': []}

            for rel in img['relationships']:
                self.objects[rel['subject_id']]['relations'].append(rel)

        # obj_ids.sort()
        # for h_id, obj_id in enumerate(obj_ids):
        #     if obj_id in self.objects:
        #         self.objects[obj_id]['h5_id'] = h_id

        print('we have %s images.' % len(self.images))
        print('we have %s (filtered) objects, %.1f per image.'
              % (len(self.objects), len(self.objects) * 1.0 / len(self.images)))
        print('we have %s relations, %.1f per image.'
              % (len(self.relations), len(self.relations) * 1.0 / len(self.images)))

    @property
    def vocab_size(self):
        return len(self.word_embed.ix_to_word)

    @staticmethod
    def is_same_category(obj1, obj2):
        names1 = obj1['names']
        names2 = obj2['names']
        if len(set(names1).intersection(names2)) == 0:
            return False
        return True

    def is_same_relation(self, rel1, rel2):
        if rel1['predicate'].strip().lower() != rel2['predicate'].strip().lower():
            return False
        sbj1 = self.objects[rel1['subject_id']]
        obj1 = self.objects[rel1['object_id']]
        sbj2 = self.objects[rel2['subject_id']]
        obj2 = self.objects[rel2['object_id']]
        if self.is_same_category(sbj1, sbj2) and self.is_same_category(obj1, obj2):
            return True
        return False

    def encode_labels(self, sent_str_list):
        return self.word_embed.encode_sentences_to_labels(sent_str_list, self.phrase_length)

    def decode_labels(self, labels):
        return self.word_embed.decode_labels_to_sentences(labels)

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
                # if self.name_count.get(name, 0) >= self.cat_count_thresh:
                if name in self.label_to_cat[1:-1]:
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
                    if self.is_same_category(obj, f_obj):
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

    def phrase_struct_to_str(self, obj):
        string = '[' + '|'.join(obj['names']) + ']'
        if len(obj['attributes']) > 0:
            string = '{' + '|'.join(obj['attributes']) + '}' + string
        if len(obj['relations']) > 0:
            string += '{'
            for rel in obj['relations']:
                obj_name = random.choice(self.objects[rel['object_id']]['names'])
                rel_str = rel['predicate'] + ' ' + obj_name
                string += rel_str + '|'
            string += '}'
        return string
