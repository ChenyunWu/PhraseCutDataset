import json
import random


def slim_ref():
    """
    Format of refer_filtered_instance_refine_xxx.json:
    List of dicts ('task_id', 'image_id', 'ann_ids', 'Polygons', 'instance_boxes', 'phrase', 'phrase_structure',
                    'iou', 'iob', 'iop', 'turk_id')
    - task_id: unique id for each phrase-region pair (constructed from image_id and ann_ids)
    - image_id: image id from Visual Genome
    - ann_ids: all object ids (in Visual Genome) that match the phrase
    - instance_boxes: list of referred instance boxes (xywh format)
    - Polygons: list of "instance_polygons", same length as instance_boxes. "instance_polygons":
        list of "polygon"s for a single instance. "polygon": list of [x, y] points, representing one polygon.
    - phrase: the referring phrase as a string
    - phrase_structure: dict of ('name', 'attributes', 'relations', 'relation_descriptions', 'type')
            - relations: list of dict, same as relations from VG: {'subject_id': ann_id (in VG) of the target,
                should be the same as the first one in ann_ids; 'predicate': string for the relation predicate;
                'object_id': ann_id (in VG) of the supporting object; 'relationship_id': from VG;
                'synsets': of the predicate, from VG}
            - relation_descriptions: list of relation_description. each relation_description is a list of two elements:
                string for the predicate, string (category name) for the supporting object
            - type: name (name is unique), attribute (att+name is unique), relation (name+relation is unique),
            verbose (not unique)
    - iou / iop / turk_id: used only for filtering the data

    After slim:
    List of dicts ('task_id', 'image_id', 'ann_ids', 'Polygons', 'instance_boxes', 'phrase', 'phrase_structure')
    - phrase_structure: dict of ('name', 'attributes', 'relation_ids', 'relation_descriptions', 'type')
    """
    # for split in ['miniv', 'val', 'test', 'train']:
    fpath = 'data/refvg/amt_result/refer_filtered_instance_refine.json'
    ref = json.load(open(fpath))
    for task in ref:
        for k in ['iou', 'iop', 'turk_id']:
            del task[k]
        ps = task['phrase_structure']
        rids = [r['relationship_id'] for r in ps['relations']]
        ps['relation_ids'] = rids
        del ps['relations']

    new_path = 'data/refvg/amt_result/refer_filtered_instance_refine_slim.json'
    with open(new_path, 'w') as f:
        json.dump(ref, f)
    print('%s saved.' % new_path)


def slim_skip():
    """
    Format of skip_vxx.json:
    List of dicts ('task_id', 'image_id', 'ann_ids', 'skip_reason', 'phrase', 'phrase_structure', 'turk_id', 'boxes')
    - phrase_structure: dict of ('name', 'attributes', 'relations', 'type')
            - relations: list of dict, same as relations from VG: {'subject_id': ann_id (in VG) of the target,
                should be the same as the first one in ann_ids; 'predicate': string for the relation predicate;
                'object_id': ann_id (in VG) of the supporting object; 'relationship_id': from VG;
                'synsets': of the predicate, from VG}
              - type: name (name is unique), attribute (att+name is unique), relation (name+relation is unique),
            verbose (not unique)

    After slim:
    remove 'turk_id', replace 'relations' with 'relation_ids' in phrase_structure
    List of dicts ('task_id', 'image_id', 'ann_ids', 'skip_reason', 'phrase', 'phrase_structure', 'boxes')
    - phrase_structure: dict of ('name', 'attributes', 'relation_ids', 'type')
    """
    skipped = list()
    for v in ['v1.01', 'v1.2', 'v2']:
        fpath = 'data/refvg/amt_result/skip_%s.json' % v
        skipped += json.load(open(fpath))

    for task in skipped:
        del task['turk_id']
        if 'phrase_structure' in task:
            ps = task['phrase_structure']
            rids = [r['relationship_id'] for r in ps['relations']]
            ps['relation_ids'] = rids
            del ps['relations']

    print(len(skipped))
    new_path = 'data/refvg/amt_result/skip_slim.json'
    with open(new_path, 'w') as f:
        json.dump(skipped, f)
    print('%s saved.' % new_path)


def update_split():
    """
    Update image_data_split3000.json and scene_graphs_train.json:
        - remove train images that are not annotated (include only images in refer_train and skip_train)
        - larger miniv with 100 images from val
    """
    # remove train images that are not annotated
    fpath = 'data/VGPhraseCut_v0/refer_train.json'
    ref = json.load(open(fpath))
    train_ids = set()
    for task in ref:
        train_ids.add(task['image_id'])

    skip_f = 'data/VGPhraseCut_v0/skip.json'
    skip = json.load(open(skip_f))
    skip_ids = set()
    for task in skip:
        skip_ids.add(task['image_id'])

    keep_ids = train_ids.union(skip_ids)

    with open('data/refvg/image_data_split3000.json') as f:
        info = json.load(f)
    to_remove = list()
    for img in info:
        if img['split'] == 'train' and img['image_id'] not in keep_ids:
            to_remove.append(img['image_id'])

    new_info = [img for img in info if img['image_id'] not in to_remove]
    print(len(info), len(new_info))

    val_ids = [img['image_id'] for img in new_info if img['split'] in ['val', 'miniv']]
    miniv_ids = random.sample(val_ids, 100)
    for img in new_info:
        if img['image_id'] in val_ids:
            img['split'] = 'val'
        if img['image_id'] in miniv_ids:
            img['split'] = 'miniv'

    with open('data/refvg/image_data_split3000_100_slim.json', 'w') as f:
        json.dump(new_info, f)
    print('data/refvg/image_data_split3000_100_slim.json saved.')


def split_scene_graph():
    raw_file = '../data/VisualGenome1.2/scene_graphs.json'
    info_file = 'data/refvg/image_data_split3000_100_slim.json'
    splits = ['miniv', 'test', 'val', 'train']

    info = json.load(open(info_file))
    id_to_split = {img['image_id']: img['split'] for img in info}
    print(len(id_to_split))

    rel = json.load(open(raw_file))
    rels = dict()
    for s in splits:
        rels[s] = list()
    for r in rel:
        img_id = r['image_id']
        if img_id not in id_to_split:
            continue
        s = id_to_split[img_id]
        rels[s].append(r)

    for k, v in rels.items():
        out_fpath = 'data/VGPhraseCut_v0/scene_graphs_%s.json' % k
        with open(out_fpath, 'w') as f:
            json.dump(v, f)
        print(k, len(v), 'saved to %s' % out_fpath)


def split_ref():
    raw_file = 'data/refvg/amt_result/refer_filtered_instance_refine_slim.json'
    info_file = 'data/refvg/image_data_split3000_100_slim.json'
    splits = ['miniv', 'test', 'val', 'train']

    info = json.load(open(info_file))
    id_to_split = {img['image_id']: img['split'] for img in info}
    print(len(id_to_split))

    rel = json.load(open(raw_file))
    rels = dict()
    for s in splits:
        rels[s] = list()
    for r in rel:
        img_id = r['image_id']
        s = id_to_split[img_id]
        rels[s].append(r)

    for k, v in rels.items():
        out_fpath = 'data/refvg/amt_result/refer_filtered_instance_refine_slim_%s.json' % k
        with open(out_fpath, 'w') as f:
            json.dump(v, f)
        print(k, len(v), 'saved to %s' % out_fpath)


def test_blind():
    """
    Format of refer_filtered_instance_refine_slim_xxx.json:
    List of dicts ('task_id', 'image_id', 'ann_ids', 'Polygons', 'instance_boxes', 'phrase', 'phrase_structure')
    - phrase_structure: dict of ('name', 'attributes', 'relation_ids', 'relation_descriptions', 'type')

    After blind:
    remove gt annotations: Polygons, instance_boxes
    List of dicts ('task_id', 'image_id', 'ann_ids', 'phrase', 'phrase_structure')
    """

    for split in ['test']:
        fpath = 'data/VGPhraseCut_v0/refer_%s.json' % split
        ref = json.load(open(fpath))
        for task in ref:
            for k in ['Polygons', 'instance_boxes']:
                del task[k]

        new_path = 'data/VGPhraseCut_v0/refer_%s_blind.json' % split
        with open(new_path, 'w') as f:
            json.dump(ref, f)
        print('%s saved.' % new_path)


if __name__ == '__main__':
    # slim_ref()
    # update_split()
    # split_scene_graph()
    split_ref()
