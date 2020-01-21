import json
import pymysql
import sshtunnel

import matplotlib.pyplot as plt
plt.switch_backend('agg')

from ..utils.iou import iou_boxes_polygons


# Used to upload data for Turk annotation.
def upload_to_db(json_f='data/refvg/turker_prep/phrases_train1000.json', commit=False, start_count=0):
    # input json_f is generated from 'pyutils/visual_genome_python_driver/preprocess.ipynb'
    with open(json_f, 'r') as f:
        dataset = json.load(f)

    with sshtunnel.SSHTunnelForwarder(('ec2-13-56-248-100.us-west-1.compute.amazonaws.com', 22),
                                      ssh_username='ubuntu',
                                      ssh_pkey='../CollectDataVG/bui.pem',
                                      remote_bind_address=('localhost', 3306),
                                      local_bind_address=('localhost', 3306)) as tunnel:
        print('ssh')
        conn = pymysql.connect(host='localhost', port=3306, user='root', passwd='ka1st', db='PhraseCut_RefVG',
                               charset='utf8')
        print(conn)
        try:
            with conn.cursor() as cur:
                for i in range(start_count, len(dataset)):
                    data = dataset[i]
                    # for i, data in enumerate(dataset):
                    # if i <= 196000: continue
                    # if i < 196200:
                    #     cur.execute('SELECT COUNT(1) as n FROM %s WHERE hits > 0 and img_id=%d' %
                    #                 ('ChenYunVisualGenomeSourceTable', data['image_id']))
                    #     row = cur.fetchone()
                    #     if row[0] != 0:
                    #         print('EXISTING IMG: ', row)
                    if '\\' in data['phrase']:
                        print('skipped: %d task_id=%s, phrase=%s' % (i, data['task_id'], data['phrase']))
                        continue
                    cur.execute(
                        'INSERT INTO %s (task_id, image_url, phrase) VALUES (\'%s\', \'%s\', \'%s\')'
                        % ('requests', data['task_id'], data['image_url'], data['phrase'].replace("'", "''")))
                    # print(data['task_id'])

                    if i % 100 == 0:
                        print('%d / %d' % (i, len(dataset)))

                    if i % 1000 == 0 and commit:
                        conn.commit()
                        print('%d: commit to db' % i)
            if commit:
                conn.commit()
        except Exception as e:
            print('ERR', e)
        conn.close()
    return


def get_data_from_db(mode, out_path=None, start_id=0, end_id=-1, rule_sql_str=None):
    """
    Get the raw mask / skip data from AMT through MySQL
    Note: only for training split. Raw MySQL data for test/val is lost.
    :param mode: 'mask' or 'skip'
    :param out_path: output json file path. If None, only return the data without saving to file
    :param start_id: key_id range in MySQL mask / skip table
    :param end_id: key_id range in MySQL mask / skip table
    :param rule_sql_str: additional MySQL rule to filter the entries. e.g. 'where turk_id = "chenyun"'
    :return:
        'mask' mode: list of {'mask_key_id', 'task_id', 'phrase', 'polygons_str', 'turk_id', 'datetime'}
        'skip' mode: list of {'skip_key_id', 'task_id', 'phrase', 'reason', 'turk_id', 'datetime'}

    """
    db = {}

    # with sshtunnel.SSHTunnelForwarder(('ec2-52-53-188-154.us-west-1.compute.amazonaws.com', 22),
    #                                   ssh_username='ubuntu',
    #                                   ssh_pkey='../CollectDataVG/bui.pem',
    #                                   remote_bind_address=('localhost', 3306),
    #                                   local_bind_address=('localhost', 3306)) as tunnel:
    #     print('ssh')
    conn = pymysql.connect(host='localhost', port=3306, user='root', passwd='Wcy!8615', db='PhraseCut_RefVG',
                           charset='utf8')
    print(conn)
    try:
        with conn.cursor() as cur:
            sql = 'SELECT * FROM '
            if 'mask' in mode:
                sql += 'mask'
            else:  # 'skip' in mode:
                sql += 'skip'
            if rule_sql_str:
                sql += ' ' + rule_sql_str

            cur.execute(sql)
            rows = cur.fetchall()
            for i, row in enumerate(rows):
                # print(row)
                if row[3] == 'null':
                    print('WARNING:', row)
                    continue
                if row[0] < start_id or row[0] > end_id > 0:
                    continue
                if 'mask' in mode:
                    entry = {'mask_key_id': row[0], 'task_id': row[1], 'phrase': row[2], 'polygons_str': row[3],
                             'turk_id': row[4], 'datetime': str(row[5])}
                else:
                    entry = {'skip_key_id': row[0], 'task_id': row[1], 'phrase': row[2], 'reason': row[3],
                             'turk_id': row[4], 'datetime': str(row[5])}
                print('%d / %d' % (i, len(rows)))
                if row[1] not in db:
                    db[row[1]] = [entry]
                else:
                    db[row[1]].append(entry)
    except Exception as e:
        print('ERR', e)

    conn.close()

    if out_path:
        with open(out_path, 'w') as f:
            json.dump(db, f)
    return db


def gather_collected_data(mask_json, skip_json, split='train'):
    """
    Gather the AMT data (json files from get_data_from_db(...)) of a split, save to 'refer?.json', 'skip?.json'
    'refer?.json': list of {'task_id', 'image_id', 'ann_ids', 'phrase', 'phrase_structure',
                            'Polygons'(only for train) or 'polygons', 'turk_id', 'iou', 'iob', 'iop'}
    'skip?.json': list of {'task_id', 'image_id', 'ann_ids', 'phrase', 'phrase_structure',
                           'turk_id', 'skip_reason', 'boxes'}
    Note: code only to run on split='tain'. we have the output file for test/val
    'polygons': [[[x1, y1],[x2,y2],...]]
    'Polygons': [[[[x1, y1],[x2,y2],...], [(another polygon)], ... (one instance)]]
    :param mask_json: mask json file path from get_data_from_db(...)
    :param skip_json: skip json file path from get_data_from_db(...)
    :param split: 'test' 'val' 'train'
    :return: None
    """
    # load amt_collect mask and skip
    with open(mask_json) as f:
        mask_amt = json.load(f)
    with open(skip_json) as f:
        skip_amt = json.load(f)
    print('amt_collect mask + skip loaded.')

    # load VG img_info data
    with open('data/refvg/image_data_split1000.json', 'r') as f:
        img_info = json.load(f)
        info_dict = {}
        for img in img_info:
            # if img['split'] == split:
                info_dict[img['image_id']] = [img['width'], img['height'], img['url']]
    print('image_data_split1000.json loaded')

    # load VG object data
    with open('data/refvg/objects.json', 'r') as f:
        imgs = json.load(f)
        object_dict = {}
        for img in imgs:
            for obj in img['objects']:
                object_dict[obj['object_id']] = obj
    print('objects.json loaded')

    # load generated phrases
    with open('data/refvg/turker_prep/phrases_%s1000.json' % split, 'r') as f:
        gen_data = json.load(f)
    print('phrases_%s1000.json loaded' % split)

    for entry in gen_data:
        entry['boxes'] = []
        entry['width'], entry['height'], entry['image_url'] = info_dict[entry['image_id']]
        for obj_id in entry['ann_ids']:
            # x, y, w, h
            obj = object_dict[obj_id]
            x = obj['x']
            y = obj['y']
            w = obj['w']
            h = obj['h']
            box = [x, y, w, h]
            entry['boxes'].append(box)

    refer_data = []
    skip_data = []

    for i, entry in enumerate(gen_data):
        # entry: task_id,image_id,ann_ids,phrase,phrase_structure, | phrase_mode + width,height,url,boxes
        target_dataset = None
        masks = mask_amt.get(entry['task_id'], None)
        if masks:
            masks.sort(key=lambda m: -len(m['polygons_str']))
            mask = masks[0]
            entry['turk_id'] = mask['turk_id']
            entry['Polygons'] = []
            Polygons = json.loads(mask['polygons_str'])
            instances = dict()
            for Polygon in Polygons:
                if len(Polygon['points']) < 3:
                    continue
                points = []
                for pi, p_str in enumerate(Polygon['points']):
                    xy_str = p_str.split(',')
                    x = int(xy_str[0]) / 600.0 * entry['width']
                    y = int(xy_str[1]) / 600.0 * entry['width']
                    points.append([x, y])
                if Polygon['name'] in instances:
                    instances[Polygon['name']].append(points)
                else:
                    instances[Polygon['name']] = [points]
            entry['Polygons'] = instances.values()
            entry['iou'], entry['iob'], entry['iop'] = iou_boxes_polygons(entry['boxes'],
                                                                          [p for ins in entry['Polygons'] for p in ins],
                                                                          entry['width'], entry['height'],
                                                                          True, True)
            entry.pop('boxes', None)
            target_dataset = refer_data

        else:
            skips = skip_amt.get(entry['task_id'], None)
            if skips:
                skip = skips[0]
                entry['turk_id'] = skip['turk_id']
                entry['skip_reason'] = skip['reason']
                target_dataset = skip_data

        if target_dataset is None:
            pass
            # print('WARNING: %s not annotated.' % entry['task_id'])
        else:
            entry.pop('phrase_mode', None)
            entry.pop('width', None)
            entry.pop('height', None)
            entry.pop('image_url', None)
            target_dataset.append(entry)

        if i % 100 == 0:
            print('refer_data: %d; skip_data: %d' % (len(refer_data), len(skip_data)))

    print('TOTAL: refer_data: %d; skip_data: %d' % (len(refer_data), len(skip_data)))

    with open('data/amt_collect/refer_%s.json' % split, 'w') as f:
        json.dump(refer_data, f)
    with open('data/amt_collect/skip_%s.json' % split, 'w') as f:
        json.dump(skip_data, f)
    print('saved to refer_%s.json and skip_%s.json' % (split, split))

    return


if __name__ == '__main__':
    # upload_to_db(commit=True, start_count=473001)
    # get_data_from_db('mask', 'data/amt_collect/train_mask.json')
    # get_data_from_db('skip', 'data/amt_collect/train_skip.json')
    gather_collected_data('data/amt_collect/train_mask.json', 'data/amt_collect/train_skip.json')


# # One-time use only
# def collected_tasks():
#     db = get_data_from_db('masks')
#     collected = {}
#     for task_id in db.keys():
#         img_id = int(task_id.split('__')[0])
#         if img_id not in collected:
#             collected[img_id] = [task_id]
#         else:
#             collected[img_id].append(task_id)
#     print(sorted(collected.keys()))
#     with open('data/refvg/collected_imgs_old.json', 'w') as f:
#         json.dump(collected, f)
#     return


# # One-time use only
# def get_coef():
#     f = open('data/refvg/my_ann.csv')
#     csv_f = csv.reader(f)
#     data = []
#     first=True
#     for row in csv_f:
#         if first:
#             first = False
#             continue
#         row = [float(e) for e in row]
#         data.append(row)
#     data = np.array(data)
#
#     X = data[:, 1:3]  # we only take the first two features.
#     Y = (data[:, 3] >= 1).astype(int)
#
#     h = .02  # step size in the mesh
#
#     logreg = linear_model.LogisticRegression(penalty='l1')
#
#     # we create an instance of Neighbours Classifier and fit the data.
#     logreg.fit(X, Y)
#
#     print(logreg)
#
#     coef = logreg.coef_[0]
#     print (coef)
#
#     # Plot the decision boundary. For that, we will assign a color to each
#     # point in the mesh [x_min, x_max]x[y_min, y_max].
#     x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
#     y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
#     xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
#     Z = logreg.predict(np.c_[xx.ravel(), yy.ravel()])
#
#     # Put the result into a color plot
#     Z = Z.reshape(xx.shape)
#     plt.figure(1, figsize=(4, 3))
#     plt.pcolormesh(xx, yy, Z)
#
#     # Plot also the training points
#     plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolors='k')
#     plt.xlabel('Sepal length')
#     plt.ylabel('Sepal width')
#
#     plt.xlim(xx.min(), xx.max())
#     plt.ylim(yy.min(), yy.max())
#     plt.xticks(())
#     plt.yticks(())
#
#     plt.show()


# # Obsolete
# def phrases_json_to_turk_csv(json_path, start_id=0):
#     # task_id,img_id,image_url,transcripts,hits,id
#     with open(json_path, 'r') as f:
#         dataset = json.load(f)
#
#     csv_path = json_path[:-len('.json')] + '.csv'
#
#     out_f = open(csv_path, 'w')
#
#     db_id = start_id
#     for data in dataset:
#         out_f.write('%s,%d,%s,[\"%s\"],0,%d\n'
#                     % (data['task_id'], data['image_id'], data['image_url'], data['phrase'], db_id))
#         db_id += 1
#     out_f.close()


# # One-time use
# def gather_skip_val_old():
#     with open('data/refvg/collected_imgs_old.json', 'r') as f:
#         old_data = json.load(f)
#         old_img_ids = [int(i) for i in old_data.keys()]
#
#     skip_data = {}
#
#     with sshtunnel.SSHTunnelForwarder(('ec2-52-53-126-203.us-west-1.compute.amazonaws.com', 22),
#                                       ssh_username='ubuntu',
#                                       ssh_pkey='../CollectDataVG/nle.pem',
#                                       remote_bind_address=('localhost', 3306),
#                                       local_bind_address=('localhost', 3306)) as tunnel:
#         print('ssh')
#         conn = pymysql.connect(host='localhost', port=3306, user='root', passwd='ka1st', db='amturk_data',
#                                charset='utf8')
#         print(conn)
#         try:
#             with conn.cursor() as cur:
#                 for i, img_id in enumerate(old_img_ids):
#                     # entry: task_id,image_id,ann_ids,phrase,phrase_structure, | phrase_mode + width,height,url,boxes
#                     cur.execute('SELECT * FROM VG_skipped WHERE task_id LIKE "{}\_\_%"'.format(img_id))
#                     rows = cur.fetchall()
#                     print(img_id, len(rows), len(skip_data))
#                     for row in rows:
#                         task_id = row[1]
#                         if task_id in skip_data.keys():
#                             print(task_id, 'already skipped')
#                             continue
#                         img_id = int(task_id.split('__')[0])
#                         ann_ids = [int(i) for i in task_id.split('__')[1].split('_')]
#
#                         entry = {'task_id': row[1], 'image_id': img_id, 'ann_ids': ann_ids, 'turk_id': row[2],
#                                  'skip_reason': row[3]}
#
#                         cur.execute('SELECT * FROM ChenYunVisualGenomeSourceTable WHERE task_id = "%s"'% task_id)
#                         rows = cur.fetchall()
#                         row = rows[0]
#                         entry['phrase'] = row[3][2:-2]
#                         skip_data[task_id] = entry
#
#             conn.close()
#         except Exception as e:
#             conn.close()
#             print('ERR', e)
#
#     skip_data = skip_data.values()
#     with open('data/refvg/skip_val_old.json', 'w') as f:
#         json.dump(skip_data, f)
#     print('saved to skip_val_old.json')
#
#     skip_slim = []
#     s_slim_keys = ['image_id', 'phrase', 'skip_reason']
#     for entry in skip_data:
#         es = dict()
#         for k in s_slim_keys:
#             es[k] = entry[k]
#         skip_slim.append(es)
#     with open('data/refvg/skip_slim_val_old.json', 'w') as f:
#         json.dump(skip_slim, f)
#     return

# # One-time use only
# def merge_val_old():
#     for f_name in ['refer_slim', 'skip_slim', 'skip']:
#         with open('data/refvg/%s_val.json' % f_name) as f:
#             rsv = json.load(f)
#         with open('data/refvg/%s_val_old.json' % f_name) as f:
#             rsvo = json.load(f)
#         rsv += rsvo
#         with open('data/refvg/%s_val_all.json' % f_name, 'w') as f:
#             json.dump(rsv, f)
#         print(f_name)
#
#     with open('data/refvg/refer_val.json') as f:
#         rsv = json.load(f)
#     with open('data/refvg/refer_val_old.json') as f:
#         rsvo = json.load(f)
#
#     for d in rsvo:
#         d['phrase_structure'] = {}
#     rsv += rsvo
#
#     with open('data/refvg/refer_val_all.json', 'w') as f:
#         json.dump(rsv, f)
#

# def update_db(split_name, json_f):
#     with open(json_f, 'r') as f:
#         dataset = json.load(f)
#
#     with sshtunnel.SSHTunnelForwarder(('ec2-52-53-126-203.us-west-1.compute.amazonaws.com', 22),
#                                       ssh_username='ubuntu',
#                                       ssh_pkey='../CollectDataVG/nle.pem',
#                                       remote_bind_address=('localhost', 3306),
#                                       local_bind_address=('localhost', 3306)) as tunnel:
#         print('ssh')
#         conn = pymysql.connect(host='localhost', port=3306, user='root', passwd='ka1st', db='amturk_data',
#                                charset='utf8')
#         print(conn)
#         try:
#             with conn.cursor() as cur:
#                 for i, data in enumerate(dataset):
#
#                     cur.execute('UPDATE %s SET split="%s"  WHERE task_id="%s"' %
#                                 ('ChenYunVisualGenomeSourceTable', split_name, data['task_id']))
#                     if i % 50 == 0:
#                         print('%d / %d commited' % (i, len(dataset)))
#                         conn.commit()
#             conn.close()
#         except Exception as e:
#             conn.close()
#             print('ERR', e)
