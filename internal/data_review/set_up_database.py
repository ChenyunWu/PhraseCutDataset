import json
import pymysql
import sshtunnel
from utils.file_paths import refer_fpaths, img_info_fpath


# Used to upload data for Turk annotation.
def upload_refer_to_db(upload_info=False, splits='train', start_i=0):
    splits = splits.split('_')

    # with sshtunnel.SSHTunnelForwarder(('davinci.cs.umass.edu', 22),
    #                                   ssh_username='chenyun',
    #                                   ssh_pkey='~/.ssh/id_rsa',
    #                                   remote_bind_address=('localhost', 3306),
    #                                   local_bind_address=('localhost', 3306)) as tunnel:
    #     print('ssh')
    conn = pymysql.connect(host='localhost', port=3306, user='root', passwd='davinci', db='VGPhraseCut_v0',
                           charset='utf8')
    print(conn)

    try:
        with conn.cursor() as cur:
            if upload_info:
                with open(img_info_fpath) as f:
                    img_info = json.load(f)

                for img in img_info:
                    cur.execute("INSERT INTO img_info (image_id, width, height, url, split) "
                                "VALUES ('{image_id}', '{width}', '{height}', '{url}', '{split}')"
                                .format(**img))
                conn.commit()
                print('uploaded img_info to database.')

            added_tasks = set()
            for split in splits:
                with open(refer_fpaths[split], 'r') as f:
                    ref_data = json.load(f)

                print('start on %s - %d' % (split, start_i))
                for i, refer in enumerate(ref_data):
                    if i < start_i:
                        continue
                    if refer['task_id'] in added_tasks:
                        print('WARNING: duplicate task_id in dataset: %s' % refer['task_id'])
                        continue
                    added_tasks.add(refer['task_id'])
                    sql = "INSERT INTO refer (task_id, phrase, refer) VALUES ('%s', '%s', '%s')"\
                          % (refer['task_id'], refer['phrase'].replace("'", "\\'"),
                             json.dumps(refer).replace("'", "\\'").replace('\\"', '\\\\"'))
                    # print(sql)
                    cur.execute(sql)

                    if i % 100 == 0:
                        print('%d / %d' % (i, len(ref_data)))
                        conn.commit()
                conn.commit()
                print('uploaded %s to database.' % split)

    except Exception as e:
        print('ERR', e)
    conn.close()
    return


if __name__ == '__main__':
    upload_refer_to_db()
