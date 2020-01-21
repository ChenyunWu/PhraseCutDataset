import pymysql
import configparser
from flask import Flask, request, redirect, url_for, Response, render_template, jsonify, send_from_directory

app = Flask(__name__)
settings = configparser.ConfigParser()
settings_filepath = 'settings.ini'
settings.read(settings_filepath)


@app.route('/query/', methods=['POST'])
def query():
    phrase_q = request.form['phrase_q']
    print(phrase_q)
    try:
        conn = pymysql.connect(host='davinci.cs.umass.edu', port=3306, user='root', passwd='davinci',
                               db='PhraseCut_RefVG', charset='utf8')
        cur = conn.cursor()
        if phrase_q == '':
            cur.execute("SELECT * FROM requests_sub WHERE last_attempt <= (NOW()- INTERVAL 1 DAY)" + \
                        " AND submit_count=0 ORDER BY RAND() LIMIT 1")
            if cur.rowcount == 0:
                cur.execute("SELECT * FROM requests_sub WHERE submit_count = 0 ORDER BY RAND() LIMIT 1")
        else:
            cur.execute("SELECT * FROM refer WHERE phrase = %s", (phrase_q))
        row = cur.fetchone()

        print(row)
        # cur.execute("UPDATE requests SET last_attempt = CURRENT_TIMESTAMP WHERE phrase_q = '%s'" % row[1])
        cur.execute("UPDATE requests_sub SET last_attempt = CURRENT_TIMESTAMP WHERE phrase_q = '%s'" % row[1])
        conn.commit()
        conn.close()
        print("QUERY SUCCESS")
        return jsonify(
            task_id=row[1],
            image_url=row[2],
            phrase=row[3]
        )
    except Exception as e:
        conn.close()
        print(e)