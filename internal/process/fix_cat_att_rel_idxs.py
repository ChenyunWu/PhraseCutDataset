import json


# renamed 'name_att_rel_count_amt.json' to 'name_att_rel_count_amt_dict.json'
with open('data/refvg/amt_result/name_att_rel_count_amt_dict.json', 'r') as f:
    count_info = json.load(f)

cat_count_dict = count_info['name']
att_count_dict = count_info['att']
rel_count_dict = count_info['rel']

cat_count_list = [(k, v) for (k, v) in sorted(cat_count_dict.items(), key=lambda kv: -kv[1])]
att_count_list = [(k, v) for (k, v) in sorted(att_count_dict.items(), key=lambda kv: -kv[1])]
rel_count_list = [(k, v) for (k, v) in sorted(rel_count_dict.items(), key=lambda kv: -kv[1])]
info = {'cat': cat_count_list, 'att': att_count_list, 'rel': rel_count_list}

with open('data/refvg/amt_result/name_att_rel_count_amt.json', 'w') as f:
    json.dump(info, f)


