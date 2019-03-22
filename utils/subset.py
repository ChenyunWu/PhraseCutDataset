subsets = ['all',
           'c20', 'c100', 'c500', 'c500+', 'c21-100', 'c101-500',
           'i_single', 'i_multi', 'i_many',
           'p_name', 'p_att', 'p_att+', 'p_rel', 'p_rel+', 'p_verbose',
           't_stuff', 't_obj',
           's_small', 's_mid', 's_large'
           ]


def get_subset(phrase, phrase_structure, gt_boxes, gt_relative_size):
    cond = dict()
    for key in subsets:
        cond[key] = False
    cond['all'] = True

    # # people
    # cond['people'] = False
    # for name in people_names:
    #     if name in phrase:
    #         cond['people'] = True
    #         break
    # cond['non_people'] = not cond['people']

    # top_k
    top_k = 501  # top_k starts from 0
    for ni, name in enumerate(vg500_names):
        if name in phrase:
            top_k = ni
            break
    if top_k < 20:
        cond['c20'] = True
    elif top_k < 100:
        cond['c21-100'] = True
    elif top_k < 500:
        cond['c101-500'] = True
    else:
        cond['c500+'] = True
    if top_k < 100:
        cond['c100'] = True
    if top_k < 500:
        cond['c500'] = True

    # phrase mode
    if phrase_structure:
        if phrase_structure['attributes']:
            cond['p_att'] = True
        if phrase_structure['relations']:
            cond['p_rel'] = True

        if phrase_structure['type'] == 'name':
            cond['p_name'] = True
        if phrase_structure['type'] == 'attribute':
            cond['p_att+'] = True
        if phrase_structure['type'] == 'relation':
            cond['p_rel+'] = True
        if phrase_structure['type'] == 'verbose':
            cond['p_verbose'] = True

    # instance count
    if len(gt_boxes) == 1:
        cond['i_single'] = True
    if len(gt_boxes) > 1:
        cond['i_multi'] = True
    if len(gt_boxes) >= 5:
        cond['i_many'] = True
        
    # gt size
    if gt_relative_size < 0.02:
        cond['s_small'] = True
    elif gt_relative_size > 0.2:
        cond['s_large'] = True
    else:
        cond['s_mid'] = True
    # stuff or not
    is_stuff = False
    for name in stuff_names:
        if name in phrase:
            is_stuff = True
            break
    if is_stuff:
        cond['t_stuff'] = True
    else:
        cond['t_obj'] = True
    return cond


coco79 = [u'bicycle', u'car', u'motorcycle', u'airplane', u'bus', u'train', u'truck', u'boat', u'traffic light',
          u'fire hydrant', u'stop sign', u'parking meter', u'bench', u'bird', u'cat', u'dog', u'horse', u'sheep',
          u'cow', u'elephant', u'bear', u'zebra', u'giraffe', u'backpack', u'umbrella', u'handbag', u'tie',
          u'suitcase', u'frisbee', u'skis', u'snowboard', u'sports ball', u'kite', u'baseball bat', u'baseball glove',
          u'skateboard', u'surfboard', u'tennis racket', u'bottle', u'wine glass', u'cup', u'fork', u'knife', u'spoon',
          u'bowl', u'banana', u'apple', u'sandwich', u'orange', u'broccoli', u'carrot', u'hot dog', u'pizza', u'donut',
          u'cake', u'chair', u'couch', u'potted plant', u'bed', u'dining table', u'toilet', u'tv', u'laptop', u'mouse',
          u'remote', u'keyboard', u'cell phone', u'microwave', u'oven', u'toaster', u'sink', u'refrigerator', u'book',
          u'clock', u'vase', u'scissors', u'teddy bear', u'hair drier', u'toothbrush']
people_names = ['person', 'people', 'man', 'men', 'woman', 'women', 'kid', 'kids', 'baby', 'babies', 'boy', 'boys',
                'girl', 'girls', 'child', 'children', 'lady', 'player', 'guy', 'guys']
stuff_names = ['water', 'waterdrops', 'sea', 'river', 'fog', 'ground', 'field', 'platform', 'rail', 'pavement', 'road',
               'gravel', 'mud', 'dirt', 'snow', 'sand', 'solid', 'hill', 'mountain', 'stone', 'rock', 'wood', 'sky',
               'cloud', 'vegetation', 'straw', 'moss', 'branch', 'leaf', 'leaves', 'bush', 'tree', 'grass',
               'forest', 'railing', 'net', 'cage', 'fence', 'building', 'roof', 'tent', 'bridge', 'skyscraper', 'house',
               'food', 'vegetable', 'salad', 'textile', 'banner', 'blanket', 'curtain', 'cloth',
               'napkin', 'towel', 'mat', 'rug', 'stairs', 'light', 'counter', 'mirror', 'cupboard', 'cabinet', 'shelf',
               'table', 'desk', 'door', 'window', 'floor', 'carpet', 'ceiling', 'wall', 'brick', 'metal', 'plastic',
               'paper', 'cardboard', 'street', 'snow', 'shadow', 'sidewalk', 'plant', 'wave', 'reflection', 'ocean',
               'beach']  # 'flower', 'fruit', 'pillow'
vg500_names = [u'man', u'person', u'window', u'shirt', u'tree', u'building', u'wall', u'sky', u'woman', u'sign',
               u'ground', u'grass', u'table', u'pole', u'head', u'light', u'car', u'water', u'hair', u'hand', u'people',
               u'leg', u'clouds', u'trees', u'plate', u'leaves', u'ear', u'pants', u'fence', u'door', u'chair', u'eye',
               u'hat', u'floor', u'train', u'road', u'jacket', u'street', u'snow', u'wheel', u'line', u'shadow', u'boy',
               u'nose',
               u'shoe', u'letter', u'cloud', u'clock', u'boat', u'tail', u'handle', u'sidewalk', u'field', u'girl',
               u'flower', u'leaf', u'horse', u'helmet', u'bus', u'shorts', u'bird', u'elephant', u'giraffe', u'plane',
               u'umbrella', u'dog', u'bag', u'arm', u'face', u'windows', u'zebra', u'glass', u'sheep', u'cow', u'bench',
               u'cat', u'food', u'bottle', u'tile', u'rock', u'kite', u'stripe', u'post', u'tire', u'number', u'truck',
               u'flowers', u'logo', u'surfboard', u'shoes', u'bear', u'roof', u'picture', u'cap', u'spot', u'bowl',
               u'glasses', u'motorcycle', u'jeans', u'skateboard', u'player', u'background', u'foot', u'box', u'bike',
               u'mirror', u'pizza', u'pillow', u'top', u'tracks', u'shelf', u'lights', u'legs', u'house', u'mouth',
               u'dirt',
               u'part', u'cup', u'plant', u'board', u'trunk', u'banana', u'counter', u'bush', u'ball', u'wave',
               u'lines',
               u'button', u'bed', u'lamp', u'sink', u'brick', u'beach', u'flag', u'writing', u'sand', u'coat', u'neck',
               u'vase', u'letters', u'paper', u'seat', u'glove', u'wing', u'child', u'vehicle', u'toilet',
               u'reflection',
               u'laptop', u'airplane', u'phone', u'book', u'sunglasses', u'branch', u'edge', u'cake', u'rocks', u'desk',
               u'tie', u'frisbee', u'animal', u'tower', u'hill', u'eyes', u'stripes', u'cabinet', u'mountain',
               u'headlight',
               u'container', u'frame', u'lady', u'wheels', u'ceiling', u'ocean', u'towel', u'racket', u'skier',
               u'keyboard',
               u'hands', u'design', u'windshield', u'back', u'pot', u'feet', u'basket', u'track', u'fork', u'bat',
               u'waves',
               u'fruit', u'orange', u'finger', u'guy', u'railing', u'engine', u'suit', u'broccoli', u'knife', u'couch',
               u'collar', u'cars', u'sock', u'apple', u'backpack', u'suitcase', u'knob', u'surfer', u'cheese',
               u'screen',
               u'donut', u'dress', u'buildings', u'blanket', u'paw', u'bananas', u'bicycle', u'bushes', u'van',
               u'lettering',
               u'tag', u'sticker', u'lid', u'bread', u'photo', u'skis', u'sweater', u'uniform', u'curtain', u'watch',
               u'tray', u'stand', u'stone', u'ears', u'men', u'wood', u'wire', u'sandwich', u'court', u'branches',
               u'room',
               u'bridge', u'traffic light', u'stem', u'white', u'mane', u'napkin', u'word', u'pavement', u'cone',
               u'faucet',
               u'fur', u'kid', u'carrot', u'camera', u'arrow', u'object', u'air', u'numbers', u'ski', u'hole',
               u'drawer',
               u'key', u'spoon', u'wrist', u'computer', u'platform', u'area', u'side', u'plants', u'meat', u'poles',
               u'cord',
               u'vest', u'strap', u'mountains', u'curb', u'base', u'bar', u'sauce', u't-shirt', u'patch', u'snowboard',
               u'bathroom', u'stove', u'luggage', u'hydrant', u'socks', u'carpet', u'paint', u'rail', u'candle',
               u'mouse',
               u'corner', u'license plate', u'front', u'dish', u'sleeve', u'horn', u'statue', u'rope', u'spots',
               u'beak',
               u'tomato', u'wires', u'belt', u'cloth', u'trim', u'rug', u'circle', u'a', u'can', u'street sign',
               u'gloves',
               u'signs', u'teddy bear', u'monitor', u'pipe', u'tv', u'ring', u'rack', u'books', u'jersey', u'black',
               u'goggles', u'banner', u'street light', u'boot', u'words', u'tiles', u'catcher', u'vegetable', u'stick',
               u'hoof', u'skirt', u'distance', u'surface', u'chain', u'purse', u'sneaker', u'tennis racket', u'oven',
               u'bricks', u'this', u'the', u'doors', u'shade', u'ramp', u'cell phone', u'pillar', u'remote', u'hot dog',
               u'boots', u'label', u'gravel', u'ski pole', u'wetsuit', u'kitchen', u'net', u'body', u'park', u'stairs',
               u'ripples', u'sneakers', u'band', u'balcony', u'jet', u'gate', u'cover', u'scarf', u'tennis court',
               u'horns',
               u'elephants', u'knee', u'chairs', u'television', u'graffiti', u'skateboarder', u'tree trunk', u'awning',
               u'name', u'cows', u'shore', u'walkway', u'batter', u'string', u'zebras', u'vegetables', u'umpire',
               u'bun',
               u'tusk', u'tip', u'baby', u'wine', u'river', u'bracelet', u'bottom', u'sofa', u'pen', u'headlights',
               u'cellphone', u'cart', u'outlet', u'refrigerator', u'metal', u'pepper', u'clothes', u'baseball', u'pan',
               u'path', u'necklace', u'giraffes', u'runway', u'crust', u'jar', u'sun', u'train car', u'microwave',
               u'red',
               u'trash can', u'concrete', u'onion', u'horses', u'birds', u'panel', u'curtains', u'buttons', u'square',
               u'doughnut', u'bolt', u'pocket', u'pillows', u'blinds', u'bucket', u'pond', u'beard', u'fire hydrant',
               u'structure', u'painting', u'holder', u'pattern', u'weeds', u'column', u'blue', u'animals', u'doorway',
               u'feathers', u'mug', u'piece', u'scene', u'steps', u'slice', u'whiskers', u'tent', u'shoulder',
               u'cabinets',
               u'log', u'dot', u'speaker', u'shadows', u'green', u'text', u'lettuce', u'outfit', u'poster', u'boats',
               u'frosting', u'cushion', u'toothbrush', u'city', u'tablecloth', u'stop sign', u'snowboarder', u'group',
               u'forest', u'controller', u'teeth', u'star', u'arms', u'pitcher', u'tennis ball', u'fridge', u'scissors',
               u'chimney', u'duck', u'baseball player', u'hood', u'spectator', u'suv', u'train tracks', u'toy', u'case',
               u'pepperoni', u'advertisement', u'crack', u'women', u'item', u'handles', u'symbol', u'slope', u'sheet']
vg500_names_count = [96218, 71049, 70758, 51604, 50913, 48690, 46783, 42933, 42501, 42439, 39749, 34734, 30069, 29934,
                     28382,
                     28290, 27531, 27085, 26171, 25471, 22629, 22405, 22192, 22068, 19202, 18956, 18436, 18204, 17539,
                     17419,
                     16747, 16697, 16567, 16566, 16444, 16162, 15641, 15605, 15593, 15037, 14789, 14450, 13799, 13656,
                     13287,
                     12962, 12947, 12919, 12886, 12828, 12781, 12761, 12389, 12305, 12163, 12008, 11982, 11972, 11918,
                     11898,
                     11797, 11719, 11653, 11626, 11509, 11259, 11014, 10973, 10961, 10788, 10776, 10511, 10366, 10215,
                     10136,
                     9714, 9611, 9514, 9493, 9373, 9270, 9117, 9101, 9015, 8594, 8499, 8454, 8392, 8294, 8290, 8255,
                     8208,
                     8193, 8152, 8147, 8067, 8057, 8019, 7941, 7914, 7840, 7698, 7695, 7678, 7592, 7558, 7446, 7415,
                     7318,
                     7234, 7234, 7220, 7079, 7075, 7001, 6996, 6951, 6931, 6797, 6739, 6734, 6607, 6408, 6303, 6284,
                     6268,
                     6237, 6231, 6218, 6193, 6173, 6157, 6110, 6066, 6021, 5948, 5904, 5904, 5842, 5802, 5722, 5717,
                     5560,
                     5517, 5462, 5403, 5372, 5320, 5311, 5308, 5179, 5175, 5073, 5066, 5064, 5055, 5039, 5017, 4929,
                     4890,
                     4846, 4832, 4805, 4765, 4719, 4712, 4705, 4624, 4613, 4596, 4577, 4571, 4547, 4400, 4330, 4272,
                     4266,
                     4200, 4050, 4023, 4000, 3993, 3987, 3883, 3881, 3880, 3870, 3848, 3847, 3843, 3812, 3800, 3761,
                     3717,
                     3703, 3668, 3651, 3607, 3598, 3540, 3529, 3516, 3503, 3488, 3447, 3423, 3419, 3410, 3385, 3371,
                     3361,
                     3352, 3339, 3315, 3300, 3299, 3293, 3249, 3245, 3240, 3219, 3218, 3212, 3209, 3207, 3181, 3181,
                     3166,
                     3145, 3135, 3129, 3119, 3107, 3097, 3071, 3057, 3017, 2984, 2984, 2965, 2956, 2953, 2952, 2936,
                     2934,
                     2929, 2920, 2890, 2879, 2858, 2850, 2848, 2830, 2796, 2778, 2749, 2744, 2742, 2728, 2724, 2721,
                     2711,
                     2709, 2689, 2677, 2656, 2644, 2633, 2622, 2617, 2601, 2591, 2578, 2571, 2542, 2537, 2517, 2517,
                     2509,
                     2466, 2463, 2456, 2456, 2447, 2437, 2432, 2419, 2413, 2404, 2401, 2396, 2396, 2376, 2375, 2366,
                     2360,
                     2348, 2345, 2318, 2317, 2316, 2314, 2311, 2291, 2287, 2284, 2270, 2266, 2257, 2255, 2254, 2247,
                     2246,
                     2245, 2235, 2234, 2228, 2217, 2216, 2194, 2187, 2167, 2156, 2154, 2135, 2131, 2126, 2122, 2118,
                     2112,
                     2096, 2091, 2086, 2052, 2039, 2030, 2012, 2007, 2007, 1994, 1987, 1987, 1982, 1962, 1955, 1948,
                     1933,
                     1916, 1913, 1908, 1905, 1901, 1896, 1895, 1885, 1878, 1875, 1853, 1836, 1826, 1823, 1807, 1805,
                     1797,
                     1796, 1782, 1777, 1769, 1763, 1760, 1759, 1751, 1747, 1741, 1740, 1735, 1733, 1732, 1723, 1712,
                     1707,
                     1705, 1703, 1681, 1670, 1662, 1650, 1650, 1648, 1647, 1646, 1645, 1641, 1641, 1629, 1620, 1619,
                     1608,
                     1603, 1600, 1600, 1594, 1590, 1590, 1587, 1586, 1585, 1584, 1582, 1578, 1573, 1568, 1560, 1554,
                     1550,
                     1541, 1540, 1532, 1525, 1525, 1522, 1519, 1516, 1509, 1506, 1504, 1495, 1481, 1481, 1477, 1477,
                     1476,
                     1475, 1475, 1475, 1463, 1463, 1460, 1459, 1448, 1443, 1442, 1441, 1436, 1432, 1427, 1427, 1423,
                     1415,
                     1413, 1411, 1410, 1408, 1407, 1395, 1388, 1384, 1382, 1371, 1350, 1350, 1349, 1346, 1344, 1340,
                     1336,
                     1328, 1326, 1317, 1317, 1312, 1307, 1307, 1303, 1301, 1297, 1295, 1290, 1287, 1283, 1280, 1274,
                     1268,
                     1266, 1262, 1259, 1258, 1253, 1252, 1249, 1248, 1242, 1237, 1228, 1227, 1225, 1222, 1221, 1220,
                     1211]
