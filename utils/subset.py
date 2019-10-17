import json

subsets = ['all', 'c_coco',
           'c20', 'c100', 'c500', 'c21-100', 'c101-500', 'c500+',
           'i_single', 'i_multi', 'i_many',
           'p_name', 'p_att', 'p_att+', 'p_rel', 'p_rel+', 'p_verbose', 'p_attm', 'p_relm',
           't_stuff', 't_obj',
           's_small', 's_mid', 's_large',
           'a20', 'a100', 'a200', 'a21-100', 'a101-200', 'a200+',
           'a_color', 'a_shape', 'a_material', 'a_texture', 'a_state', 'a_adj', 'a_noun', 'a_loc', 'a_count', 'a_bad',
           ]

with open('data/refvg/amt_result/name_att_rel_count_amt.json', 'r') as f:
    count_info = json.load(f)
cat_count_list = count_info['cat']  # list of (cat_name, count), count from high to low
att_count_list = count_info['att']
rel_count_list = count_info['rel']
cat_sorted = [p[0] for p in cat_count_list]
att_sorted = [p[0] for p in att_count_list]
rel_sorted = [p[0] for p in rel_count_list]


def get_subset(phrase_structure, gt_boxes, gt_relative_size):
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

    # top_k: ICCV submission
    # top_k = 501  # top_k starts from 0
    # for ni, name in enumerate(vg500_names):
    #     if name in phrase:
    #         top_k = ni
    #         break

    # c_coco
    if phrase_structure['name'] in coco:
        cond['c_coco'] = True

    # cat freq ranking
    cat_topk = 501
    if phrase_structure['name'] in cat_sorted:
        cat_topk = cat_sorted.index(phrase_structure['name'])

    if cat_topk < 20:
        cond['c20'] = True
    elif cat_topk < 100:
        cond['c21-100'] = True
    elif cat_topk < 500:
        cond['c101-500'] = True
    else:
        cond['c500+'] = True

    if cat_topk < 100:
        cond['c100'] = True
    if cat_topk < 500:
        cond['c500'] = True

    # att freq ranking
    att_topk = 201
    for att in phrase_structure['attributes']:
        if att in att_sorted:
            att_topk = min(att_sorted.index(att), att_topk)

    if att_topk < 20:
        cond['a20'] = True
    elif att_topk < 100:
        cond['a21-100'] = True
    elif att_topk < 200:
        cond['a101-200'] = True
    else:
        cond['a200+'] = True

    if att_topk < 100:
        cond['a100'] = True
    if att_topk < 200:
        cond['a200'] = True

    # phrase mode
    if phrase_structure:
        if len(phrase_structure['attributes']) > 0:
            cond['p_att'] = True
        if len(phrase_structure['attributes']) > 1:
            cond['p_attm'] = True
        if len(phrase_structure['relations']) > 0:
            cond['p_rel'] = True
        if len(phrase_structure['relations']) > 1:
            cond['p_relm'] = True

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
    elif 5 > len(gt_boxes) > 1:
        cond['i_multi'] = True
    elif len(gt_boxes) >= 5:
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
        # if name in phrase:  # iccv submission
        if name in phrase_structure['name']:
            is_stuff = True
            break
    if is_stuff:
        cond['t_stuff'] = True
    else:
        cond['t_obj'] = True

    # att type
    if phrase_structure:
        if phrase_structure['attributes']:
            for att in phrase_structure['attributes']:
                if att in att_color:
                    cond['a_color'] = True
                if att in att_shape:
                    cond['a_shape'] = True
                if att in att_material:
                    cond['a_material'] = True
                if att in att_texture:
                    cond['a_texture'] = True
                if att in att_state:
                    cond['a_state'] = True
                if att in att_adj:
                    cond['a_adj'] = True
                if att in att_noun:
                    cond['a_noun'] = True
                if att in att_loc:
                    cond['a_loc'] = True
                if att in att_bad:
                    cond['a_bad'] = True
    return cond


people_names = ['person', 'people', 'man', 'men', 'woman', 'women', 'kid', 'kids', 'baby', 'boy', 'boys', 'girl',
                'girls', 'child', 'children', 'lady', 'player', 'players', 'guy', 'skier', 'crowd', 'skateboarder',
                'tennis player', 'snowboarder', 'spectators', 'baseball player', 'male', 'skiers', 'he', 'passengers']
stuff_names = ['water', 'waterdrops', 'sea', 'river', 'fog', 'ground', 'field', 'platform', 'rail', 'pavement', 'road',
               'gravel', 'mud', 'dirt', 'snow', 'sand', 'solid', 'hill', 'mountain', 'stone', 'rock', 'wood', 'sky',
               'cloud', 'vegetation', 'straw', 'moss', 'branch', 'leaf', 'leaves', 'bush', 'tree', 'grass',
               'forest', 'railing', 'net', 'cage', 'fence', 'building', 'roof', 'tent', 'bridge', 'skyscraper', 'house',
               'food', 'vegetable', 'salad', 'textile', 'banner', 'blanket', 'curtain', 'cloth',
               'napkin', 'towel', 'mat', 'rug', 'stairs', 'light', 'counter', 'mirror', 'cupboard', 'cabinet', 'shelf',
               'table', 'desk', 'door', 'window', 'floor', 'carpet', 'ceiling', 'wall', 'brick', 'metal', 'plastic',
               'paper', 'cardboard', 'street', 'snow', 'shadow', 'sidewalk', 'plant', 'wave', 'reflection', 'ocean',
               'beach']  # 'flower', 'fruit', 'pillow'

att_color = ['white', 'black', 'blue', 'green', 'brown', 'red', 'yellow', 'gray', 'grey', 'silver', 'orange', 'dark',
             'pink', 'tan', 'purple', 'beige', 'bright', 'gold', 'colorful', 'blonde', 'light brown', 'light blue',
             'colored', 'multicolored', 'maroon', 'dark blue', 'dark brown', 'golden', 'dark green', 'black and white',
             'blond', 'evergreen', 'light colored', 'dark grey', 'multi-colored', 'light skinned', 'dark colored',
             'multi colored', 'blue and white', 'light green', 'bright blue', 'red and white', 'dark gray',
             'cream colored', 'light grey', 'teal', 'navy blue', 'turquoise', 'murky', 'navy']

att_shape = ['large', 'small', 'tall', 'long', 'big', 'short', 'round', 'grassy', 'little', 'thick', 'square', 'thin',
             'sliced', 'curved', 'rectangular', 'flat', 'high', 'wide', 'stacked', 'arched', 'chain link', 'circular',
             'bent', 'cut', 'huge', 'metallic', 'cream', 'pointy', 'extended', 'curly', 'skinny', 'pointed', 'narrow',
             'piled', 'tiny', 'vertical', 'oval', 'curled', 'row', 'straight', 'smaller', 'triangular', 'horizontal',
             'crossed', 'sharp', 'upside down', 'pointing', 'chopped', 'slice', 'rectangle', 'shallow', 'wispy',
             'rounded', 'piece', 'scattered', 'giant', 'slanted', 'tied', 'sparse', 'circle', 'patchy', 'tilted', 'fat',
             'upright', 'larger']

att_material = ['wooden', 'metal', 'wood', 'brick', 'cloudy', 'glass', 'concrete', 'plastic', 'stone', 'tiled',
                'cement', 'dirt', 'sandy', 'leafy', 'fluffy', 'rocky', 'snowy', 'leather', 'steel', 'paper',
                'chocolate', 'tile', 'ceramic', 'grass', 'furry', 'iron', 'water', 'stainless steel', 'hardwood',
                'marble', 'khaki', 'cardboard', 'porcelain', 'snow covered', 'asphalt', 'chrome', 'rock', 'wicker',
                'rubber', 'denim', 'muddy', 'foamy', 'granite', 'bricked', 'gravel', 'snow-covered', 'clay', 'sand',
                'red brick']

att_texture = ['clear', 'wet', 'striped', 'dirty', 'paved', 'shiny', 'painted', 'dry', 'plaid', 'clean', 'blurry',
               'hazy', 'floral', 'rusty', 'splashing', 'cloudless', 'worn', 'smooth', 'checkered', 'spotted',
               'patterned', 'reflecting', 'wrinkled', 'reflective', 'shining', 'choppy', 'rough', 'reflected', 'rusted',
               'lined', 'fuzzy', 'blurred', 'faded', 'printed', 'foggy', 'dusty', 'glazed', 'rippled', 'transparent',
               'frosted']

att_state = ['standing', 'open', 'sitting', 'walking', 'parked', 'hanging', 'playing', 'closed', 'empty', 'on',
             'looking', 'watching', 'flying', 'eating', 'skiing', 'covered', 'surfing', 'skateboarding', 'full',
             'jumping', 'holding', 'close', 'leaning', 'running', 'riding', 'folded', 'waiting', 'moving', 'laying',
             'grazing', 'off', 'talking', 'parking', 'calm', 'posing', 'crashing', 'melted', 'skating', 'seated',
             'raised', 'playing tennis', 'sleeping', 'opened', 'broken', 'resting', 'dried', 'snowboarding',
             'crouching', 'driving', 'fried', 'swinging', 'cracked', 'drinking', 'burnt', 'kneeling', 'stopped',
             'rolling', 'sitting down', 'trimmed', 'breaking', 'crouched', 'bending', 'dressed', 'standing up',
             'wrapped', 'attached', 'floating', 'rolled up', 'lying', 'squatting', 'held', 'cutting', 'outstretched',
             'illuminated', 'reading', 'turned', 'swimming', 'turning']

att_adj = ['young', 'old', 'smiling', 'bare', 'light', 'part', 'dead', 'cooked', 'framed', 'pictured', 'overcast',
           'leafless', 'beautiful', 'stuffed', 'growing', 'decorative', 'electrical', 'electric', 'bald', 'older',
           'lit', 'fresh', 'lush', 'wire', 'happy', 'puffy', 'sunny', 'ripe', 'male', 'palm', 'shirtless', 'female',
           'asian', 'hairy', 'ornate', 'bushy', 'deep', 'wavy', 'toasted', 'barefoot', 'potted', 'short sleeved',
           'edge', 'wild', 'busy', 'decorated', 'double decker', 'long sleeved', 'partial', 'soft', 'flat screen',
           'healthy', 'floppy', 'plain', 'filled', 'modern', 'long sleeve', 'overgrown', 'displayed', 'digital', 'cast',
           'airborne', 'delicious', 'hard', 'carpeted', 'heavy', 'new', 'grilled', 'sleeveless', 'pale', 'pretty',
           'different', 'american', 'nice', 'fake', 'designed', 'cute', 'manicured', 'written']

att_noun = ['tennis', 'baseball', "man's", 'baby', 'train', "woman's", 'pine', 'tree', 'street', 'passenger', 'traffic',
            'computer', 'adult', 'ski', 'man', 'wine', 'burgundy', 'stop', 'snow', 'bathroom', 'city', 'teddy',
            'kitchen', 'patch', 'nike', 'woman', 'wall', 'fire', 'clock', 'window', 'straw', 'flower', 'ground',
            'pizza', 'apple', 'power', 'coffee', 'tennis player', 'toy', 'ocean']

att_loc = ['distant', 'background', 'back', 'behind', 'in background', 'side', 'up', 'rear', 'down', 'top', 'far',
           'overhead', 'low', 'above', 'outdoors', 'in distance', 'in the background', 'inside', 'outdoor', 'bottom',
           'in air']

att_bad = ['here', 'present', 'wearing', 'in the picture', 'some', 'daytime', 'existing', 'ready', 'made', 'in picture']


refvg_names20 = ['man', 'sky', 'wall', 'building', 'shirt', 'tree', 'grass', 'woman', 'person', 'ground', 'trees',
                 'window', 'water', 'table', 'sign', 'head', 'fence', 'floor', 'pole', 'road', 'hair', 'pants',
                 'people', 'car', 'door', 'shadow', 'street', 'clouds', 'sidewalk', 'plate', 'chair', 'jacket', 'leg',
                 'field', 'line', 'train', 'leaves', 'boy', 'girl', 'snow', 'hand', 'bus', 'background', 'tail', 'face',
                 'dog', 'light', 'shorts', 'roof', 'arm', 'cloud', 'cat', 'windows', 'jeans', 'plane', 'umbrella',
                 'bench', 'horse', 'bag', 'giraffe', 'food', 'tracks', 'legs', 'glass', 'boat', 'ceiling', 'elephant',
                 'truck', 'ear', 'wave', 'house', 'clock', 'beach', 'dirt', 'lines', 'sand', 'bed', 'bowl', 'mirror',
                 'pillow', 'surfboard', 'hill', 'motorcycle', 'box', 'top', 'player', 'counter', 'handle', 'board',
                 'mountain', 'bear', 'zebra', 'post', 'toilet', 'shelf', 'flowers', 'hat', 'bike', 'wheel', 'pizza',
                 'laptop', 'tire', 'ocean', 'wing', 'bush', 'shoes', 'cow', 'reflection', 'sink', 'picture', 'trunk',
                 'coat', 'plant', 'bird', 'tower', 'skateboard', 'cabinet', 'airplane', 'desk', 'lamp', 'sheep', 'rock',
                 'neck', 'stripe', 'bottle', 'keyboard', 'paper', 'tile', 'seat', 'frame', 'waves', 'kite', 'lights',
                 'rocks', 'writing', 'lady', 'buildings', 'screen', 'railing', 'cup', 'bushes', 'couch', 'suit',
                 'blanket', 'uniform', 'vehicle', 'sweater', 'helmet', 'windshield', 'shoe', 'curtain', 'mountains',
                 'racket', 'child', 'fork', 'court', 'cars', 'track', 'cake', 'vase', 'letters', 'back', 'dress',
                 'knife', 'men', 'towel', 'carpet', 'platform', 'tie', 'edge', 'flower', 'napkin', 'banana', 'bridge',
                 'stripes', 'container', 'wheels', 'logo', 'suitcase', 'tray', 'branch', 'van', 'pavement', 'bread',
                 'stand', 'computer', 'curb', 'nose', 'rug', 'guy', 'front', 'area', 'plants', 'basket', 'wires',
                 'side', 'wire', 'animal', 'cheese', 'branches', 'leaf', 'collar', 'broccoli', 'fur', 'glasses',
                 'design', 'engine', 'sandwich', 'mane', 'ears', 'stove', 'wood', 'phone', 't-shirt', 'monitor',
                 'backpack', 'surfer', 'cap', 'book', 'pond', 'lid', 'tv', 'signs', 'hands', 'skier', 'poles', 'flag',
                 'net', 'foot', 'bicycle', 'spoon', 'bat', 'shore', 'skis', 'meat', 'catcher', 'feet', 'bananas',
                 'tiles', 'walkway', 'sleeve', 'fruit', 'part', 'rail', 'cabinets', 'pot', 'cord', 'rope', 'mouth',
                 'path', 'ramp', 'body', 'cloth', 'oven', 'gate', 'sauce', 'orange', 'drawer', 'kid', 'eyes', 'doors',
                 'crust', 'wetsuit', 'snowboard', 'statue', 'jersey', 'sofa', 'gravel', 'refrigerator', 'tennis court',
                 'teddy bear', 'surface', 'apple', 'chairs', 'umpire', 'curtains', 'base', 'corner', 'bricks', 'runway',
                 'donut', 'hydrant', 'pipe', 'skirt', 'doorway', 'banner', 'forest', 'distance', 'strap', 'frisbee',
                 'luggage', 'pillows', 'lettering', 'stairs', 'vest', 'air', 'vegetables', 'television', 'books',
                 'trim', 'blinds', 'room', 'microwave', 'smiling woman', 'river', 'painting', 'words', 'paint',
                 'tennis racket', 'balcony', 'shadows', 'animals', 'dish', 'headboard', 'photo', 'white', 'ski',
                 'elephants', 'park', 'fridge', 'tree trunk', 'concrete', 'zebras', 'object', 'batter', 'clothes',
                 'glove', 'bar', 'jet', 'cover', 'women', 'cart', 'pillar', 'boats', 'purse', 'bottom', 'brick',
                 'group', 'stone', 'bun', 'steps', 'paw', 'awning', 'giraffes', 'lake', 'horses', 'outfit', 'graffiti',
                 'rack', 'finger', 'crowd', 'baby', 'skateboarder', 'tablecloth', 'structure', 'this', 'ripples',
                 'bathroom', 'letter', 'shade', 'scarf', 'arms', 'hills', 'faucet', 'birds', 'stem', 'cushion',
                 'street light', 'word', 'hillside', 'street sign', 'slope', 'whiskers', 'weeds', 'store', 'feathers',
                 'stick', 'land', 'traffic light', 'sun', 'smoke', 'lettuce', 'parking lot', 'ball', 'string', 'pan',
                 'walls', 'gloves', 'tennis player', 'players', 'tent', 'wings', 'carrot', 'sheet', 'tank', 'chain',
                 'foam', 'advertisement', 'cows', 'fire hydrant', 'socks', 'case', 'boots', 'bumper', 'train tracks',
                 'crosswalk', 'trash can', 'row', 'kitchen', 'column', 'slice', 'log', 'chest', 'metal', 'vegetable',
                 'frosting', 'poster', 'stop sign', 'patch', 'mat', 'spot', 'mouse', 'can', 'couple', 'panel', 'city',
                 'pattern', 'train car', 'tarmac', 'ski pole', 'skies', 'circle', 'suv', 'airport', 'clothing',
                 'tomato', 'carrots', 'vegetation', 'the', 'a', 'number', 'lawn', 'tub', 'label', 'belt', 'sneakers',
                 'camera', 'shower', 'umbrellas', 'countertop', 'remote', 'numbers', 'scissors', 'toy', 'vehicles',
                 'spots', 'headlights', 'black', 'hot dog', 'fencing', 'salad', 'plates', 'houses', 'dock', 'name',
                 'pitcher', 'eye', 'barrier', 'necklace', 'arrow', 'fabric', 'snowboarder', 'oranges', 'sweatshirt',
                 'jar', 'boot', 'text', 'spectators', 'doughnut', 'mug', 'crack', 'fireplace', 'sunlight', 'speaker',
                 'ladder', 'palm tree', 'display', 'tires', 'hood', 'pictures', 'cement', 'bathtub', 'wine', 'game',
                 'papers', 'tarp', 'comforter', 'power lines', 'baseball player', 'holder', 'station', 'border',
                 'tank top', 'ledge', 'toothbrush', 'ship', 'stool', 'thumb', 'fingers', 'drawers', 'grill', 'canopy',
                 'horizon', 'tennis', 'ribbon', 'kites', 'headlight', 'bucket', 'hoodie', 'cupboard', 'woods',
                 'sticker', 'apples', 'piece', 'blue', 'bookshelf', 'sunglasses', 'bottles', 'hole', 'handles',
                 'stones', 'bears', 'pocket', 'buttons', 'rails', 'cell phone', 'hay', 'toilet seat', 'counter top',
                 'towels', 'dresser', 'shoulder', 'clock tower', 'trail', 'beam', 'mound', 'scooter', 'chicken',
                 'toppings', 'rider', 'coffee table', 'apron', 'planes', 'fan', 'onion', 'church', 'lot', 'shrubs',
                 'brush', 'racquet', 'utensil', 'foliage', 'flooring', 'beard', 'shrub', 'posts', 'skater', 'pile',
                 'cable', 'seats', 'archway', 'propeller', 'items', 'entrance', 'flags', 'pepper', 'light pole', 'skin',
                 'bikes', 'donuts', 'hotdog', 'arch', 'sea', 'cutting board', 'boxes', 'pasture', 'bin', 'sock',
                 'parking meter', 'billboard', 'outdoors', 'meter', 'paws', 'horns', 'motorbike', 'machine', 'shelves',
                 'trains', 'step', 'cellphone', 'planter', 'vent', 'tag', 'kids', 'bedspread', 'tshirt', 'bars',
                 'belly', 'sheets', 'tomatoes', 'green', 'children', 'male', 'hose', 'rice', 'egg', 'trailer',
                 'wine glass', 'duck', 'yard', 'cab', 'icing', 'decoration', 'rim', 'cone', 'stands', 'section', 'bags',
                 'pen', 'boys', 'drink', 'blade', 'steeple', 'square', 'lamp post', 'dugout', 'candle', 'red',
                 'benches', 'staircase', 'liquid', 'placemat', 'roadway', 'controller', 'onions', 'straw', 'asphalt',
                 'pier', 'bacon', 'dogs', 'ski poles', 'skiers', 'urinal', 'right', 'dishwasher', 'leash', 'blender',
                 'baseball bat', 'signal', 'carriage', 'keys', 'plastic', 'cockpit', 'boulder', 'porch', 'horn', 'deck',
                 'left', 'dessert', 'lamb', 'wrist', 'sail', 'saucer', 'shoreline', 'surf board', 'saddle', 'he',
                 'pepperoni', 'streetlight', 'fruits', 'newspaper', 'cables', 'meal', 'bunch', 'scene', 'footprints',
                 'pastry', 'taxi', 'bookcase', 'mud', 'splash', 'sticks', 'fish', 'moss', 'columns', 'blouse', 'crane',
                 'strip', 'front leg', 'nightstand', 'markings', 'pool', 'strings', 'button', 'print', 'table cloth',
                 'grate', 'potatoes', 'street lamp', 't shirt', 'rod', 'kitten', 'soup', 'computer monitor', 'sleeves',
                 'menu', 'beer', 'toilet bowl', 'slab', 'island', 'goat', 'motorcycles', 'intersection', 'girls',
                 'driveway', 'topping', 'trash', 'view', 'dome', 'buses', 'art', 'fries', 'watch', 'item', 'tape',
                 'cords', 'sneaker', 'eggs', 'garden', 'front wheel', 'cliff', 'forehead', 'wallpaper', 'doll',
                 'remote control', 'image', 'pant', 'train station', 'aircraft', 'garage', 'utensils', 'landscape',
                 'containers', 'harness', 'traffic', 'clock face', 'crate', 'palm trees', 'enclosure', 'wrapper',
                 'bleachers', 'silverware', 'pathway', 'ketchup', 'furniture', 'tee shirt', 'traffic signal',
                 'restaurant', 'napkins', 'shower curtain', 'cups', 'band', 'left hand', 'toilet paper', 'train track',
                 'zipper', 'bull', 'equipment', 'right leg', 'front tire', 'magazine', 'hooves', 'platter', 'short',
                 'peppers', 'juice', 'stop', 'cage', 'day', 'garbage can', 'train engine', 'coffee', 'front legs',
                 'greenery', 'two', 'stack', 'sausage', 'stuffed animal', 'pine tree', 'cabin', 'sedan', 'yellow',
                 'herd', 'bank', 'light post', 'pad', 'right hand', 'ripple', 'pillars', 'stroller', 'beak', 'crumbs',
                 'wrinkles', 'handlebars', 'surfboards', 'lemon', 'mattress', 'dispenser', 'barn', 'block', 'stairway',
                 'left arm', 'sailboat', 'spinach', 'divider', 'table top', 'space', 'hedge', 'outside', 'stems',
                 'telephone', 'middle', 'cats', 'artwork', 'beans', 'pipes', 'female', 'bow', 'spokes', 'plank',
                 'foreground', 'home', 'pebbles', 'support', 'chocolate', 'snout', 'knee', 'power line', 'sunset',
                 'wool', 'soil', 'cupcake', 'barrel', 'sandals', 'mouse pad', 'stomach', 'spectator', 'drawing', 'ice',
                 'tusk', 'brown', 'ad', 'family', 'trouser', 'mustard', 'straps', 'sculpture', 'lap', 'slats',
                 'skate park', 'eyeglasses', 'landing gear', 'bus stop', 'cones', 'trashcan', 'pickle', 'driver',
                 'rose', 'marks', 'wipers', 'cabinet door', 'right arm', 'floors', 'trunks', 'freezer', 'highway',
                 'spatula', 'center', 'adult', 'necktie', 'railings', 'bicycles', 'water bottle', 'printer', 'shop',
                 'light fixture', 'cream', 'doughnuts', 'knees', 'lamp shade', 'toddler', 'material', 'knob',
                 'cardboard', 'toast', 'baseball', 'headband', 'two people', 'suitcases', 'potato', 'fountain',
                 'bridle', 'spray', 'foil', 'mushrooms', 'covering', 'skyscraper', 'netting', 'paddle', 'she', 'clocks',
                 'shed', 'mitt', 'boards', 'cracks', 'ham', 'officer', 'device', 'tree branches', 'map', 'wagon',
                 'canoe', 'cucumber', 'tables', 'tunnel', 'bouquet', 'decorations', 'mountain range', 'elbow', 'grapes',
                 'french fries', 'left leg', 'roll', 'ring', 'patio', 'drapes', 'ottoman', 'logs', 'mushroom', 'human',
                 'puddle', 'pair', 'traffic lights', 'bark', 'handbag', 'veggies', 'stadium', 'lamps', 'candles',
                 'slat', 'wake', 'rear', 'audience', 'jeep', 'opening', 'fender', 'shirts', 'knobs', 'notebook',
                 'costume', 'soda', 'chandelier', 'ski lift', 'living room', 'guitar', 'wiper', 'lighting', 'shutters',
                 'engines', 'vanity', 'inside', 'tip', 'robe', 'overpass', 'card', 'tube', 'mask', 'window sill',
                 'chimney', 'stove top', 'bedroom', 'chips', 'quilt', 'lampshade', 'blind', 'zoo', 'stain', 'railroad',
                 'bracelet', 'basin', 'moped', 'strawberry', 'airplanes', 'symbol', 'beams', 'blocks', 'visor',
                 'armrest', 'feather', 'pasta', 'ice cream', 'slices', 'shrubbery', 'pots', 'sugar', 'greens', 'cattle',
                 'noodles', 'cleats', 'vases', 'headphones', 'siding', 'panels', 'shelter', 'pane', 'burner',
                 'passengers', 'radiator', 'palm', 'harbor', 'booth', 'calf', 'end', 'lighthouse', 'left ear', 'turf',
                 'appliance', 'right ear', 'debris', 'locomotive', 'hotel', 'ducks', 'polar bear', 'fans',
                 'pizza slice', 'stage', 'package', 'dishes', 'baseboard', 'kettle', 'mast', 'handlebar', 'steam',
                 'toaster', 'milk', 'median', 'color', 'vines', 'fixture', 'back legs', 'stream', 'trucks',
                 'toilet lid', 'bowls', 'tree branch', 'mother', 'computer screen', 'arm rest', 'stickers', 'weed',
                 'antenna', 'picnic table', 'town', 'stalk', 'handle bars', 'oar', 'license plate', 'roman numerals',
                 'strawberries', 'carton', 'gear', 'sprinkles', 'goggles', 'mantle', 'down', 'skyline', 'stop light',
                 'telephone pole', 'storefront', 'cheek', 'skiier', 'cabbage', 'it', 'trash bin', 'marking', 'ponytail',
                 'shutter', 'wine bottle', 'pieces', 'stump', 'cross', 'back wheel', 'star', 'ski slope', 'cooler',
                 'whisker', 'roadside', 'pumpkin', 'twig', 'mark', 'parasail', 'mousepad', 'roses', 'burger', 'cookie',
                 'traffic sign', 'range', 'minivan', 'coffee cup', 'pie', 'mulch', 'portion', 'bubbles', 'ropes',
                 'caps', 'time', 'biker', 'spire', 'lamppost', 'figure', 'chin', 'calendar', 'street lights', 'toys',
                 'steering wheel', 'pack', 'holes', 'sill', 'thigh', 'night stand', 'terrain', 'chains', 'tusks',
                 'pineapple', 'magazines', 'policeman', 'window pane', 'seagull', 'market', 'mirrors', 'drain',
                 'castle', 'pedestrian', 'street signs', 'set', 'surfers', 'heart', 'designs', 'towel rack', 'balloon',
                 'match', 'sign post', 'silver', 'coffee maker', 'bracket', 'teapot', 'place', 'stars', 'teeth',
                 'triangle', 'blazer', 'lips', 'beverage', 'objects', 'fence post', 'soap', 'molding', 'trolley',
                 'muzzle', 'back tire', 'windshield wiper', 'worker', 'peel', 'bow tie', 'paper plate', 'pedestal',
                 'puppy', 'lemons', 'someone', 'sandwiches', 'scale', 'wii', 'brocolli', 'microphone', 'gray', 'tongue',
                 'patches', 'wooden', 'end table', 'tool', 'lift', 'handrail', 'corn', 'cuff', 'graphic', 'note',
                 'olives', 'jug', 'soap dispenser', 'vine', 'facial hair', 'bikini', 'desert', 'limb', 'watermark',
                 'blades', 'police officer', 'piano', 'stuff', 'key', 'front window', 'parrot', 'dots', 'passenger',
                 'outlet', 'bristles', 'deer', 'twigs', 'reins', 'diamond', 'feeder', 'grey', 'stall', 'reflections',
                 'petal', 'bib', 'shades', 'seeds', 'picture frame', 'cauliflower', 'coaster', 'hedges', 'hats',
                 'trousers', 'cans', 'plaque', 'shaker', 'arrows', 'numerals', 'windowsill', 'salt shaker', 'cushions',
                 'video games', 'garbage', 'eyebrows', 'claws', 'skiis', 'rain', 'character', 'ski pants', 'stains',
                 'spices', 'an', 'large', 'motorcyclist', 'knee pads', 'beanie', 'candy', 'up', 'monkey', 'mannequin',
                 'pastries', 'squares', 'french fry', 'ram', 'tissue', 'photographer', 'beads', 'back leg', 'pony',
                 'produce', 'goose', 'hot dogs', 'knives', 'nails', 'lanyard', 'sandal', 'air conditioner', 'petals',
                 'control', 'tabletop', 'tennis shoe']

coco = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
        'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
        'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
        'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
        'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
        'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
