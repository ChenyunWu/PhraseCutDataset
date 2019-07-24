subsets = ['all',
           'c20', 'c100', 'c500', 'c500+', 'c21-100', 'c101-500',
           'i_single', 'i_multi', 'i_many',
           'p_name', 'p_att', 'p_att+', 'p_rel', 'p_rel+', 'p_verbose',
           't_stuff', 't_obj',
           's_small', 's_mid', 's_large',
           'a_color', 'a_shape', 'a_material', 'a_texture', 'a_state', 'a_adj', 'a_noun', 'a_loc', 'a_count', 'a_bad'
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

    # top_k: ICCV submission
    # top_k = 501  # top_k starts from 0
    # for ni, name in enumerate(vg500_names):
    #     if name in phrase:
    #         top_k = ni
    #         break
    top_k = 501
    if phrase_structure['name'] in refvg_names20:
        top_k = refvg_names20.index(phrase_structure['name'])

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

coco79 = [u'bicycle', u'car', u'motorcycle', u'airplane', u'bus', u'train', u'truck', u'boat', u'traffic light',
          u'fire hydrant', u'stop sign', u'parking meter', u'bench', u'bird', u'cat', u'dog', u'horse', u'sheep',
          u'cow', u'elephant', u'bear', u'zebra', u'giraffe', u'backpack', u'umbrella', u'handbag', u'tie',
          u'suitcase', u'frisbee', u'skis', u'snowboard', u'sports ball', u'kite', u'baseball bat', u'baseball glove',
          u'skateboard', u'surfboard', u'tennis racket', u'bottle', u'wine glass', u'cup', u'fork', u'knife', u'spoon',
          u'bowl', u'banana', u'apple', u'sandwich', u'orange', u'broccoli', u'carrot', u'hot dog', u'pizza', u'donut',
          u'cake', u'chair', u'couch', u'potted plant', u'bed', u'dining table', u'toilet', u'tv', u'laptop', u'mouse',
          u'remote', u'keyboard', u'cell phone', u'microwave', u'oven', u'toaster', u'sink', u'refrigerator', u'book',
          u'clock', u'vase', u'scissors', u'teddy bear', u'hair drier', u'toothbrush']

vg500_names = [u'man', u'person', u'window', u'shirt', u'tree', u'building', u'wall', u'sky', u'woman', u'sign',
               u'ground', u'grass', u'table', u'pole', u'head', u'light', u'car', u'water', u'hair', u'hand', u'people',
               u'leg', u'clouds', u'trees', u'plate', u'leaves', u'ear', u'pants', u'fence', u'door', u'chair', u'eye',
               u'hat', u'floor', u'train', u'road', u'jacket', u'street', u'snow', u'wheel', u'line', u'shadow', u'boy',
               u'nose', u'shoe', u'letter', u'cloud', u'clock', u'boat', u'tail', u'handle', u'sidewalk', u'field',
               u'girl', u'flower', u'leaf', u'horse', u'helmet', u'bus', u'shorts', u'bird', u'elephant', u'giraffe',
               u'plane', u'umbrella', u'dog', u'bag', u'arm', u'face', u'windows', u'zebra', u'glass', u'sheep', u'cow',
               u'bench', u'cat', u'food', u'bottle', u'tile', u'rock', u'kite', u'stripe', u'post', u'tire', u'number',
               u'truck', u'flowers', u'logo', u'surfboard', u'shoes', u'bear', u'roof', u'picture', u'cap', u'spot',
               u'bowl', u'glasses', u'motorcycle', u'jeans', u'skateboard', u'player', u'background', u'foot', u'box',
               u'bike', u'mirror', u'pizza', u'pillow', u'top', u'tracks', u'shelf', u'lights', u'legs', u'house',
               u'mouth', u'dirt', u'part', u'cup', u'plant', u'board', u'trunk', u'banana', u'counter', u'bush',
               u'ball', u'wave', u'lines', u'button', u'bed', u'lamp', u'sink', u'brick', u'beach', u'flag', u'writing',
               u'sand', u'coat', u'neck', u'vase', u'letters', u'paper', u'seat', u'glove', u'wing', u'child',
               u'vehicle', u'toilet', u'reflection', u'laptop', u'airplane', u'phone', u'book', u'sunglasses',
               u'branch', u'edge', u'cake', u'rocks', u'desk', u'tie', u'frisbee', u'animal', u'tower', u'hill',
               u'eyes', u'stripes', u'cabinet', u'mountain', u'headlight', u'container', u'frame', u'lady', u'wheels',
               u'ceiling', u'ocean', u'towel', u'racket', u'skier', u'keyboard', u'hands', u'design', u'windshield',
               u'back', u'pot', u'feet', u'basket', u'track', u'fork', u'bat', u'waves', u'fruit', u'orange', u'finger',
               u'guy', u'railing', u'engine', u'suit', u'broccoli', u'knife', u'couch', u'collar', u'cars', u'sock',
               u'apple', u'backpack', u'suitcase', u'knob', u'surfer', u'cheese', u'screen', u'donut', u'dress',
               u'buildings', u'blanket', u'paw', u'bananas', u'bicycle', u'bushes', u'van', u'lettering', u'tag',
               u'sticker', u'lid', u'bread', u'photo', u'skis', u'sweater', u'uniform', u'curtain', u'watch', u'tray',
               u'stand', u'stone', u'ears', u'men', u'wood', u'wire', u'sandwich', u'court', u'branches', u'room',
               u'bridge', u'traffic light', u'stem', u'white', u'mane', u'napkin', u'word', u'pavement', u'cone',
               u'faucet', u'fur', u'kid', u'carrot', u'camera', u'arrow', u'object', u'air', u'numbers', u'ski',
               u'hole', u'drawer', u'key', u'spoon', u'wrist', u'computer', u'platform', u'area', u'side', u'plants',
               u'meat', u'poles', u'cord', u'vest', u'strap', u'mountains', u'curb', u'base', u'bar', u'sauce',
               u't-shirt', u'patch', u'snowboard', u'bathroom', u'stove', u'luggage', u'hydrant', u'socks', u'carpet',
               u'paint', u'rail', u'candle', u'mouse', u'corner', u'license plate', u'front', u'dish', u'sleeve',
               u'horn', u'statue', u'rope', u'spots', u'beak', u'tomato', u'wires', u'belt', u'cloth', u'trim', u'rug',
               u'circle', u'a', u'can', u'street sign', u'gloves', u'signs', u'teddy bear', u'monitor', u'pipe', u'tv',
               u'ring', u'rack', u'books', u'jersey', u'black', u'goggles', u'banner', u'street light', u'boot',
               u'words', u'tiles', u'catcher', u'vegetable', u'stick', u'hoof', u'skirt', u'distance', u'surface',
               u'chain', u'purse', u'sneaker', u'tennis racket', u'oven', u'bricks', u'this', u'the', u'doors',
               u'shade', u'ramp', u'cell phone', u'pillar', u'remote', u'hot dog', u'boots', u'label', u'gravel',
               u'ski pole', u'wetsuit', u'kitchen', u'net', u'body', u'park', u'stairs', u'ripples', u'sneakers',
               u'band', u'balcony', u'jet', u'gate', u'cover', u'scarf', u'tennis court', u'horns', u'elephants',
               u'knee', u'chairs', u'television', u'graffiti', u'skateboarder', u'tree trunk', u'awning', u'name',
               u'cows', u'shore', u'walkway', u'batter', u'string', u'zebras', u'vegetables', u'umpire', u'bun',
               u'tusk', u'tip', u'baby', u'wine', u'river', u'bracelet', u'bottom', u'sofa', u'pen', u'headlights',
               u'cellphone', u'cart', u'outlet', u'refrigerator', u'metal', u'pepper', u'clothes', u'baseball', u'pan',
               u'path', u'necklace', u'giraffes', u'runway', u'crust', u'jar', u'sun', u'train car', u'microwave',
               u'red', u'trash can', u'concrete', u'onion', u'horses', u'birds', u'panel', u'curtains', u'buttons',
               u'square', u'doughnut', u'bolt', u'pocket', u'pillows', u'blinds', u'bucket', u'pond', u'beard',
               u'fire hydrant', u'structure', u'painting', u'holder', u'pattern', u'weeds', u'column', u'blue',
               u'animals', u'doorway', u'feathers', u'mug', u'piece', u'scene', u'steps', u'slice', u'whiskers',
               u'tent', u'shoulder', u'cabinets', u'log', u'dot', u'speaker', u'shadows', u'green', u'text', u'lettuce',
               u'outfit', u'poster', u'boats', u'frosting', u'cushion', u'toothbrush', u'city', u'tablecloth',
               u'stop sign', u'snowboarder', u'group', u'forest', u'controller', u'teeth', u'star', u'arms', u'pitcher',
               u'tennis ball', u'fridge', u'scissors', u'chimney', u'duck', u'baseball player', u'hood', u'spectator',
               u'suv', u'train tracks', u'toy', u'case', u'pepperoni', u'advertisement', u'crack', u'women', u'item',
               u'handles', u'symbol', u'slope', u'sheet']
vg500_names_count = [96218, 71049, 70758, 51604, 50913, 48690, 46783, 42933, 42501, 42439, 39749, 34734, 30069, 29934,
                     28382, 28290, 27531, 27085, 26171, 25471, 22629, 22405, 22192, 22068, 19202, 18956, 18436, 18204,
                     17539, 17419, 16747, 16697, 16567, 16566, 16444, 16162, 15641, 15605, 15593, 15037, 14789, 14450,
                     13799, 13656, 13287, 12962, 12947, 12919, 12886, 12828, 12781, 12761, 12389, 12305, 12163, 12008,
                     11982, 11972, 11918, 11898, 11797, 11719, 11653, 11626, 11509, 11259, 11014, 10973, 10961, 10788,
                     10776, 10511, 10366, 10215, 10136, 9714, 9611, 9514, 9493, 9373, 9270, 9117, 9101, 9015, 8594,
                     8499, 8454, 8392, 8294, 8290, 8255, 8208, 8193, 8152, 8147, 8067, 8057, 8019, 7941, 7914, 7840,
                     7698, 7695, 7678, 7592, 7558, 7446, 7415, 7318, 7234, 7234, 7220, 7079, 7075, 7001, 6996, 6951,
                     6931, 6797, 6739, 6734, 6607, 6408, 6303, 6284, 6268, 6237, 6231, 6218, 6193, 6173, 6157, 6110,
                     6066, 6021, 5948, 5904, 5904, 5842, 5802, 5722, 5717, 5560, 5517, 5462, 5403, 5372, 5320, 5311,
                     5308, 5179, 5175, 5073, 5066, 5064, 5055, 5039, 5017, 4929, 4890, 4846, 4832, 4805, 4765, 4719,
                     4712, 4705, 4624, 4613, 4596, 4577, 4571, 4547, 4400, 4330, 4272, 4266, 4200, 4050, 4023, 4000,
                     3993, 3987, 3883, 3881, 3880, 3870, 3848, 3847, 3843, 3812, 3800, 3761, 3717, 3703, 3668, 3651,
                     3607, 3598, 3540, 3529, 3516, 3503, 3488, 3447, 3423, 3419, 3410, 3385, 3371, 3361, 3352, 3339,
                     3315, 3300, 3299, 3293, 3249, 3245, 3240, 3219, 3218, 3212, 3209, 3207, 3181, 3181, 3166, 3145,
                     3135, 3129, 3119, 3107, 3097, 3071, 3057, 3017, 2984, 2984, 2965, 2956, 2953, 2952, 2936, 2934,
                     2929, 2920, 2890, 2879, 2858, 2850, 2848, 2830, 2796, 2778, 2749, 2744, 2742, 2728, 2724, 2721,
                     2711, 2709, 2689, 2677, 2656, 2644, 2633, 2622, 2617, 2601, 2591, 2578, 2571, 2542, 2537, 2517,
                     2517, 2509, 2466, 2463, 2456, 2456, 2447, 2437, 2432, 2419, 2413, 2404, 2401, 2396, 2396, 2376,
                     2375, 2366, 2360, 2348, 2345, 2318, 2317, 2316, 2314, 2311, 2291, 2287, 2284, 2270, 2266, 2257,
                     2255, 2254, 2247, 2246, 2245, 2235, 2234, 2228, 2217, 2216, 2194, 2187, 2167, 2156, 2154, 2135,
                     2131, 2126, 2122, 2118, 2112, 2096, 2091, 2086, 2052, 2039, 2030, 2012, 2007, 2007, 1994, 1987,
                     1987, 1982, 1962, 1955, 1948, 1933, 1916, 1913, 1908, 1905, 1901, 1896, 1895, 1885, 1878, 1875,
                     1853, 1836, 1826, 1823, 1807, 1805, 1797, 1796, 1782, 1777, 1769, 1763, 1760, 1759, 1751, 1747,
                     1741, 1740, 1735, 1733, 1732, 1723, 1712, 1707, 1705, 1703, 1681, 1670, 1662, 1650, 1650, 1648,
                     1647, 1646, 1645, 1641, 1641, 1629, 1620, 1619, 1608, 1603, 1600, 1600, 1594, 1590, 1590, 1587,
                     1586, 1585, 1584, 1582, 1578, 1573, 1568, 1560, 1554, 1550, 1541, 1540, 1532, 1525, 1525, 1522,
                     1519, 1516, 1509, 1506, 1504, 1495, 1481, 1481, 1477, 1477, 1476, 1475, 1475, 1475, 1463, 1463,
                     1460, 1459, 1448, 1443, 1442, 1441, 1436, 1432, 1427, 1427, 1423, 1415, 1413, 1411, 1410, 1408,
                     1407, 1395, 1388, 1384, 1382, 1371, 1350, 1350, 1349, 1346, 1344, 1340, 1336, 1328, 1326, 1317,
                     1317, 1312, 1307, 1307, 1303, 1301, 1297, 1295, 1290, 1287, 1283, 1280, 1274, 1268, 1266, 1262,
                     1259, 1258, 1253, 1252, 1249, 1248, 1242, 1237, 1228, 1227, 1225, 1222, 1221, 1220, 1211]
