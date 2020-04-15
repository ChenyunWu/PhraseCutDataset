import os
from tqdm import tqdm
from PhraseCutDataset.utils.refvg_loader import RefVGLoader
from PhraseCutDataset.utils.visualize_utils import gt_visualize_to_file
from PhraseCutDataset.utils.file_paths import gt_plot_path_color


loader = RefVGLoader()
plot_phrases = ['walking people', 'wipers on trains', 'zebra lying on savanna', 'black shirt', 'glass bottles',
                'blonde hair']

for img_id in tqdm(loader.img_ids):
    img_data = loader.get_img_ref_data(img_id)
    for task_i, phrase in enumerate(img_data['phrases']):
        if phrase in plot_phrases:
            task_id = img_data['task_ids'][task_i]
            gt_visualize_to_file(img_data, task_id, fig_path=os.path.join(gt_plot_path_color, phrase, '%s.jpg' % task_id),
                                 gray_img=False)
            print(task_id, phrase)
