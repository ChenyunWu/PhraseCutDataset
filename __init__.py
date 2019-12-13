import sys
import os


fpath = os.path.abspath(__file__)
dataset_f = os.path.dirname(fpath)
sys.path.append(dataset_f)

# print('init of PhraseCutDataset:', sys.path[-1])
