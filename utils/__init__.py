import sys
import os


fpath = os.path.abspath(__file__)
util_f = os.path.dirname(fpath)
sys.path.append(util_f)

# print('init of PhraseCutDataset/utils:', sys.path[-1])
