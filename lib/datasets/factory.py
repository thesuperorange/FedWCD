# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__sets = {}
from datasets.pascal_voc import pascal_voc
from datasets.coco import coco
from datasets.cityscape import cityscape
from datasets.foggy_cityscape import foggy_cityscape
from datasets.kitti import kitti
from datasets.sim10k import sim10k
from datasets.bdd100k import bdd100k

from datasets.imagenet import imagenet
from datasets.vg import vg
from datasets.wider_face import wider_face
from datasets.mi3 import mi3
from datasets.kaist import kaist
import numpy as np

# Set up voc_<year>_<split>
for year in ['2007', '2012']:
  for split in ['train', 'val', 'trainval', 'test']:
    name = 'voc_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: pascal_voc(split, year))
    
    
for year in ['2007', '2012']:
  for split in ['trainval', 'test']:  # trainval=train=2975, test=val=500
    name = 'cityscape_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: cityscape(split, year))

for year in ['2007', '2012']:
  for split in ['train', 'test']:  #test=500=val
    name = 'foggy_cityscape_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: foggy_cityscape(split, year))    
    
for split in ['trainval10k']:
  name = 'sim10k_{}'.format(split)
  __sets[name] = (lambda split=split : sim10k(split))
    

# Set up coco_2014_<split>
for year in ['2014']:
  for split in ['train', 'val', 'minival', 'valminusminival', 'trainval']:
    name = 'coco_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: coco(split, year))

# Set up coco_2014_cap_<split>
for year in ['2014']:
  for split in ['train', 'val', 'capval', 'valminuscapval', 'trainval']:
    name = 'coco_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: coco(split, year))

# Set up coco_2015_<split>
for year in ['2015']:
  for split in ['test', 'test-dev']:
    name = 'coco_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: coco(split, year))

# Set up vg_<split>
# for version in ['1600-400-20']:
#     for split in ['minitrain', 'train', 'minival', 'val', 'test']:
#         name = 'vg_{}_{}'.format(version,split)
#         __sets[name] = (lambda split=split, version=version: vg(version, split))
for version in ['150-50-20', '150-50-50', '500-150-80', '750-250-150', '1750-700-450', '1600-400-20']:
    for split in ['minitrain', 'smalltrain', 'train', 'minival', 'smallval', 'val', 'test']:
        name = 'vg_{}_{}'.format(version,split)
        __sets[name] = (lambda split=split, version=version: vg(version, split))

for split in ['train', 'val']:
    name = 'bdd100k_{}'.format(split)
    __sets[name] = (lambda split=split: bdd100k(split))        
        
for split in ['train', 'val']:
    name = 'kitti_{}'.format(split)
    __sets[name] = (lambda split=split: kitti(split))
#    tgt_name = 'kitti_{}_tgt'.format(split)
#    __sets[tgt_name] = (lambda split=split, num_shot=num_shot: kitti(split, num_shot))        
        
# set up image net.
for split in ['train', 'val', 'val1', 'val2', 'test']:
    name = 'imagenet_{}'.format(split)
    devkit_path = 'data/imagenet/ILSVRC/devkit'
    data_path = 'data/imagenet/ILSVRC'
    __sets[name] = (lambda split=split, devkit_path=devkit_path, data_path=data_path: imagenet(split,devkit_path,data_path))

for split in ['train', 'val', 'test']:
    name = 'wider_face_{}'.format(split)
    __sets[name] = (lambda split=split: wider_face(split))

for split in ['train', 'val', 'test','train_Bus','train_Pathway','train_Doorway','train_Room','train_Staircase']:
    name = 'MI3_{}'.format(split)
    __sets[name] = (lambda split=split: mi3(split))

for split in['train', 'test','campus', 'downtown','road','fake','train_cr','train_cd','train_rd']:
    name = 'KAIST_{}'.format(split)
    __sets[name] = (lambda split=split: kaist(split))

   

def get_imdb(name):
  """Get an imdb (image database) by name."""
  if name not in __sets:
    raise KeyError('Unknown dataset: {}'.format(name))
  return __sets[name]()


def list_imdbs():
  """List all registered imdbs."""
  return list(__sets.keys())
