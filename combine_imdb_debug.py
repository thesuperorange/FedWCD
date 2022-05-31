import _init_paths
import pickle
from roi_data_layer.roidb import prepare_roidb,filter_roidb,rank_roidb_ratio
from model.utils.config import cfg
from datasets.factory import get_imdb

def get_training_roidb(imdb):
    """Returns a roidb (Region of Interest database) for use in training."""
    
    print('Appending horizontally-flipped training examples...')
    imdb.append_flipped_images()
    print('done')

    print('Preparing training data...')

    prepare_roidb(imdb)
    #ratio_index = rank_roidb_ratio(imdb)
    print('done')

    return imdb.roidb
  
def get_roidb(imdb_name):
    imdb = get_imdb(imdb_name)
    print('Loaded dataset `{:s}` for training'.format(imdb.name))
    imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)
    print('Set proposal method: {:s}'.format(cfg.TRAIN.PROPOSAL_METHOD))
    roidb = get_training_roidb(imdb)
    return roidb


imdb_names = 'kitti_train_small'
roidb = get_roidb(imdb_names) 
print("roidb len={}".format(len(roidb)))
#print(roidb[59])
# if len(roidbs) > 1:
#     for r in roidbs[1:]:
#         roidb.extend(r)
#     tmp = get_imdb(imdb_names.split('+')[1])
#     imdb = datasets.imdb.imdb(imdb_names, tmp.classes)
# else:
#     imdb = get_imdb(imdb_names)


roidb = filter_roidb(roidb)
print("[after filter] roidb len={}".format(len(roidb)))

cache_file = 'kitti_train_small.pkl'
with open(cache_file, 'wb') as fid:
    pickle.dump(roidb, fid, pickle.HIGHEST_PROTOCOL)
print('wrote gt roidb to {}'.format(cache_file))
