import _init_paths
from roi_data_layer.roidb import combined_roidb
import sys

print (sys.argv[1])
imdb_name = sys.argv[1] #'MI3_train_Doorway'
imdb, roidb, ratio_list, ratio_index = combined_roidb(imdb_name)
