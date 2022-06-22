import numpy as np
import pickle
import argparse
import _init_paths

from datasets.voc_eval import voc_eval
from datasets.factory import get_imdb

import os


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='evaluate results')
    parser.add_argument('--dataset', dest='dataset',
                      help='training dataset',
                      default='cityscape', type=str)
    parser.add_argument("--output_folder", type=str, default="output_folder", help="path to output folder")

    args = parser.parse_args()
    return args
if __name__ == '__main__':

    args = parse_args()

    print('Called with args:')
    print(args)
    
    if args.dataset == "pascal_voc":
        args.imdb_name = "voc_2007_test"

    elif args.dataset == "cityscape":
        print('loading our dataset...........')   
        args.imdb_name="cityscape_2007_test"     

    elif args.dataset == "foggy_cityscape":        
        args.imdb_name = "foggy_cityscape_2007_test"

    elif args.dataset == "bdd100k":
        args.imdb_name = "bdd100k_val"
        
    elif args.dataset == "kitti":
        args.imdb_name = "kitti_val"
     
    
    custermized_name = args.output_folder

    imdb = get_imdb(args.imdb_name)
    
    imdb.competition_mode(on=True)

    data_path = imdb._data_path
    
    print(imdb.config['use_salt'])

    classes = imdb._classes
    use_07_metric = True if int(imdb._year) < 2010 else False
    print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
    annopath = os.path.join(imdb._data_path, 'Annotations', '{:s}.xml')
    imagesetfile = os.path.join(  imdb._data_path,'ImageSets', 'Main', imdb._image_set + '.txt')
    cachedir = os.path.join(imdb._devkit_path, 'annotations_cache')
    aps = []    
    for i, cls in enumerate(classes):
        if cls == '__background__':
            continue
        filename = imdb._get_voc_results_file_template(custermized_name).format(cls)   
        
        
        rec, prec, ap = voc_eval(
            filename, annopath, imagesetfile, cls, cachedir, ovthresh=0.5, use_07_metric=use_07_metric)
        aps += [ap]
        print(ap)
        print('AP for {} = {:.4f}'.format(cls, ap))
         
    print('Mean AP = {:.4f}'.format(np.mean(aps)))
    print('~~~~~~~~')
    print('Results:')
    for ap in aps:
        print('{:.3f}'.format(ap))
    print('{:.3f}'.format(np.mean(aps)))
