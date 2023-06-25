
'''
Import packages
'''
import glob
import json
import tqdm
import time
import argparse
import numpy as np
import tensorflow as tf
from collections import defaultdict
import lucid.optvis.render as render
import lucid.modelzoo.vision_models as models
from keras.applications.inception_v3 import preprocess_input
import os


'''
Main function
'''
def main():
    '''
    Parse the arguments
    '''
    args = parse_args()
    layer = args.layer
    gpu = args.gpu
    batch = args.batch

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    
    googlenet = models.InceptionV1()
    googlenet.load_graphdef()
    nodes = googlenet.graph_def.node

    # filenames = glob.glob('/media/fred/strawberry/imagenet-tf-records/*')
    # filenames = glob.glob('test-images/imagenet-tf-records/*')
    filenames = glob.glob('/raid/hpark407/imagenet-tf-records/*')
    I_mat_dirpath = '/raid/fhohman3/I-mat/'
    # chain_dirpath = './chain/'

    num_class = 1000
    all_layers = get_layers(nodes)
    mixed_layers = [layer for layer in all_layers if 'mixed' in layer]

    layers = {
        'mixed3a': 256,