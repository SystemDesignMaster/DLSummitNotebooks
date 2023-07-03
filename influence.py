
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
        'mixed3b': 480,
        'mixed4a': 508,
        'mixed4b': 512,
        'mixed4c': 512,
        'mixed4d': 528,
        'mixed4e': 832,
        'mixed5a': 832,
        'mixed5b': 1024
    }

    layer_fragment_sizes = {layer: get_channel_sizes(layer, nodes) for layer in mixed_layers}
    weight_sizes = get_weight_sizes(nodes, all_layers)
    act_sizes = get_act_sizes(weight_sizes, mixed_layers)

    k = 5
    # chain_k = 3
    mixed_layer = layer.split('_')[0]
    prev_layer = get_prev_layer(all_layers, mixed_layer)

    a_sz = act_sizes[mixed_layer]
    f_sz = layer_fragment_sizes[mixed_layer]

    frag_sz = [f_sz[0], f_sz[1], f_sz[2], f_sz[3], a_sz[1], a_sz[2]]

    outlier_nodes = ['mixed3a-67', 'mixed3a-190', 'mixed3b-390', 'mixed3b-399', 'mixed3b-412']
    outlier_nodes_idx = [int(n.split('-')[1]) for n in outlier_nodes if layer in n]

    # Get top impactful previous neurons and generate I-matrices
    # Get layer info
    is_mixed = '_' not in layer
    branch = None if is_mixed else int(layer.split('_')[-1])

    # Initialize I
    num_channel = layers[layer] if is_mixed else act_sizes[layer[:-2]][branch]
    I_layer = gen_empty_I(num_class, num_channel)

    # Run
    with tf.Graph().as_default():
        # Get dataset
        dataset = tf.data.TFRecordDataset(filenames)
        dataset = dataset.map(_parse_function)
        dataset = dataset.map(lambda img, lab, syn: (preprocess_input(img), lab, syn))
        dataset = dataset.batch(batch)

        # Iterate tf-records
        iterator = dataset.make_one_shot_iterator()
        t_preprocessed_images, t_labels, t_synsets = iterator.get_next()

        # Import googlenet
        T = render.import_model(googlenet, t_preprocessed_images, None)

        # Get weight tensors
        t_w0, t_w1, t_w2, t_w3, t_w4, t_w5 = get_weight_tensors(mixed_layer)

        # Get intermediate layer tensors
        t_a0, t_a1, t_a2 = get_intermediate_layer_tensors(prev_layer, mixed_layer)
        
        # Define intermediate conv output tensors
        t_inf_0 = get_infs(t_a0, t_w0)
        t_inf_1 = get_infs(t_a1, t_w2)
        t_inf_2 = get_infs(t_a2, t_w4)
        t_inf_3 = get_infs(t_a0, t_w5)
        t_inf_4 = get_infs(t_a0, t_w1)
        t_inf_5 = get_infs(t_a0, t_w3)

        # Run with batch
        progress_counter = 0
        with tf.Session() as sess:
            start = time.time()

            try:
                with tqdm.tqdm(total=1281167, unit='imgs') as pbar:

                    while(True):
                        progress_counter += 1
                        
                        # Run the session
                        if is_mixed:
                            labels, inf_0, inf_1, inf_2, inf_3 = sess.run([t_labels, t_inf_0, t_inf_1, t_inf_2, t_inf_3])

                        elif branch == 1:
                            labels, inf_4 = sess.run([t_labels, t_inf_4])

                        elif branch == 2:
                            labels, inf_5 = sess.run([t_labels, t_inf_5])
                        
                        # Add up the counts
                        if is_mixed:
                            channel = 0
                            for frag, inf in enumerate([inf_0, inf_1, inf_2, inf_3]):
                                channel = update_I(layer, inf, channel, I_layer, labels, frag_sz[frag], k, outlier_nodes_idx)

                        elif branch == 1:
                            update_I(layer, inf_4, 0, I_layer, labels, frag_sz[4], k, outlier_nodes_idx)

                        elif branch == 2:
                            update_I(layer, inf_5, 0, I_layer, labels, frag_sz[5], k, outlier_nodes_idx)