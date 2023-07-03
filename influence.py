
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

                        pbar.update(len(labels))
                        # print(inf_0.shape, inf_1.shape, inf_2.shape, inf_3.shape, inf_4.shape, inf_5.shape)

            except tf.errors.OutOfRangeError:
                pass
            
            # Save I_layer
            with open(I_mat_dirpath + 'I_%s.json' % layer, 'w') as f:
                json.dump(I_layer, f, indent=2)

            end = time.time()
            print(end - start)
            print(progress_counter)
            print(progress_counter * batch)

    # Generate chains
    # pred_class = 270
    # channels = [1, 120]
    # generate_save_chain(pred_class, all_layers, I_mat_dirpath, channels)


def parse_args():
    '''
    Parse arguments and pass as arguments object
    '''

    parser = argparse.ArgumentParser('Summit')

    parser.add_argument('--layer', type=str, default='mixed3a',
                        help='name of layer to generate I matrix')

    parser.add_argument('--gpu', type=int, default=0,
                        help='gpu cuda visible device')

    parser.add_argument('--batch', type=int, default=500,
                        help='batch size for loading images')

    return parser.parse_args()


def _parse_function(example_proto, image_size=224):
    '''
    Parse datasets
    '''
    
    def _bytes_feature(value):
        return tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[value]))

    def _int64_feature(value):
        return tf.train.Feature(
            int64_list=tf.train.Int64List(value=[value]))
    
    feature_set = {
        'image/filename': tf.FixedLenFeature([], tf.string),
        'image/encoded': tf.FixedLenFeature([], tf.string),
        'image/height': tf.FixedLenFeature([], tf.int64),
        'image/width': tf.FixedLenFeature([], tf.int64),
        'image/channels': tf.FixedLenFeature([], tf.int64),
        'image/class/label': tf.FixedLenFeature([], tf.int64),
        'image/class/synset': tf.FixedLenFeature([], tf.string)}
  
    parsed_features = tf.parse_single_example(example_proto, feature_set)
    label = parsed_features['image/class/label']
    synset = parsed_features['image/class/synset']
    image = parsed_features['image/encoded']
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize_images(image, tf.constant([image_size, image_size]))
    
    return image, label, synset


def get_weight_tensors(layer):
    '''
    Get weight tensors for given layer
    '''
    # Get weight tensors
    t_w0 = tf.get_default_graph().get_tensor_by_name('import/%s_1x1_w:0' % layer)
    t_w1 = tf.get_default_graph().get_tensor_by_name('import/%s_3x3_bottleneck_w:0' % layer)
    t_w2 = tf.get_default_graph().get_tensor_by_name('import/%s_3x3_w:0' % layer)
    t_w3 = tf.get_default_graph().get_tensor_by_name('import/%s_5x5_bottleneck_w:0' % layer)
    t_w4 = tf.get_default_graph().get_tensor_by_name('import/%s_5x5_w:0' % layer)
    t_w5 = tf.get_default_graph().get_tensor_by_name('import/%s_pool_reduce_w:0' % layer)
    
    return t_w0, t_w1, t_w2, t_w3, t_w4, t_w5


def get_intermediate_layer_tensors(prev_layer, layer):
    # Get intermediate layer tensors
    t_a0 = tf.get_default_graph().get_tensor_by_name('import/%s:0' % prev_layer)
    t_a1 = tf.get_default_graph().get_tensor_by_name('import/%s_3x3_bottleneck:0' % layer)
    t_a2 = tf.get_default_graph().get_tensor_by_name('import/%s_5x5_bottleneck:0' % layer)
    return t_a0, t_a1, t_a2


def get_layers(graph_nodes):
    '''
    Get all layers
    * input
        - graph_nodes: tensorflow graph nodes
    * output
        - layers: list of all layers
    '''
    layers = []
    for n in graph_nodes:
        node_name = n.name
        if node_name[-2:] == '_w':
            layer = node_name.split('_')[0]
            if layer not in layers:
                layers.append(layer)
    return layers


def get_channel_sizes(layer, weight_nodes):
    '''
    Get channel sizes
    * input
        - layer: the name of layer
        - weight_nodes: tensorflow nodes for all filters
    * output
        - channel_sizes: list of channel size for all pre-concatenated blocks
    '''
    
    channel_sizes = [get_shape_of_node(n)[0] for n in weight_nodes if layer in n.name and '_b' == n.name[-2:] and 'bottleneck' not in n.name]
    return channel_sizes


def get_shape_of_node(n):
    '''
    Get the shape of the tensorflow node
    * input
        - n: tensorflow node
    * output
        - tensor_shape: shape of n
    '''
    dims = n.attr['value'].tensor.tensor_shape.dim
    tensor_shape = [d.size for d in dims]
    return tensor_shape


def get_num_channel(layer, weight_nodes):
    '''
    Get the number of channels in the layer
    * input
        - layer: the name of layer (e.g. 'mixed5a' for normal layer, 'mixed5a_1' for 1st branch after mixed_5a layer)
        - is_branch: whether the layer is in a branch
    * output
        - num_channel: the number of channel
    '''
    
    is_branch = '_' in layer
    
    if is_branch:
        layer_name = layer[:-2]
        branch = int(layer[-1])
        branch_weights = [n for n in weight_nodes if layer_name in n.name and 'bottleneck_w' in n.name]
        branch_weight = branch_weights[branch - 1]
        num_channel = get_shape_of_node(branch_weight)[-1]
        return num_channel
    
    else:
        num_channel = np.sum(get_channel_sizes(layer, weight_nodes))
        return num_channel


def get_prev_layer(layers, layer):
    '''
    Get previous layer
    * input
        - layers: list of all layers
        - layer: the name of a layer
    * output
        - prev_layer: the name of a previuos layer
    '''
    prev_layer = layers[layers.index(layer) - 1]
    return prev_layer