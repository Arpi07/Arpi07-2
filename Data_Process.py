import os
import pickle
import imageio
import numpy as np
import multiprocessing
import sys

from scipy.io import loadmat

import Dataset
import datasets_utils
import training_utils
import Get_Dataset
import Get_Dataset_utils


def matlab_mat_to_numpy():
    """Conversion of colour infoormation to an nd array"""
    colors = loadmat(Dataset.COLORMAP_ORIG_PATH)['colors']
    background_colour = np.zeros([1, 3], dtype=np.uint8)
    colors = np.concatenate([background_colour, colors], axis=0)
    np.save(Dataset.COLORMAP_PATH[:-4], colors)


def parse_object_info():
    """text infoormation parsing """
    is_header = True
    # integer_id 
    meta_data = {}
    with open(Dataset.ORIG_OBJECT_INFO_PATH, 'r') as text_file:
        for row in text_file:
            if is_header:
                is_header = False
                continue
            else:
                info = row.split()
                id_ = int(info[0])
                ratio = float(info[1])
                train = int(info[2])
                val = int(info[3])
                names = info[4]
                meta_data[id_] = {
                    'ratio': ratio,
                    'train': train,
                    'val': val,
                    'names': names,}
    with open(Dataset.OBJECT_INFO_PATH, 'wb') as pfile:
        pickle.dump(meta_data, pfile)


##################################################################
# Construction of edge maps
#################################################################


def edge_path_from_label_path(label_path):
    label_name = os.path.basename(label_path)
    label_dir = os.path.dirname(label_path)
    edge_name = Dataset.EDGE_PREFIX + label_name
    edge_path = os.path.join(label_dir, edge_name)
    return edge_path


def label_path_to_edge_saved(label_path):
    edge_path = edge_path_from_label_path(label_path)
    label = imageio.imread(label_path)
    edge = training_utils.flat_label_to_edge_label(label, Dataset.N_CLASSES)
    imageio.imsave(edge_path, edge)

#labelling of boundary edge mask
def create_edge_labels():
    pool = multiprocessing.Pool(10)
    train_labels = [os.path.join(Dataset.TRAINING_ANNOTATION_DIR, x) for x in os.listdir(
        Dataset.TRAINING_ANNOTATION_DIR)]
    val_labels = [os.path.join(Dataset.VALIDATION_ANNOTATION_DIR, x) for x in os.listdir(
        Dataset.VALIDATION_ANNOTATION_DIR)]

    num_train = len(train_labels)
    print('creating training edge maps')
    for i, _ in enumerate(pool.imap_unordered(label_path_to_edge_saved, train_labels), 1):
        sys.stderr.write('\rdone {0:%}'.format(i / num_train))

    num_val = len(val_labels)
    print('creating val edge maps')
    for i, _ in enumerate(pool.imap_unordered(label_path_to_edge_saved, val_labels), 1):
        sys.stderr.write('\rdone {0:%}'.format(i / num_val))

 # download the data and conversion of colour information  to numpy array

def get_dataset():
    # Convert text-file of object info to python dictionary
    parse_object_info()
    #conversion of colour information into numpy array
    matlab_mat_to_numpy()
    # Labelling of boundary edge mask
    create_edge_labels()
    # Dataset dictionary
   datasets_utils.list_files(Dataset.DATA_DIR)


if __name__ == '__main__':
    pass
