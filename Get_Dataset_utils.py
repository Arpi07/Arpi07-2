import random
import numpy as np
import imageio
import os

from Attention_cnn import Data


def build_legend_info(object_ids):
    names = []
    colours = []
    for object_id in object_ids:
        if object_id == 0:
            names.append('other')
        else:
            object_info = Dataset.OBJECT_INFO[object_id]
            names.append(object_info['names'])
        colours.append(Dataset.COLOURS[object_id])
    return names, colours


def flat_label_to_plottable(label):
    coloured_image = Dataset.COLOURS[label]
    objects_present = np.unique(label)
    names, colours = build_legend_info(objects_present)
    return coloured_image, (names, colours)


def paths_from_example_id(example_id):
    image_path = os.path.join(Dataset.TRAINING_IM_DIR, example_id + '.jpg')
    label_path = os.path.join(Dataset.TRAINING_ANNOTATION_DIR, example_id + '.png')
    return image_path, label_path


def example_paths_from_single_path(single_path):
    example_id = os.path.basename(single_path)[:-4]
    return paths_from_example_id(example_id)


def get_random_example_paths():
    im_path = random.choice(os.listdir(Dataset.TRAINING_IM_DIR))
    return example_paths_from_single_path(im_path)


def get_random_example():
    im_p, label_p = get_random_example_paths()
    return imageio.imread(im_p), imageio.imread(label_p)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    im, label = get_random_example()
    plt.subplot(211)
    plt.imshow(im)
    plt.subplot(212)
    plt.imshow(label)
    plt.show()
