import tensorflow as tf
import os


import Dataset

class Get_Dataset(Dataset):

    def __init__(
            self,
            batch_size,
            network_input_h,
            network_input_w,
            max_crop_downsample,
            colour_aug_factor,
            debug,):
        super(Get_Dataset, self).__init__(
            batch_size,
            network_input_h,
            network_input_w,
            max_crop_downsample,
            colour_aug_factor,
            debug,
            val_batch_size=1)

    def get_paths(self, train):
        folders = Dataset.get_dataset.TRAINING_DIRS if train else Dataset.get_dataset.VALIDATION_DIRS
        image_dir = folders[Dataset.get_dataset.IMAGES]
        label_dir = folders[Dataset.get_dataset.LABELS]

        example_ids = []
        for x in os.listdir(image_dir):
            example_ids.append(x[:-4])

        image_paths = [os.path.join(image_dir, x + '.jpg') for x in example_ids]
        label_paths = [os.path.join(label_dir, x + '.png') for x in example_ids]
        edge_paths = [os.path.join(label_dir, Dataset.get_dataset.EDGE_PREFIX + x + '.png') for x in example_ids]

        return image_paths, label_paths, edge_paths

    def flat_to_one_hot(self, labels, edges):
        labels = tf.one_hot(labels[..., 0], Dataset.get_dataset.N_CLASSES)
        edges = tf.one_hot(edges[..., 0], 2)
        return labels, edges

    def build_validation_dataset(self):
        """
        val dataset:
            - full size images
            - no augmentations
            - fixed batch size of VAL_BATCH (=2)
        """
        # get dataset of tensors (im, label, edge)
        dataset = self.get_raw_tensor_dataset(train=False)
        dataset = dataset.map(self.resize_images)
        # batch process
        dataset = dataset.batch(self.val_batch_size, drop_remainder=False)
        dataset = dataset.map(self.process_validation_batch, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        if self.debug:
            dataset = dataset.take(1)
        return dataset


if __name__ == '__main__':
    pass
