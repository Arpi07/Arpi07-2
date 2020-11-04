import tensorflow as tf

from Attention_cnn import Data
import Attention_cnn.Data.Get_Dataset
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

from Attention_cnn import ACNN
from Attention_cnn import Trainer


#dataset parameters
batch_size = 8
network_input_h = network_input_w = 256
max_crop_downsample = 0.95
colour_aug_factor = 0.2

# loss hyperparams
l1 = 17.
l2 = 1.

loss_weights = [l1, l2]

# optimiser
n_train_images = 1000
n_steps_in_epoch = n_train_images // batch_size
momentum = 0.9
learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
    0.001,
    n_steps_in_epoch ,
    0.000001)
optimiser = tf.keras.optimizers.adam(learning_rate=learning_rate_fn, epsilon=1e-05)
#Dataset Loader....
dataset_loader = Attention_cnn.Data.Get_Dataset.SceneParsing(
    batch_size,
    network_input_h,
    network_input_w,
    max_crop_downsample,
    colour_aug_factor,
    build_for_keras=False,
    debug=False)

# Model building..................
model = ACNN(n_classes=Attention_cnn.Data.N_CLASSES)

# train
trainer = Trainer(
    model,
    dataset_loader.build_training_dataset(),
    dataset_loader.build_validation_dataset(),
    epochs=2000,
    optimiser=optimiser,
    log_dir='logsBalloon',
    model_dir='logsBalloon/model',
    loss_weights=loss_weights,
    accumulation_iterations=2,)
trainer.train_loop()









