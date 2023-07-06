import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
from tensorflow.keras import layers

# Tuple of NumPy arrays: 
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()

print("x_train type: ", type(x_train))
# ouputs: x_train type: <class 'numpy.ndarray'>
print("y_train type: ", type(y_train))
# ouputs: y_train type: <class 'numpy.ndarray'>

print(f"x_train shape: {tf.shape(x_train)}")
# outputs: x_train shape: [50000    32    32     3]
# where [0] is batch_size, [1] and [2] are image size, [3] is num_channels

print(f"y_train shape: {tf.shape(y_train)}")
# outputs: y_train type:  <class 'numpy.ndarray'>

# loading custom dataset
image_size = (180, 180)
batch_size = 128

# If label_mode is None, it yields float32 tensors of shape (batch_size, image_size[0], image_size[1], num_channels),
train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
    "train",
    validation_split=0.2,
    subset="both",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)

print("train_ds type: ", type(train_ds))
# outputs: train_ds type:  <class 'tensorflow.python.data.ops.batch_op._BatchDataset'>

# setting label_mode yields a tuple (images, labels), where images has shape (batch_size, image_size[0], image_size[1], num_channels)
# setting label_mode as int, the labels are an int32 tensor of shape (batch_size,)
train_ds2, val_ds2 = tf.keras.utils.image_dataset_from_directory(
    "train",
    validation_split=0.2,
    subset="both",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
    label_mode="int",
)

print("train_ds2 type: ", type(train_ds2))
# outputs: train_ds type:  <class 'tensorflow.python.data.ops.batch_op._BatchDataset'>, note, doesnt change the type

(train, val, test), metadata = tfds.load(
    'tf_flowers',
    split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
    with_info=True,
    # as_supervised=True returns a 2-tuple structure (input, label)
    as_supervised=True,
    # batch_size=-1 will return the dataset as tf.Tensors
    batch_size=-1,
)

print("type of train: ", type(train))
# outputs: type of train:  <class 'tuple'>

(train2, val2, test2), metadata = tfds.load(
    'tf_flowers',
    split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
    with_info=True,
    # as_supervised=False returns a dictionary with the features
    as_supervised=False,
    # batch_size=-1 will return the dataset as tf.Tensors
    batch_size=-1,
)

print("type of train2: ", type(train2))
# outputs: type of train2:  <class 'dict'>

(train3, val3, test3), metadata = tfds.load(
    'tf_flowers',
    split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
    with_info=True,
    # as_supervised=True returns a 2-tuple structure (input, label)
    as_supervised=True,
    # batch_size=-1 will return the dataset as tf.Tensors
    batch_size=1,
)

print("type of train3: ", type(train3))
# outputs: type of train3:  <class 'tensorflow.python.data.ops.prefetch_op._PrefetchDataset'>

(train4, val4, test4), metadata = tfds.load(
    'tf_flowers',
    split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
    with_info=True,
    # as_supervised=True returns a 2-tuple structure (input, label)
    as_supervised=False,
    # batch_size=-1 will return the dataset as tf.Tensors
    batch_size=1,
)

print("type of train4: ", type(train4))
# outputs: type of train4:  <class 'tensorflow.python.data.ops.prefetch_op._PrefetchDataset'>

data_augmentation = keras.Sequential(
    [
        layers.Normalization(),
        layers.Resizing(image_size, image_size),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(factor=0.02),
        layers.RandomZoom(
            height_factor=0.2, width_factor=0.2
        ),
    ],
    name="data_augmentation",
)
# Compute the mean and the variance of the training data for normalization.
# Note, does not work for any data type other then <class 'numpy.ndarray'> and a shape of shape: [50000    32    32     3] # where [0] is batch_size, [1] and [2] are image size, [3] is num_channels
data_augmentation.layers[0].adapt(x_train)


