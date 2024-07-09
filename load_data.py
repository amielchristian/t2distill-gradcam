import tensorflow as tf
import pandas as pd

image_size = (384, 384)
batch_size = 8

def load_data():
    train_dir = '../data/train'
    test_dir = '../data/test'

    train_labels = list(pd.read_csv('../data/train_labels.csv').sort_values('image').iloc[:, 1])
    test_labels = list(pd.read_csv('../data/test_labels.csv').sort_values('image').iloc[:, 1])

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir,
        seed=1337,
        image_size=image_size,
        batch_size=batch_size,
        labels=train_labels,
        label_mode='categorical'
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        test_dir,
        seed=1337,
        image_size=image_size,
        batch_size=batch_size,
        labels=test_labels,
        label_mode='categorical'
    )

    return train_ds, val_ds