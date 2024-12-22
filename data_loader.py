import tensorflow as tf
import pandas as pd

"""

This module handles data loading and augmentation.

"""

def load_data(image_size, batch_size, seed, validation_split):
    data_dir = '../data/train_zscore'
    
    labels_df = pd.read_csv('../data/train_labels.csv')
    labels_df['filename'] = labels_df['image'].apply(lambda x: f"{data_dir}/{x}.jpeg")
    labels_df['level'] = labels_df['level'].astype(str)

    train_data_augmentation = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.8, 1.2],
        zoom_range=[1.15, 0.87],
        validation_split=validation_split,
        rotation_range=360,
        fill_mode='constant',
    )

    # Define data augmentation for validation (only rescaling)
    val_data_augmentation = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        validation_split=validation_split,
    )

    # Load training dataset with augmentation
    train_ds = train_data_augmentation.flow_from_dataframe(
        dataframe=labels_df,
        x_col='image',
        y_col='level',
        target_size=image_size,
        batch_size=batch_size,
        subset='training',
        seed=seed
    )

    # Load validation dataset without augmentation
    val_ds = val_data_augmentation.flow_from_dataframe(
        dataframe=labels_df,
        x_col='image',
        y_col='level',
        target_size=image_size,
        batch_size=batch_size,
        subset='validation',
        seed=seed
    )
    
    return train_ds, val_ds