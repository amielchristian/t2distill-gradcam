"""
This is the baseline ViT model, known as 384_B_32 in the paper.
According to the paper:
    - The model contains 12 layers, 768 hidden units, and 12 heads.
    - The model is pre-trained on the ImageNet-21k dataset and fine-tuned on the EyePACS dataset (try reading this later: https://keras.io/examples/vision/image_classification_using_global_context_vision_transformer/).
    - Loss weights are set to [1.0,1.0,1.0,2.0,2.0] according to the data distribution so that the model is fully trained.
    - Adam is the optimizer with a learning rate of 2e-4.
    - Image size is set to 384x384 with a patch size of 32x32.
"""
import tensorflow as tf
from load_data import load_data
from keras_vit import vit

train_ds, val_ds = load_data()
model = vit.ViT_B32(
    weights='imagenet21k',
    image_size = 256,
    num_classes=5
    )
model.summary()