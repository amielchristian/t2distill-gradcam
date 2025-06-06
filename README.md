# T2Distill-GradCAM: Learning Vision Transformers using Two-Teacher Knowledge Distillation with Autoencoders and Gradient-weighted Class Activation Maps in Diabetic Retinopathy Classification

This repository contains the code for the study T2Distill-GradCAM: Learning Vision Transformers using Two-Teacher Knowledge Distillation with Autoencoders and Gradient-weighted Class Activation Maps in Diabetic Retinopathy Classification, accepted for publication and presentation at the 25th International Conference on Computing and Artificial Intelligence, authored by myself, [Patricia Denise Poblete](https://github.com/PatriciaDeniseP), and [Ann Clarisse Salazar](https://github.com/ClarisseSalazar).

The implementation of the Vision Transformer architecture in this study is based on that of [faustomorales](https://github.com/faustomorales/vit-keras). The preprocessing techniques are also based in part on [sveitser](https://github.com/sveitser/kaggle_diabetic/blob/master/convert.py)'s implementation.

Note that this repository does not contain the EyePACS fundus image dataset, which can be accessed through its [Kaggle page](https://www.kaggle.com/competitions/diabetic-retinopathy-detection). The labels are already provided in this repository, although it must be noted that the test labels were obtained from an [unofficial source](https://www.kaggle.com/datasets/c7934597/resized-2015-2019-diabetic-retinopathy-detection).

#### Setup
1. Install required packages with `pip install -r requirements.txt`.
2. Obtain images and place in the `data` folder under `train` and `test`. The `.csv` files containing the labels are preprovided.
3. 