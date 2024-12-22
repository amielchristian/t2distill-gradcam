import tensorflow as tf
from tensorflow.keras.layers import Layer, Lambda
import keras
from keras.models import Model
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np

# helper class
class OperationsLayer(tf.keras.layers.Layer):
    def __init__(self, filters, **kwargs):
        super(OperationsLayer, self).__init__(**kwargs)
        self.conv1 = tf.keras.layers.Conv2D(filters, kernel_size=1, padding='same') # Pointwise Convolution
        self.norm1 = tf.keras.layers.LayerNormalization() # Layer Normalization
        self.activation1 = tf.keras.layers.Activation('relu') # ReLU activation

    def call(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation1(x)
        return x

class Distiller(keras.Model):
    def __init__(self, student, teacher):
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.teacher_output_weights = self.teacher.layers[-1].get_weights()[0]
        self.jet_colors = tf.constant(
            np.array([plt.get_cmap("jet")(i)[:3] for i in range(256)]),
            dtype=tf.float32
        )

    @property
    def metrics(self):
        metrics = super().metrics

        return metrics

    def compile(
        self,
        optimizer,
        metrics,
        student_loss_fn,
        logit_loss_fn,
        feature_loss_fn,
        alpha=0.1,
        beta=0.1,
        temperature=3,
    ):
        super(Distiller, self).compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.logit_loss_fn = logit_loss_fn
        self.feature_loss_fn = feature_loss_fn
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature


    def create_combined_model(self, model, output_layers):
        return Model(inputs=model.input, outputs=[model.output] + output_layers)

    def spatial_alignment(self, teacher_features, student_features):
        aligned_features = []

        for teacher_feat, student_feat_list in zip(teacher_features, student_features):
            # Select the main feature tensor (the first element in each student feature list)
            # has the same shape as the teacher feature map
            student_feat = student_feat_list[0]

            # Step 0: Remove the class token
            student_feat = student_feat[:, 1:]

            # Step 1: Reshape student tokens to match CNN features
            # in short, (None, 196, 768) -> (None, 14, 14, 768)
            sqrt = int(np.sqrt(student_feat.shape[1]))
            student_resized = tf.reshape(student_feat, (-1, sqrt, sqrt, student_feat.shape[-1]))

            # Step 2: Perform bilinear interpolation to upsample student features
            h, w = teacher_feat.shape[1:3]
            student_interpolated = tf.image.resize(student_resized, (h, w))

            # Steps 3-4: Perform operations layer (pointwise convolution, layer normalization, ReLU activation)
            student_activated = OperationsLayer(teacher_feat.shape[-1])(student_interpolated)

            aligned_features.append((teacher_feat, student_activated))

        return aligned_features

    def train_step(self, data):
        x, y = data

        # Teacher model pass
        teacher_conv_layers = [layer.output for layer in self.teacher.layers if isinstance(layer, tf.keras.layers.Conv2D)]
        teacher_model = self.create_combined_model(self.teacher, [teacher_conv_layers[-1]])
        teacher_logits, *teacher_features = teacher_model(x, training=False)

        with tf.GradientTape() as tape:
            student_model = self.create_combined_model(self.student, [layer.output for layer in self.student.layers if 'Transformer-encoderblock' in layer.name][-(spatial_alignment_layers):])
            student_logits, *student_features = student_model(x, training=True)

            student_loss = tf.reduce_mean(self.student_loss_fn(y, tf.nn.softmax(student_logits)))

            logit_loss = tf.reduce_mean(self.logit_loss_fn(
                    tf.nn.softmax(teacher_logits / self.temperature, axis=-1),
                    tf.nn.softmax(student_logits / self.temperature, axis=-1)
                ) * self.temperature ** 2)

            aligned_features = self.spatial_alignment(teacher_features, student_features)
            feature_loss = tf.reduce_mean(sum(
                [self.feature_loss_fn(
                    tf.nn.l2_normalize(teacher),
                    tf.nn.l2_normalize(student),
                    ) + 1
                for teacher, student in aligned_features]))

            total_loss = self.alpha * student_loss + (1 - self.alpha) * logit_loss + self.beta * feature_loss

        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        other_metrics = []
        for metric in self.metrics:
            metric.update_state(y, student_logits)
            if metric.name == "compile_metrics":
                other_metrics.append(metric.result())

        updated_metrics = {"total_loss": total_loss,
                           "student_loss": student_loss,
                           "logit_loss": logit_loss,
                           "feature_loss": feature_loss}
        for metric in other_metrics:
            updated_metrics.update(metric)
        return updated_metrics

    def test_step(self, data):
        x, y = data

        # Forward passes
        teacher_model = self.create_combined_model(self.teacher, [layer.output for layer in self.teacher.layers])
        teacher_logits, *teacher_features = teacher_model(x, training=False)
        student_model = self.create_combined_model(self.student, [layer.output for layer in self.student.layers if 'Transformer-encoderblock' in layer.name][-(spatial_alignment_layers):])
        student_logits, *student_features = student_model(x, training=False)

        # Compute student loss
        student_loss = tf.reduce_mean(self.student_loss_fn(y, tf.nn.softmax(student_logits)))

        logit_loss = tf.reduce_mean(self.logit_loss_fn(
                    tf.nn.softmax(teacher_logits / self.temperature, axis=-1),
                    tf.nn.softmax(student_logits / self.temperature, axis=-1)
                ) * self.temperature ** 2)

        aligned_features = self.spatial_alignment(teacher_features, student_features)
        feature_loss = tf.reduce_mean(sum(
            [self.feature_loss_fn(
                tf.nn.l2_normalize(teacher),
                tf.nn.l2_normalize(student),
                ) + 1
            for teacher, student in aligned_features]))
        
        # Combine losses (with weighting)
        # alpha times the student ViT's loss
        # plus (1-alpha) times the logit loss (logit loss should also be cross-entropy)
        # plus beta times the feature-based distillation loss
        total_loss = self.alpha * student_loss + (1 - self.alpha) * logit_loss + self.beta * feature_loss

        other_metrics = []
        for metric in self.metrics:
            metric.update_state(y, student_logits)
            if metric.name == "compile_metrics":
                other_metrics.append(metric.result())

        updated_metrics = {"total_loss": total_loss,
                           "student_loss": student_loss,
                           "logit_loss": logit_loss,
                           "feature_loss": feature_loss}
        for metric in other_metrics:
            updated_metrics.update(metric)
        return updated_metrics
    
class CAM_Distiller(keras.Model):
    def __init__(self, student, teacher, autoencoder):
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.autoencoder = autoencoder
        self.teacher_output_weights = self.teacher.layers[-1].get_weights()[0]
        self.jet_colors = tf.constant(
            np.array([plt.get_cmap("jet")(i)[:3] for i in range(256)]),
            dtype=tf.float32
        )

    @property
    def metrics(self):
        metrics = super().metrics

        return metrics

    def compile(
        self,
        optimizer,
        metrics,
        student_loss_fn,
        logit_loss_fn,
        feature_loss_fn,
        alpha=0.1,
        beta=0.1,
        temperature=3,
    ):
        super(CAM_Distiller, self).compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.logit_loss_fn = logit_loss_fn
        self.feature_loss_fn = feature_loss_fn
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature


    def create_combined_model(self, model, output_layers):
        return Model(inputs=model.input, outputs=[model.output] + output_layers)

    def spatial_alignment(self, teacher_features, student_features):
        aligned_features = []

        for teacher_feat, student_feat_list in zip(teacher_features, student_features):
            # Select the main feature tensor (the first element in each student feature list)
            # has the same shape as the teacher feature map
            student_feat = student_feat_list[0]

            # Step 0: Remove the class token
            student_feat = student_feat[:, 1:]

            # Step 1: Reshape student tokens to match CNN features
            # in short, (None, 196, 768) -> (None, 14, 14, 768)
            sqrt = int(np.sqrt(student_feat.shape[1]))
            student_resized = tf.reshape(student_feat, (-1, sqrt, sqrt, student_feat.shape[-1]))

            # Step 2: Perform bilinear interpolation to upsample student features
            h, w = teacher_feat.shape[1:3]
            student_interpolated = tf.image.resize(student_resized, (h, w))

            # Steps 3-4: Perform operations layer (pointwise convolution, layer normalization, ReLU activation)
            student_activated = OperationsLayer(teacher_feat.shape[-1])(student_interpolated)

            aligned_features.append((teacher_feat, student_activated))

        return aligned_features

    def train_step(self, data):
        x, y = data

        # Teacher model pass
        teacher_conv_layers = [layer.output for layer in self.teacher.layers if isinstance(layer, tf.keras.layers.Conv2D)]
        teacher_model = self.create_combined_model(self.teacher, [teacher_conv_layers[-1]])
        teacher_logits, conv_outputs = teacher_model(X1, training=False)

        # Precomputed teacher output weights (initialized outside train_step)
        output_weights = self.teacher_output_weights  # Pre-stored TensorFlow tensor
        if output_weights.shape[0] != conv_outputs.shape[-1]:
            output_weights = tf.reshape(output_weights, (conv_outputs.shape[-1], -1))

        # Compute CAMs in TensorFlow
        pred_indices = tf.argmax(tf.nn.softmax(teacher_logits), axis=-1)
        # Compute CAMs for all classes (shape: batch_size, height, width, num_classes)
        cams_all_classes = tf.einsum('bwhc,co->bwho', conv_outputs, output_weights)

        cams = tf.gather(cams_all_classes, pred_indices, axis=-1, batch_dims=1) 

        # Apply ReLU and normalize CAMs
        cams = tf.maximum(cams, 0)  # ReLU activation
        cams /= tf.reduce_max(cams, axis=[1, 2], keepdims=True) + 1e-5  # Avoid division by zero
        cams = tf.image.resize(tf.expand_dims(cams, axis=-1), [X1.shape[1], X1.shape[2]]) # Shape: (batch_size, h, w, 3)
        cams = tf.squeeze(cams, axis=-1)

        # Map CAM values to RGB colormap
        heatmaps = tf.gather(self.jet_colors, tf.cast(cams * 255, tf.int32))  # RGB mapping (batch_size, h, w, 3)

        # Resize heatmaps to match input image size
        heatmaps_resized = tf.image.resize(heatmaps, [X1.shape[1], X1.shape[2]])  # Shape: (batch_size, h, w, 3)

        # Blend with input image (X1 should have 3 channels)
        alpha = 0.4
        X2 = X1 + (heatmaps_resized * alpha)
        

        autoencoder_outputs = self.autoencoder(X2, training=False)

        with tf.GradientTape() as tape:
            student_model = self.create_combined_model(self.student, [layer.output for layer in self.student.layers if 'Transformer-encoderblock' in layer.name][-(spatial_alignment_layers):])
            student_logits, *student_features = student_model(X1, training=True)

            student_loss = tf.reduce_mean(self.student_loss_fn(y, tf.nn.softmax(student_logits)))

            logit_loss = tf.reduce_mean(self.logit_loss_fn(
                    tf.nn.softmax(teacher_logits / self.temperature, axis=-1),
                    tf.nn.softmax(student_logits / self.temperature, axis=-1)
                ) * self.temperature ** 2)

            # autoencoder part
            aligned_features = self.spatial_alignment(autoencoder_outputs, student_features)
            # aligned_features = self.spatial_alignment(teacher_features_orig, student_features)
            feature_loss = tf.reduce_mean(sum(
                [self.feature_loss_fn(
                    tf.nn.l2_normalize(teacher),
                    tf.nn.l2_normalize(student),
                    ) + 1
                for teacher, student in aligned_features]))

            # Combine losses (with weighting)
            # alpha times the student ViT's loss
            # plus (1-alpha) times the logit loss (logit loss should also be cross-entropy)
            # plus beta times the feature-based distillation loss
            total_loss = self.alpha * student_loss + (1 - self.alpha) * logit_loss + self.beta * feature_loss

        # Compute gradients and update student weights
        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update metrics
        other_metrics = []
        for metric in self.metrics:
            metric.update_state(y, student_logits)
            if metric.name == "compile_metrics":
                other_metrics.append(metric.result())

        # Update loss tracker and report progress
        updated_metrics = {"total_loss": total_loss,
                           "student_loss": student_loss,
                           "logit_loss": logit_loss,
                           "feature_loss": feature_loss}
        for metric in other_metrics:
            updated_metrics.update(metric)
        return updated_metrics

    def test_step(self, data):
        X1, y = data

        student_model = self.create_combined_model(self.student, [layer.output for layer in self.student.layers if 'Transformer-encoderblock' in layer.name][-(spatial_alignment_layers):])
        student_logits, *student_features = student_model(X1, training=False)

        student_loss = tf.reduce_mean(self.student_loss_fn(y, tf.nn.softmax(student_logits)))

        # Update metrics
        other_metrics = []
        for metric in self.metrics:
            metric.update_state(y, student_logits)
            if metric.name == "compile_metrics":
                other_metrics.append(metric.result())

        updated_metrics = {"student_loss": student_loss}
        for metric in other_metrics:
            updated_metrics.update(metric)
        return updated_metrics
    
class GradCAM_Distiller(keras.Model):
    def __init__(self, student, teacher, autoencoder):
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.autoencoder = autoencoder
        self.teacher_output_weights = self.teacher.layers[-1].get_weights()[0]
        self.jet_colors = tf.constant(
            np.array([plt.get_cmap("jet")(i)[:3] for i in range(256)]),
            dtype=tf.float32
        )

    @property
    def metrics(self):
        metrics = super().metrics

        return metrics

    def compile(
        self,
        optimizer,
        metrics,
        student_loss_fn,
        logit_loss_fn,
        feature_loss_fn,
        alpha=0.1,
        beta=0.1,
        temperature=3,
    ):
        super(GradCAM_Distiller, self).compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.logit_loss_fn = logit_loss_fn
        self.feature_loss_fn = feature_loss_fn
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature


    def create_combined_model(self, model, output_layers):
        return Model(inputs=model.input, outputs=[model.output] + output_layers)

    def spatial_alignment(self, teacher_features, student_features):
        aligned_features = []

        for teacher_feat, student_feat_list in zip(teacher_features, student_features):
            # Select the main feature tensor (the first element in each student feature list)
            # has the same shape as the teacher feature map
            student_feat = student_feat_list[0]

            # Step 0: Remove the class token
            student_feat = student_feat[:, 1:]

            # Step 1: Reshape student tokens to match CNN features
            # in short, (None, 196, 768) -> (None, 14, 14, 768)
            sqrt = int(np.sqrt(student_feat.shape[1]))
            student_resized = tf.reshape(student_feat, (-1, sqrt, sqrt, student_feat.shape[-1]))

            # Step 2: Perform bilinear interpolation to upsample student features
            h, w = teacher_feat.shape[1:3]
            student_interpolated = tf.image.resize(student_resized, (h, w))

            # Steps 3-4: Perform operations layer (pointwise convolution, layer normalization, ReLU activation)
            student_activated = OperationsLayer(teacher_feat.shape[-1])(student_interpolated)

            aligned_features.append((teacher_feat, student_activated))

        return aligned_features

    def train_step(self, data):
        X1, y = data

        # Teacher model pass
        teacher_conv_layers = [layer.output for layer in self.teacher.layers if isinstance(layer, tf.keras.layers.Conv2D)]
        teacher_model = self.create_combined_model(self.teacher, [teacher_conv_layers[-1]])
        teacher_logits, conv_outputs = teacher_model(X1, training=False)

        # GradCAM computation
        with tf.GradientTape() as tape1:
            teacher_logits, conv_outputs = teacher_model(X1, training=True)
            pred_indices = tf.argmax(tf.nn.softmax(teacher_logits), axis=-1)
            class_output = tf.gather(teacher_logits, pred_indices, axis=1, batch_dims=1)

        grads = tape1.gradient(class_output, conv_outputs)

        conv_outputs = tf.stop_gradient(conv_outputs)
        teacher_logits = tf.stop_gradient(teacher_logits)

        # Get the mean of the gradients
        pooled_grads = tf.reduce_mean(grads, axis=(1, 2))

        # Multiply the last conv layer output by the pooled grads
        heatmaps = tf.einsum('bijk,bk->bij', conv_outputs, pooled_grads)

        # Apply ReLU to the heatmap to keep only positive values
        heatmaps = tf.maximum(heatmaps, 0)  # Keep only positive values
        heatmaps = heatmaps / tf.reduce_max(heatmaps, axis=(1, 2), keepdims=True)
        heatmaps = tf.gather(self.jet_colors, tf.cast(heatmaps * 255, tf.int32))  # RGB mapping (batch_size, h, w, 3)

        # Resize heatmaps to match input image size
        heatmaps_resized = tf.image.resize(heatmaps, [X1.shape[1], X1.shape[2]])  # Shape: (batch_size, h, w, 3)

        # Blend with input image (X1 should have 3 channels)
        alpha = 0.4
        X2 = X1 + (heatmaps_resized * alpha)

        autoencoder_outputs = self.autoencoder(X2, training=False)

        with tf.GradientTape() as tape2:
            student_model = self.create_combined_model(self.student, [layer.output for layer in self.student.layers if 'Transformer-encoderblock' in layer.name][-(spatial_alignment_layers):])
            student_logits, *student_features = student_model(X1, training=True)

            student_loss = tf.reduce_mean(self.student_loss_fn(y, tf.nn.softmax(student_logits)))

            logit_loss = tf.reduce_mean(self.logit_loss_fn(
                    tf.nn.softmax(teacher_logits / self.temperature, axis=-1),
                    tf.nn.softmax(student_logits / self.temperature, axis=-1)
                ) * self.temperature ** 2)

            # autoencoder part
            aligned_features = self.spatial_alignment(autoencoder_outputs, student_features)
            # aligned_features = self.spatial_alignment(teacher_features_orig, student_features)
            feature_loss = tf.reduce_mean(sum(
                [self.feature_loss_fn(
                    tf.nn.l2_normalize(teacher),
                    tf.nn.l2_normalize(student),
                    ) + 1
                for teacher, student in aligned_features]))

            # Combine losses (with weighting)
            # alpha times the student ViT's loss
            # plus (1-alpha) times the logit loss (logit loss should also be cross-entropy)
            # plus beta times the feature-based distillation loss
            total_loss = self.alpha * student_loss + (1 - self.alpha) * logit_loss + self.beta * feature_loss

        # Compute gradients and update student weights
        trainable_vars = self.student.trainable_variables
        gradients = tape2.gradient(total_loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update metrics
        other_metrics = []
        for metric in self.metrics:
            metric.update_state(y, student_logits)
            if metric.name == "compile_metrics":
                other_metrics.append(metric.result())

        # Update loss tracker and report progress
        updated_metrics = {"total_loss": total_loss,
                           "student_loss": student_loss,
                           "logit_loss": logit_loss,
                           "feature_loss": feature_loss}
        for metric in other_metrics:
            updated_metrics.update(metric)
        return updated_metrics

    def test_step(self, data):
        X1, y = data

        student_model = self.create_combined_model(self.student, [layer.output for layer in self.student.layers if 'Transformer-encoderblock' in layer.name][-(spatial_alignment_layers):])
        student_logits, *student_features = student_model(X1, training=False)

        student_loss = tf.reduce_mean(self.student_loss_fn(y, tf.nn.softmax(student_logits)))

        # Update metrics
        other_metrics = []
        for metric in self.metrics:
            metric.update_state(y, student_logits)
            if metric.name == "compile_metrics":
                other_metrics.append(metric.result())

        updated_metrics = {"student_loss": student_loss}
        for metric in other_metrics:
            updated_metrics.update(metric)
        return updated_metrics