# Original SwinIR Code(Pytorch): https://github.com/JingyunLiang/SwinIR
# Get code from https://github.com/sayakpaul/swin-transformers-tf
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow_addons as tfa
import einops
from layers import *

import numpy as np

from typing import Sequence, Union


class SwinIR(tf.keras.models.Model):
    def __init__(
            self,
            n_filters: int,
            n_rstb_blocks: int,
            n_stl_blocks: int,
            window_size: int,
            n_heads: int,
            qkv_bias: bool,
            drop_rate: float,
            res_connection: str,
            upsample_type: str,
            upsample_rate: int,
            img_range: Union[int, float],
            output_channel: int = 3,
            mean: Sequence[float] = [.4488, .4371, .4040]
    ):
        super(SwinIR, self).__init__()
        self.n_filters = n_filters
        self.n_rstb_blocks = n_rstb_blocks
        self.n_stl_blocks = n_stl_blocks
        self.window_size = window_size
        self.n_heads = n_heads
        self.qkv_bias = qkv_bias
        self.drop_rate = drop_rate
        self.res_connection = res_connection
        self.upsample_type = upsample_type
        self.upsample_rate = upsample_rate
        self.img_range = img_range
        self.output_channel = output_channel
        self.mean = tf.reshape(
            tf.convert_to_tensor(mean, dtype='float32'),
            (1, 1, 1, -1)
        )

        self.upsample_filters = 64

        stochastic_depth_rate = tf.linspace(
            0., self.drop_rate, self.n_rstb_blocks * self.n_stl_blocks
        )

        self.shallow_feature_extractor = tf.keras.layers.Conv2D(
            self.n_filters,
            (3, 3),
            strides=(1, 1),
            padding='SAME',
            kernel_initializer=tf.keras.initializers.VarianceScaling(.02)
        )
        self.deep_feature_extractor = tf.keras.Sequential([
            ResidualSwinTransformerBlock(
                self.n_stl_blocks,
                self.n_filters,
                self.window_size,
                self.n_heads,
                self.qkv_bias,
                stochastic_depth_rate[n * self.n_stl_blocks:(n + 1) * self.n_stl_blocks],
                self.res_connection
            ) for n in range(self.n_rstb_blocks)
        ] + [
            tf.keras.layers.LayerNormalization()
        ] + [
            tf.keras.layers.Conv2D(
                self.n_filters,
                (3, 3),
                strides=(1, 1),
                padding='SAME',
                kernel_initializer=tf.keras.initializers.VarianceScaling(.02)
            ) if self.res_connection == '1conv' else tf.keras.Sequential([
                tf.keras.layers.Conv2D(
                    self.n_filters//4,
                    (3, 3),
                    strides=(1, 1),
                    padding='SAME',
                    kernel_initializer=tf.keras.initializers.VarianceScaling(.02),
                    activation=tf.keras.layers.LeakyReLU(alpha=.2)
                ),
                tf.keras.layers.Conv2D(
                    self.n_filters//4,
                    (1, 1),
                    strides=(1, 1),
                    padding='VALID',
                    kernel_initializer=tf.keras.initializers.VarianceScaling(.02),
                    activation=tf.keras.layers.LeakyReLU(alpha=.2)
                ),
                tf.keras.layers.Conv2D(
                    self.n_filters,
                    (3, 3),
                    strides=(1, 1),
                    padding='SAME',
                    kernel_initializer=tf.keras.initializers.VarianceScaling(.02)
                )
            ])
        ])

        if self.upsample_type == 'pixelshuffle':
            self.reconstructor = Upsample(
                self.upsample_filters, self.output_channel, self.upsample_rate
            )
        elif self.upsample_type == 'pixelshuffle_onestep':
            self.reconstructor = UpsampleOneStep(
                self.output_channel, self.upsample_rate
            )
        elif self.upsample_type == 'nearest_conv':
            self.reconstructor = UpsampleNearestConv(
                self.upsample_filters, self.output_channel, self.upsample_rate
            )
        else:
            self.reconstructor = tf.keras.layers.Conv2D(
                self.output_channel,
                (3, 3),
                strides=(1, 1),
                padding='SAME',
                kernel_initializer=tf.keras.initializers.VarianceScaling(.02)
            )

    def forward(self, x, training=False):
        x = (x - self.mean) * self.img_range

        if self.upsample_type in ['pixelshuffle', 'pixelshuffle_onestep', 'nearest_conv']:
            features = self.shallow_feature_extractor(x, training=training)
            features = self.deep_feature_extractor(features, training=training) + features
            reconstruction = self.reconstructor(features, training=training)
        else:
            features = self.shallow_feature_extractor(x)
            features = self.deep_feature_extractor(features) + features
            reconstruction = self.reconstructor(features) + x

        reconstruction = (reconstruction / self.img_range) + self.mean
        return reconstruction

    def call(self, inputs, training=None, mask=None):
        if training is None:
            training = False
        return self.forward(inputs, training=training)
