# Original SwinIR Code(Pytorch): https://github.com/JingyunLiang/SwinIR
# Get code from https://github.com/sayakpaul/swin-transformers-tf
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow_addons as tfa
import einops

import numpy as np
import math

from typing import Sequence, Union


def window_partition(x, window_size):
    return einops.rearrange(
        x, 'B (H hw) (W ww) C -> (B H W) hw ww C',
        hw=window_size, ww=window_size
    )


def window_reverse(x, h, w):
    return einops.rearrange(
        x, '(B H W) hw ww C -> B (H hw) (W ww) C',
        H=h, W=w
    )

class DropPath(tf.keras.layers.Layer):
    def __init__(
            self,
            survival_prob
    ):
        super(DropPath, self).__init__()
        self.survival_prob = survival_prob

    def call(self, inputs, *args, **kwargs):

        def _call_train():
            state = K.random_bernoulli(
                shape=(), p=self.survival_prob
            )
            return inputs / self.survival_prob * state

        def _call_test():
            return inputs

        return K.in_train_phase(
            _call_train,
            _call_test
        )


class MLP(tf.keras.layers.Layer):
    def __init__(
            self,
            n_filters: int,
            expansion_rate: float = 4.,
            drop_rate: float = 0.,
            act=tf.nn.gelu
    ):
        super(MLP, self).__init__()
        self.n_filters = n_filters
        self.expansion_rate = expansion_rate
        self.drop_rate = drop_rate
        self.act = act

        self.forward = tf.keras.Sequential([
            tf.keras.layers.Dense(
                self.n_filters * self.expansion_rate,
                kernel_initializer=tf.keras.initializers.VarianceScaling(.02)
            ),
            tf.keras.layers.Lambda(lambda x: self.act(x)),
            tf.keras.layers.Dense(
                self.n_filters,
                kernel_initializer=tf.keras.initializers.VarianceScaling(.02)
            )
        ] + [
            DropPath(1. - self.drop_rate) if self.drop_rate > 0. else tf.keras.layers.Layer()
        ])

    def call(self, inputs, *args, **kwargs):
        return self.forward(inputs)


class WindowAttention(tf.keras.layers.Layer):
    def __init__(
            self,
            n_filters: int,
            drop_rate: float,
            window_size: int,
            shift_size: Union[int, None],
            n_heads: int,
            qk_scale: Union[float, None] = None,
            qkv_bias: bool = True
    ):
        super(WindowAttention, self).__init__()
        self.n_filters = n_filters
        self.drop_rate = drop_rate
        self.n_heads = n_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.qk_scale = qk_scale or n_filters ** -.5
        self.qkv_bias = qkv_bias

        self.to_qkv = tf.keras.layers.Dense(
            self.n_filters * 3,
            use_bias=self.qkv_bias,
            kernel_initializer=tf.keras.initializers.VarianceScaling(.02)
        )
        self.relative_position_index = self.get_relative_position_index(self.window_size, self.window_size)
        self.proj = tf.keras.layers.Dense(
            self.n_filters,
            kernel_initializer=tf.keras.initializers.VarianceScaling(.02)
        )
        self.droppath = DropPath(1. - self.drop_rate) if self.drop_rate > 0. else tf.identity

    def get_relative_position_index(self, win_h, win_w):
        xx, yy = tf.meshgrid(range(win_h), range(win_w))
        coords = tf.stack([yy, xx], axis=0)
        coords_flatten = tf.reshape(coords, [2, -1])

        relative_coords = (
            coords_flatten[:, :, None] - coords_flatten[:, None, :]
        )
        relative_coords = tf.transpose(
            relative_coords, perm=[1, 2, 0]
        )
        xx = (relative_coords[:, :, 0] + win_h - 1) * (2 * win_w - 1)
        yy = relative_coords[:, :, 1] + win_w - 1
        relative_coords = tf.stack([xx, yy], axis=-1)
        return tf.reduce_sum(relative_coords, axis=-1)

    def get_relative_positional_bias(self):
        relative_position_bias = tf.gather(
            self.relative_position_bias_table,
            self.relative_position_index,
            axis=0
        )
        return tf.transpose(relative_position_bias, [2, 0, 1])

    def build(self, input_shape):
        self.relative_position_bias_table = self.add_weight(
            shape=((2 * self.window_size - 1) * (2 * self.window_size - 1), self.n_heads),
            initializer='zeros',
            trainable=True,
            name='relative_position_bias_table'
        )
        super(WindowAttention, self).build(input_shape)

    def call(self, inputs, mask=None, *args, **kwargs):
        # input : B N C
        _, N, _ = tf.shape(inputs)
        q, k, v = tf.unstack(
            einops.rearrange(
                self.to_qkv(
                    inputs
                ), 'B N (QKV H C) -> QKV B H N C',
                QKV=3, H=self.n_heads
            ), num=3, axis=0
        )
        attention_map = tf.matmul(q, k, transpose_b=True) * self.qk_scale
        attention_map = attention_map + self.get_relative_positional_bias()

        if tf.is_tensor(mask):
            num_wins = tf.shape(mask)[0]
            attention_map = tf.reshape(
                attention_map, (-1, num_wins, self.n_heads, N, N)
            )
            attention_map = attention_map + tf.expand_dims(mask, 1)[None, ...]

            attention_map = tf.reshape(attention_map, (-1, self.n_heads, N, N))

        attention = tf.nn.softmax(attention_map, axis=-1)

        out = einops.rearrange(
            tf.matmul(attention, v), 'B H N C -> B N (H C)'
        )
        out = self.droppath(
            self.proj(out)
        )
        return out


class SwinTransformerLayer(tf.keras.layers.Layer):
    def __init__(
            self,
            n_filters: int,
            window_size: int,
            shift_size: Union[int, None],
            n_heads: int,
            qkv_bias: bool,
            drop_rate: float,
            norm=tf.keras.layers.LayerNormalization
    ):
        super(SwinTransformerLayer, self).__init__()
        self.n_filters = n_filters
        self.window_size = window_size
        self.shift_size = shift_size
        self.n_heads = n_heads
        self.qkv_bias = qkv_bias
        self.drop_rate = drop_rate

        self.ln_attn = norm()
        self.window_attention = WindowAttention(
            self.n_filters, self.drop_rate, self.window_size, self.shift_size,
            self.n_heads, qkv_bias=self.qkv_bias
        )
        self.ln_ffn = norm()
        self.ffn = MLP(
            self.n_filters, self.drop_rate
        )

    def get_attention_mask(self, input_shape: Sequence):
        img_mask = np.zeros((1, input_shape[0], input_shape[1], 1))
        cnt = 0
        for h in (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        ):
            for w in (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None)
            ):
                img_mask[:, h, w, :] = cnt
                cnt += 1

        img_mask = tf.convert_to_tensor(img_mask, dtype='float32')
        mask_windows = window_partition(
            img_mask, self.window_size
        )
        mask_windows = tf.reshape(
            mask_windows, (-1, self.window_size * self.window_size)
        )
        attn_mask = tf.expand_dims(mask_windows, 1) - tf.expand_dims(mask_windows, 2)
        attn_mask = tf.where(attn_mask != 0, -100., attn_mask)
        return tf.where(attn_mask == 0, 0., attn_mask)

    def call(self, inputs, *args, **kwargs):
        _, h, w, _ = inputs.shape  # B H W C

        attn_res = self.ln_attn(inputs)
        if self.shift_size is not None:
            attn_res = tf.roll(
                attn_res, shift=(-self.shift_size, -self.shift_size), axis=(1, 2)
            )
            mask = self.get_attention_mask((h, w))
        else:
            mask = None

        attn_res = window_partition(attn_res, self.window_size)  #n_windows*B window_size window_size C
        attn_res = einops.rearrange(
            attn_res, 'B W1 W2 C -> B (W1 W2) C'
        )  # n_windows*B window_size*window_size C

        attn_res = self.window_attention(attn_res, mask=mask)  # n_windows*B window_size*window_size C
        attn_res = einops.rearrange(
            attn_res, 'B (W1 W2) C -> B W1 W2 C',
            W1=self.window_size, W2=self.window_size
        )  # n_windows*B window_size*window_size C
        attn_res = window_reverse(attn_res, h//self.window_size, w//self.window_size)  # B H W C

        if self.shift_size is not None:
            attn_res = tf.roll(
                attn_res, shift=(self.shift_size, self.shift_size), axis=(1, 2)
            )  # B H W C

        inputs += attn_res

        inputs += self.ffn(self.ln_ffn(inputs))

        return inputs


class ResidualSwinTransformerBlock(tf.keras.layers.Layer):
    def __init__(
            self,
            n_blocks: int,
            n_filters: int,
            window_size: int,
            n_heads: int,
            qkv_bias: bool,
            drop_rate: Sequence[float],
            res_connection: str
    ):
        super(ResidualSwinTransformerBlock, self).__init__()
        self.n_blocks = n_blocks
        self.n_filters = n_filters
        self.window_size = window_size
        self.shift_size = window_size//2
        self.n_heads = n_heads
        self.qkv_bias = qkv_bias
        assert len(drop_rate) == n_blocks
        self.drop_rate = drop_rate
        self.res_connection = res_connection

        self.forward = tf.keras.Sequential([
            SwinTransformerLayer(
                self.n_filters,
                self.window_size,
                None if (i % 2 == 0) else self.shift_size,
                self.n_heads,
                self.qkv_bias,
                dr
            ) for i, dr in enumerate(self.drop_rate)
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

    def call(self, inputs, *args, **kwargs):
        return inputs + self.forward(inputs)


class PixelShuffle(tf.keras.layers.Layer):
    def __init__(self, upsample_rate: int):
        super(PixelShuffle, self).__init__()
        self.upsample_rate = upsample_rate

    def call(self, inputs, *args, **kwargs):
        return tf.nn.depth_to_space(
            inputs, self.upsample_rate
        )


class UpsampleOneStep(tf.keras.layers.Layer):
    def __init__(self, output_channel: int, upsample_rate: int):
        super(UpsampleOneStep, self).__init__()
        self.output_channel = output_channel
        self.upsample_rate = upsample_rate

        self.forward = tf.keras.Sequential([
            tf.keras.layers.Conv2D(
                self.output_channel * (self.upsample_rate ** 2),
                (3, 3),
                strides=(1, 1),
                padding='SAME',
                kernel_initializer=tf.keras.initializers.VarianceScaling(.02)
            ),
            PixelShuffle(self.upsample_rate)
        ])

    def call(self, inputs, *args, **kwargs):
        return self.forward(inputs)


class Upsample(tf.keras.layers.Layer):
    def __init__(self, n_filters: int, output_channel: int, upsample_rate: int):
        super(Upsample, self).__init__()
        self.n_filters = n_filters
        self.output_channel = output_channel
        self.upsample_rate = upsample_rate

        self.forward = tf.keras.Sequential()
        self.forward.add(
            tf.keras.layers.Conv2D(
                self.n_filters,
                (3, 3),
                strides=(1, 1),
                padding='SAME',
                kernel_initializer=tf.keras.initializers.VarianceScaling(.02),
                activation=tf.keras.layers.LeakyReLU(alpha=.01)
            )
        )
        if (self.upsample_rate & (self.upsample_rate - 1)) == 0:
            for _ in range(int(math.log(self.upsample_rate, 2))):
                self.forward.add(
                    tf.keras.layers.Conv2D(
                        4 * self.n_filters,
                        (3, 3),
                        strides=(1, 1),
                        padding='SAME',
                        kernel_initializer=tf.keras.initializers.VarianceScaling(.02)
                    )
                )
                self.forward.add(
                    PixelShuffle(2)
                )
        elif self.upsample_rate == 3:
            self.forward.add(
                tf.keras.layers.Conv2D(
                    self.n_filters * 9,
                    (3, 3),
                    strides=(1, 1),
                    padding='SAME',
                    kernel_initializer=tf.keras.initializers.VarianceScaling(.02)
                )
            )
            self.forward.add(
                PixelShuffle(3)
            )
        else:
            raise NotImplementedError('Invalid scale')
        self.forward.add(
            tf.keras.layers.Conv2D(
                self.output_channel,
                (3, 3),
                strides=(1, 1),
                padding='SAME',
                kernel_initializer=tf.keras.initializers.VarianceScaling(.02)
            )
        )

    def call(self, inputs, *args, **kwargs):
        return self.forward(inputs)


class UpsampleNearestConv(tf.keras.layers.Layer):
    def __init__(self, n_filters: int, output_channel: int, upscale_rate: int):
        super(UpsampleNearestConv, self).__init__()
        self.n_filters = n_filters
        self.output_channel = output_channel
        self.upscale_rate = upscale_rate  # 2 or 4

        self.forward = tf.keras.Sequential([
            tf.keras.layers.Conv2D(
                self.n_filters,
                (3, 3),
                strides=(1, 1),
                padding='SAME',
                kernel_initializer=tf.keras.initializers.VarianceScaling(.02),
                activation=tf.keras.layers.LeakyReLU(alpha=.01)
            ),
            tf.keras.layers.UpSampling2D(),
            tf.keras.layers.Conv2D(
                self.n_filters,
                (3, 3),
                strides=(1, 1),
                padding='SAME',
                kernel_initializer=tf.keras.initializers.VarianceScaling(.02),
                activation=tf.keras.layers.LeakyReLU(alpha=.2)
            )
        ])
        if upscale_rate == 4:
            self.forward.add(
                tf.keras.layers.UpSampling2D()
            )
            self.forward.add(
                tf.keras.layers.Conv2D(
                    self.n_filters,
                    (3, 3),
                    strides=(1, 1),
                    padding='SAME',
                    kernel_initializer=tf.keras.initializers.VarianceScaling(.02),
                    activation=tf.keras.layers.LeakyReLU(alpha=.2)
                )
            )
        self.forward.add(
            tf.keras.layers.Conv2D(
                self.output_channel,
                (3, 3),
                strides=(1, 1),
                padding='SAME',
                kernel_initializer=tf.keras.initializers.VarianceScaling(.02)
            )
        )

    def call(self, inputs, *args, **kwargs):
        return self.forward(inputs)
