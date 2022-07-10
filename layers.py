import tensorflow as tf
import tensorflow_addons as tfa
import einops


def window_partition(x, window_size):
    return einops.rearrange(
        x, 'B (H hw) (W ww) C -> (B H W) hw ww C',
        hw=window_size, ww=window_size
    )


def window_reverse(windows, h, w):
    return einops.rearrange(
        windows, '(B H W) hw ww C -> B (H hw) (W ww) C',
        H=h, W=w
    )


class MLP(tf.keras.layers.Layer):
    def __init__(
            self,
            n_filters: int,
            expansion_rate: float = 4.,
            drop: float = 0.,
            act=tf.nn.gelu
    ):
        super(MLP, self).__init__()
        self.n_filters = n_filters
        self.expansion_rate = expansion_rate
        self.drop = drop
        self.act = act

        self.mlp1 = tf.keras.layers.Dense(
            self.n_filters * self.expansion_rate,
            kernel_initializer=tf.keras.initializers.VarianceScaling(0.02)
        )
        self.drop1 = tf.keras.layers.Dropout(self.drop) if self.drop != 0. else tf.keras.layers.Layer()
        self.mlp2 = tf.keras.layers.Dense(
            self.n_filters,
            kernel_initializer=tf.keras.initializers.VarianceScaling(.02)
        )
        self.drop2 = tf.keras.layers.Dropout(self.drop) if self.drop != 0. else tf.keras.layers.Layer()

    def call(self, inputs, *args, **kwargs):
        return self.drop2(self.mlp2(self.drop1(self.act(self.mlp1(inputs)))))


class PatchUnembed(tf.keras.layers.Layer):
    def __init__(self):
        super(PatchUnembed, self).__init__()

    def call(self, inputs, h, w, *args, **kwargs):
        _, _, c = inputs.get_shape().as_list()
        return einops.rearrange(
            inputs, 'B (H W) C -> B H W C',
            H=h, W=w, C=c
        )


class PatchEmbed(tf.keras.layers.Layer):
    def __init__(
            self,
            norm=None
    ):
        super(PatchEmbed, self).__init__()
        self.norm = norm or tf.keras.layers.Layer
        self.norm = self.norm()

    def call(self, inputs, return_HW=False, *args, **kwargs):
        embedding = einops.rearrange(
            inputs, 'B H W C -> B (H W) C'
        )
        if return_HW:
            _, h, w, _ = inputs.get_shape().as_list()
            return self.norm(embedding), h, w
        else:
            return self.norm(embedding)

