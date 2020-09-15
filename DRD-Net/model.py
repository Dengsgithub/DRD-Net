from keras.models import Model
from keras.layers import Input, Add,Subtract, PReLU, Conv2DTranspose, \
    Concatenate, MaxPooling2D, UpSampling2D, Dropout, concatenate, GlobalAveragePooling2D,\
    Reshape, Dense, multiply
from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import Callback
from keras import backend as K
import tensorflow as tf


class L0Loss:
    def __init__(self):
        self.gamma = K.variable(2.)

    def __call__(self):
        def calc_loss(y_true, y_pred):
            loss = K.pow(K.abs(y_true - y_pred) + 1e-8, self.gamma)
            return loss
        return calc_loss


class UpdateAnnealingParameter(Callback):
    def __init__(self, gamma, nb_epochs, verbose=0):
        super(UpdateAnnealingParameter, self).__init__()
        self.gamma = gamma
        self.nb_epochs = nb_epochs
        self.verbose = verbose

    def on_epoch_begin(self, epoch, logs=None):
        new_gamma = 2.0 * (self.nb_epochs - epoch) / self.nb_epochs
        K.set_value(self.gamma, new_gamma)

        if self.verbose > 0:
            print('\nEpoch %05d: UpdateAnnealingParameter reducing gamma to %s.' % (epoch + 1, new_gamma))


def tf_log10(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator


def PSNR(y_true, y_pred):
    y_true_t = y_true *255.0
    max_pixel = 255.0
    y_pred_t = K.clip(y_pred*255.0, 0.0, 255.0)
    # y_pred = K.clip(y_pred, 0.0, 1.0)
    return 10.0 * tf_log10((max_pixel ** 2) / (K.mean(K.square(y_pred_t - y_true_t))))


def get_model(model_name="the_end"):

    if model_name == "the_end":
        return get_the_end_model()
    else:
        raise ValueError("model_name should be 'srresnet'or 'unet'")


def get_the_end_model(input_channel_num=3, feature_dim=64, resunit_num=16):

    def _back_net(inputs):
        def _residual_block(inputs):
            x = _empty_block(inputs)
            x = BatchNormalization()(x)
            x = PReLU(shared_axes=[1, 2])(x)
            x = _empty_block(x)
            x = BatchNormalization()(x)
            m = Add()([x, inputs])
            return m

        def _empty_block(inputs):
            x1 = Conv2D(feature_dim, (3, 3), padding="same", kernel_initializer="he_normal")(inputs)
            x2 = Conv2D(feature_dim, (3, 3), dilation_rate=3, padding="same", kernel_initializer="he_normal")(inputs)
            x3 = Conv2D(feature_dim, (3, 3), dilation_rate=5, padding="same", kernel_initializer="he_normal")(inputs)
            x = concatenate([x1, x2, x3], axis=-1)
            x_out = Conv2D(feature_dim, (1, 1), padding="same", kernel_initializer="he_normal")(x)
            return x_out

        x = Conv2D(feature_dim, (3, 3), padding="same", kernel_initializer="he_normal")(inputs)
        x = PReLU(shared_axes=[1, 2])(x)
        x0 = x

        for i in range(resunit_num):
            x = _residual_block(x)

        x = Conv2D(feature_dim, (3, 3), padding="same", kernel_initializer="he_normal")(x)
        x = BatchNormalization()(x)
        x = Add()([x, x0])
        x = Conv2D(input_channel_num, (3, 3), padding="same", kernel_initializer="he_normal")(x)
        return x

    def _rain_net(inputs):
        def _residual_block(inputs, number):
            x = Conv2D(feature_dim, (3, 3), padding="same", kernel_initializer="he_normal")(inputs)
            x = BatchNormalization()(x)
            x = PReLU(shared_axes=[1, 2])(x)
            x = Conv2D(feature_dim, (3, 3), padding="same", kernel_initializer="he_normal")(x)
            x = BatchNormalization()(x)
            filters = 64
            se_shape = (1, 1, filters)
            se = GlobalAveragePooling2D()(x)
            se = Reshape(se_shape)(se)
            se = Dense(number, activation="relu",
                       kernel_initializer="he_normal",use_bias=False)(se)
            se = Dense(filters, activation="hard_sigmoid",
                       kernel_initializer="he_normal", use_bias=False)(se)
            x = multiply([x, se])
            m = Add()([x, inputs])
            return m

        x = Conv2D(feature_dim, (3, 3), padding="same", kernel_initializer="he_normal")(inputs)
        x = PReLU(shared_axes=[1, 2])(x)
        x0 = x

        for i in range(resunit_num):
            x = _residual_block(x, 4)

        x = Conv2D(feature_dim, (3, 3), padding="same", kernel_initializer="he_normal")(x)
        x = BatchNormalization()(x)
        x = Add()([x, x0])
        x = Conv2D(input_channel_num, (3, 3), padding="same", kernel_initializer="he_normal")(x)
        return x

    inputs = Input(shape=(None, None, input_channel_num), name='Rain_image')
    Rain = _rain_net(inputs)

    out1 = Subtract()([inputs, Rain])
    Back_in = Add()([out1, inputs])
    Back = _back_net(Back_in)
    out = Add()([out1, Back])

    model = Model(inputs=inputs, outputs=[out1, out])
    return model


def main():
    # model = get_model()
    model = get_model("unet")
    model.summary()


if __name__ == '__main__':
    main()
