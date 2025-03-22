import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout,Conv2DTranspose, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from matrics import jaccard_distance,iou, dice_coe, precision, recall, accuracy

class NestedModel:
    """
    A class that encapsulates the nested U-Net-like architecture.
    """

    def __init__(self, img_size, filter_size, init_lr=3e-5, out_ch=1):
        """
        Initialize the model parameters.
        
        Parameters:
            img_size (tuple): (height, width) of the input images.
            filter_size (int): Base filter size used for some layers.
            init_lr (float): Initial learning rate for the optimizer.
            out_ch (int): Number of output channels.
        """
        self.img_size = img_size
        self.filter_size = filter_size
        self.init_lr = init_lr
        self.out_ch = out_ch

    def build_model(self):
        """
        Builds and compiles the Keras model.
        
        Returns:
            model (tf.keras.Model): The compiled model.
        """
        k = 3  # kernel size for some transposed conv layers
        s = 2  # stride for transposed conv layers
        img_ch = 3  # number of image channels
        img_height, img_width = self.img_size

        # Input layer
        inputs = Input((img_height, img_width, img_ch))

        # Encoder
        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(pool1)
        conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
        conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

        # Decoder / Upsampling path
        up1_2 = Conv2DTranspose(32, (3, 3), strides=(2, 2), name='up12', padding='same')(conv2)
        conv1_2 = concatenate([up1_2, conv1], name='merge12', axis=3)
        conv1_2 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1_2)
        conv1_2 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1_2)

        up2_2 = Conv2DTranspose(2 * self.filter_size, (k, k), strides=(s, s), name='up13', padding='same')(conv3)
        conv2_2 = concatenate([up2_2, conv2], name='merge13', axis=3)
        conv2_2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2_2)
        conv2_2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2_2)

        up1_3 = Conv2DTranspose(self.filter_size, (k, k), strides=(s, s), name='up14', padding='same')(conv2_2)
        conv1_3 = concatenate([up1_3, conv1], name='merge14', axis=3)
        conv1_3 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1_3)
        conv1_3 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1_3)

        up3_3 = Conv2DTranspose(4 * self.filter_size, (k, k), strides=(s, s), name='up15', padding='same')(conv4)
        conv3_2 = concatenate([up3_3, conv3], name='merge15', axis=3)
        conv3_2 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3_2)
        conv3_2 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3_2)

        up2_3 = Conv2DTranspose(2 * self.filter_size, (k, k), strides=(s, s), name='up16', padding='same')(conv3_2)
        conv2_3 = concatenate([up2_3, conv2, conv2_2], name='merge16', axis=3)
        conv2_3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv2_3)
        conv2_3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv2_3)

        up1_4 = Conv2DTranspose(self.filter_size, (k, k), strides=(s, s), name='up17', padding='same')(conv2_3)
        conv1_4 = concatenate([up1_4, conv1, conv1_2, conv1_3], name='merge17', axis=3)
        conv1_4 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1_4)
        conv1_4 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1_4)

        up4_2 = Conv2DTranspose(8 * self.filter_size, (k, k), strides=(s, s), name='up18', padding='same')(conv5)
        conv4_2 = concatenate([up4_2, conv4], name='merge18', axis=3)
        conv4_2 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4_2)
        conv4_2 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4_2)

        up3_3_b = Conv2DTranspose(4 * self.filter_size, (k, k), strides=(s, s), name='up19', padding='same')(conv4_2)
        conv3_3 = concatenate([up3_3_b, conv3, conv3_2], name='merge19', axis=3)
        conv3_3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3_3)
        conv3_3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3_3)

        up2_4 = Conv2DTranspose(2 * self.filter_size, (k, k), strides=(s, s), name='up20', padding='same')(conv3_3)
        conv2_4 = concatenate([up2_4, conv2, conv2_2, conv2_3], name='merge20', axis=3)
        conv2_4 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2_4)
        conv2_4 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2_4)

        up1_5 = Conv2DTranspose(self.filter_size, (k, k), strides=(s, s), name='up21', padding='same')(conv2_4)
        conv1_5 = concatenate([up1_5, conv1, conv1_2, conv1_3], name='merge21', axis=3)
        conv1_5 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1_5)
        conv1_5 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1_5)

        # Output layer
        outputs = Conv2D(self.out_ch, (1, 1), padding='same', activation='hard_sigmoid')(conv1_5)

        model = Model(inputs, outputs)
        model.compile(optimizer=Adam(lr=self.init_lr),
                      loss=jaccard_distance,
                      metrics=[iou, dice_coe, precision, recall, accuracy])
        return model


# if __name__ == '__main__':
#     # Example usage:
#     img_size = (256, 256)
#     filter_size = 64
#     init_lr = 3e-5

#     nested_model = NestedModel(img_size, filter_size, init_lr)
#     model = nested_model.build_model()
#     model.summary()  # Prints model architecture
