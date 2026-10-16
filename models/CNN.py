import tensorflow as tf

class CNN:
    def __init__(self, ):
        self.model = self.build_model()

    def build_model(self):

        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32,(3,3), activation='relu', input_shape=(150, 150, 3)),
            tf.keras.layers.MaxPooling2D((2,2), strides=2),

            tf.keras.layers.Conv2D(64,(3,3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2,2), strides=2),


            tf.keras.layers.Conv2D(128,(3,3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2,2), strides=2),

            tf.keras.layers.GlobalAveragePooling2D(),

            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),

            tf.keras.layers.Dense(6, activation='softmax')

        ])

        return model

    def training(self, data_set):
        history = self.model.fit(
            data_set,
            epochs=10,
        )

        return history