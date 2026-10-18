import tensorflow as tf        

class DataTraitement:
    def __init__(self):
        pass
    def redimensionnement(self, data_images):
        resizable_image = []
        for image_path in data_images:

            image = tf.keras.utils.load_img(str(image_path))
            image_array = tf.keras.utils.img_to_array(image)
            resizable_image.append(
                tf.keras.preprocessing.image.smart_resize(image_array, (256, 256))
                )

        return resizable_image

    def normalisation(self, data_images):
        normalized_images = []

        for image in data_images:
            image_floated = tf.cast(image, tf.float32)
            normlize_image = image_floated / 255.0
            normalized_images.append(normlize_image)

        return normalized_images

    def geneator(self, data_images, labels):
        for image, label in zip(data_images, labels):
            yield image, label

    def chagement_par_batch(self, data_images, labels):
        
        dataset = tf.data.Dataset.from_generator(
            lambda: self.geneator(data_images, labels),
            output_signature=(
            tf.TensorSpec(shape=(256, 256, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32)
            )
        )
        dataset = dataset.shuffle(buffer_size=1000)
        dataset = dataset.batch(32)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

    def data_augmentation(self, data_images):
       Data_augmentation = tf.keras.Sequential([
           tf.keras.layers.RandomFlip("horizontal_and_vertical"),
           tf.keras.layers.RandomRotation(0.2),
           tf.keras.layers.RandomZoom(0.2),
       ])

       augemented_dataset = data_images.map(lambda x, y: (Data_augmentation(x, training=True), y))

       return augemented_dataset
        