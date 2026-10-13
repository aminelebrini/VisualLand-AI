import tensorflow as tf
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from collections import Counter
import cv2
class DatasetExplorer:
    def __init__(self, image_dataset):
        self.dataset_dir = Path(image_dataset)

    def identify_classes_and_distribution(self):

        images_class = {
            "buildings",
            "forest",
            "glacier",
            "mountain",
            "sea",
            "street"
        }
        class_count = {}
        distribution_images = {}
        count_total = 0
        for class_folder_image in self.dataset_dir.iterdir():
            
            if class_folder_image.is_dir():
                n_image = 0
                for image_path in class_folder_image.rglob("*"):
                    if image_path.is_file():
                        n_image += 1
                class_count[class_folder_image.name] = n_image

        for class_name in images_class:
            count_total += class_count.get(class_name)

        for class_name in sorted(images_class):
            n_image = class_count.get(class_name,0)

            if count_total > 0:
                distribution_images[class_name] = (n_image / count_total) * 100
            else:
                distribution_images[class_name] = 0

        print("total images:", count_total)
        print(f"distribution des images: {distribution_images[class_name]:.2f}")

        return class_count, distribution_images




    def count_image_per_class(self):

        class_count = {}
        
        for class_folder_image in self.dataset_dir.iterdir():
            if class_folder_image.is_dir():
                n_image = 0
                for image_path in class_folder_image.rglob("*"):
                    if image_path.is_file():
                        n_image += 1
        
                class_count[class_folder_image.name] = n_image
        
        
        return class_count

    def data_image_visualisation(self):

        for class_folder_image in self.dataset_dir.iterdir():
            if class_folder_image.is_dir():
                images = list(class_folder_image.glob("*.jpg"))[:3]

                for image_path in images:
                    image = tf.keras.utils.load_img(image_path, target_size=(256, 256))
                    plt.imshow(image)
                    plt.title(class_folder_image.name)
                    plt.axis("off")
                    plt.show()

    def get_dimentions_format(self):
        min_width = None
        min_height = None
        extensions = {".jpg", ".jpeg", ".png"}

        for class_folder_image in self.dataset_dir.iterdir():
            if class_folder_image.is_dir():
                for image_path in class_folder_image.glob("*"):
                    if image_path.is_file() and image_path.suffix.lower() in extensions:
                        with Image.open(image_path) as image:
                            width , height = image.size
                        
                            if min_width is None or width < min_width:
                                min_width = width
                        
                            if min_height is None or height < min_height:
                                min_height = height
                    else:
                        print("images not found!")

        return min_width, min_height

