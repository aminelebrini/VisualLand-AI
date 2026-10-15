import tensorflow as tf
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import hashlib 
from sklearn.model_selection import train_test_split




class DatasetExplorer:
    def __init__(self, image_dataset):
        self.dataset_dir = Path(image_dataset)
        self.images = []
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
        images = []
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

                        self.images.append(image_path)
                    else:
                        print("images not found!")

        return min_width, min_height

    def problematics_data(self):

        hashes = {}
        duplicates = []
        not_duplicated = []
        for image_path in self.images:
            if image_path.is_file():
                fil_hash = hashlib.sha256(image_path.read_bytes()).hexdigest()

                if fil_hash in hashes:
                    duplicates.append((hashes[fil_hash], image_path))

                else:
                    hashes[fil_hash] = image_path
                    not_duplicated.append(image_path)

        return duplicates , not_duplicated , hashes

    def separation_data(self, data_images):

        labels = []
        temps_labels = []
        for image_path in data_images:
            labels.append(image_path.parent.name)
  
        train, temp_vals = train_test_split(data_images, 
                                                test_size=0.2,
                                                random_state=42,
                                                stratify=labels
                                            )

        for image_path in temp_vals:
            temps_labels.append(image_path.parent.name)  
        
        validation, test = train_test_split(
            temp_vals, 
            test_size=0.5,
            random_state=42,
            stratify=temps_labels
        )

        return train , validation, test

        
        



