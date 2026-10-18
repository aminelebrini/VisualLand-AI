from pathlib import Path
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix
from PIL import Image
import hashlib 
import json

def get_visualize_load_normlize():
    data_image_path = Path("data/Intel_Image_Classification")
    data_images_train = data_image_path / "seg_train_test" / "seg_train"

    if not data_images_train.exists():
            raise FileNotFoundError(f"Train directory not found: {data_images_train}")
    else:
            count_image_per_class = {}
            images_class = {
                     "buildings",
                     "forest",
                     "glacier",
                     "mountain",
                     "sea",
                     "street"
                }
            for class_image_dir in data_images_train.iterdir():
                
                if class_image_dir.is_dir():
                    # print("Directory:", path.name)
                    n_image = 0
    
                    for images_path in class_image_dir.iterdir():
                        if images_path.is_file():
                            n_image += 1

                    count_image_per_class[class_image_dir.name] = n_image

            count_total = 0
            for class_name in images_class:
                count_total += count_image_per_class.get(class_name, 0)

            image_distribution = {}

            for class_name in sorted(images_class):
                n_image = count_image_per_class.get(class_name, 0)
                  
                if count_total > 0:
                    image_distribution[class_name] = (n_image / count_total) * 100
                else:
                    image_distribution[class_name] = 0

            print("total images:", count_total)
            print("image distribution:")
            for class_name in sorted(images_class):
                print(f"distribution des images: {image_distribution[class_name]:.2f}")    

            plt.figure(figsize=(10, 6))
            plt.bar(image_distribution.keys(), image_distribution.values())
            plt.xlabel("Classes")
            plt.ylabel("Distribution (%)")
            plt.title("Distribution of Images per Class")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

            plt.figure(figsize=(10, 6))
            plt.bar(count_image_per_class.keys(), count_image_per_class.values())
            plt.xlabel("Classes")
            plt.ylabel("Number of Images")
            plt.title("Number of Images per Class")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

            print("="*50)
            print("images dimension and format !")
            with Image.open(next(data_images_train.glob("**/*.jpg"))) as img:
                width, height = img.size
                format = img.format
                print(f"Image dimensions: {width}x{height}, Format: {format}")

            print("="*50)
            print("images hash and check duplicates!")

            not_duplicate_images = []
            duplicates = []
            haches = {}
            for class_image_dir in data_images_train.iterdir():
                 if class_image_dir.is_dir():
                    for image_path in class_image_dir.iterdir():
                        if image_path.is_file():
                            file_hash = hashlib.md5(image_path.read_bytes()).hexdigest()

                            if file_hash in haches:
                                duplicates.append([image_path, haches[file_hash]])    
                            else:
                                haches[file_hash] = image_path
                                not_duplicate_images.append(image_path)

            print(f"Number of duplicate images: {len(duplicates)}")
            print(f"Number of unique images: {len(not_duplicate_images)}")

            labels = [image_path.parent.name for image_path in not_duplicate_images]
            train , tem_data = train_test_split(not_duplicate_images, test_size=0.2, random_state=42, stratify=labels)
            print(f"Number of training images: {len(train)}")
            temp_labels = [image_path.parent.name for image_path in tem_data]
            validation , test = train_test_split(tem_data, test_size=0.5, random_state=42, stratify=temp_labels)
            print(f"Number of validation images: {len(validation)}")
            print(f"Number of test images: {len(test)}")

            train_paths = [str(image_path) for image_path in train]
            validation_paths = [str(image_path) for image_path in validation]
            test_paths = [str(image_path) for image_path in test]

            class_names = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
            train_dataset = [class_names.index(image_path.parent.name) for image_path in train]
            validation_dataset = [class_names.index(image_path.parent.name) for image_path in validation]
            test_dataset = [class_names.index(image_path.parent.name) for image_path in test]

            batched_train_dataset = tf.data.Dataset.from_tensor_slices((train_paths, train_dataset))
            batched_validation_dataset = tf.data.Dataset.from_tensor_slices((validation_paths, validation_dataset))
            batched_test_dataset = tf.data.Dataset.from_tensor_slices((test_paths, test_dataset))

            # resized_train_dataset = []
            # resized_train_label = []

            # resized_validation_dataset = []
            # resized_validation_label = []

            # for image_path, label in batched_train_dataset:
            #     image = tf.io.read_file(image_path)
            #     image = tf.image.decode_jpeg(image, channels=3)
            #     image = tf.image.resize(image, (256, 256))
            #     image = tf.cast(image, tf.float32) / 255.0
            #     resized_train_dataset.append(image)
            #     resized_train_label.append(label)

            # for image_path, label in val_batched_dataset:
            #     image = tf.io.read_file(image_path)
            #     image = tf.image.decode_jpeg(image, channels=3)
            #     image = tf.image.resize(image, (256, 256))
            #     image = tf.cast(image, tf.float32) / 255.0
            #     resized_validation_dataset.append(image)
            #     resized_validation_label.append(label)


            def load_data_process(path, labels):
                image = tf.io.read_file(path)
                image = tf.image.decode_jpeg(image, channels=3)
                image = tf.image.resize(image, (256,256))
                image = tf.cast(image, tf.float32) / 255.0

                return image , labels

            batched_train_dataset = batched_train_dataset.map(load_data_process, num_parallel_calls=tf.data.AUTOTUNE)
            batched_validation_dataset = batched_validation_dataset.map(load_data_process, num_parallel_calls=tf.data.AUTOTUNE)
            batched_test_dataset = batched_test_dataset.map(load_data_process, num_parallel_calls=tf.data.AUTOTUNE)

            data_augmentation = tf.keras.Sequential([
                tf.keras.layers.RandomFlip("horizontal_and_vertical"),
                tf.keras.layers.RandomRotation(0.2),
                tf.keras.layers.RandomZoom(0.2),
            ])

            batched_train_dataset = batched_train_dataset.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)

            batched_train_dataset = batched_train_dataset.shuffle(buffer_size=1000).batch(32).prefetch(tf.data.AUTOTUNE)
            batched_test_dataset = batched_test_dataset.batch(32).prefetch(tf.data.AUTOTUNE)
            batched_validation_dataset = batched_validation_dataset.batch(32).prefetch(tf.data.AUTOTUNE)

            return batched_train_dataset, batched_validation_dataset, batched_test_dataset


def build_cnn_model_train(train_ds, val_ds):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, ((3,3)), activation='relu', input_shape=(256, 256, 3)),
        tf.keras.layers.MaxPooling2D((2,2), strides=2),


        tf.keras.layers.Conv2D(64,(3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2,2), strides=2),
        
        
        tf.keras.layers.Conv2D(128,(3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2,2), strides=2),

        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(6, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    print("Model training!")

    early_stoping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=6,
        restore_best_weights=True
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        callbacks=[early_stoping]
    )

    print("model training completed !")

    history_data = history.history

    with open("history_data_model.json", "w") as file:
        json.dump(history_data, file)

    print("Training history saved to training_history.json successfully!")

train_ds, val_ds, test_ds= get_visualize_load_normlize()

# build_cnn_model_train(train_ds=train_ds, val_ds=val_ds)


with open("results/history_data_model.json", 'r') as file:
    history = json.load(file)

plt.figure(figsize=(16,6))
plt.plot(history['accuracy'], label='Training Accuracy')
plt.plot(history['loss'], label='Training Loss')
plt.title('Training Accuracy and Loss')
plt.xlabel('Epochs')
plt.ylabel('Accuracy / Loss')
plt.legend()
plt.grid()
plt.show()


plt.figure(figsize=(16,6))
plt.plot(history['val_accuracy'], label='Validation Accuracy')
plt.plot(history['val_loss'], label='Validation Loss')
plt.title('Training Accuracy and Loss')
plt.xlabel('Epochs')
plt.ylabel('Accuracy / Loss')
plt.legend()
plt.grid()
plt.show()


def prediction(train_ds):

    y_pred = []
    y_true = []
    model = tf.keras.models.load_model("models/intel_image_classifier_model.keras")

    for images, labels in train_ds:
        predictions = model.predict(images)
        predicted_labels = tf.argmax(predictions, axis=1)
        y_pred.extend(predicted_labels.numpy())
        y_true.extend(labels.numpy())

    recall = recall_score(y_true, y_pred, average='weighted')
    precision = precision_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    confusion_matrix_result = confusion_matrix(y_true, y_pred)

    print("Recall:", recall)
    print("Precision:", precision)
    print("F1 Score:", f1)
    print("Confusion Matrix:")
    print(confusion_matrix_result)


# prediction(train_ds)

def predict_single_image(image_path):

    class_names = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
    model = tf.keras.models.load_model("models/intel_image_classifier_model.keras")

    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (256, 256))
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.expand_dims(image, axis=0)

    predictions = model.predict(image)
    # print(predictions)

    predicted_label = tf.argmax(predictions[0]).numpy()
    predicted_class_name = class_names[predicted_label]
    confidence_score = float(tf.reduce_max(predictions[0])) * 100
    print(f"Predicted class: {predicted_class_name}, Confidence: {confidence_score:.2f}%")


    return predicted_class_name, confidence_score


predicted_class, confidence = predict_single_image("data/MIT-Earth-Dish-Press.jpg")



    


    