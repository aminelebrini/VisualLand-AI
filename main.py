from pathlib import Path
from src.DatasetExplorer import DatasetExplorer
from src.visualisation.Visualisation import Visualisation
from src.DataTraitement import DataTraitement
from models.CNN import CNN

if __name__ == "__main__":
    dataset_Root = Path("data/Intel_Image_Classification")
    train_dir = dataset_Root / "seg_train_test" / "seg_train"

    if not train_dir.exists():
        raise FileNotFoundError(f"Train directory not found: {train_dir}")
    else:
        for path in train_dir.iterdir():
            if path.is_dir():
                print("Directory:", path.name)
        dataset_explorer = DatasetExplorer(train_dir)
        class_and_distribution = dataset_explorer.identify_classes_and_distribution()
        # dataset_explorer.count_image_per_class()
        # dataset_explorer.data_image_visualisation()
        min_width, min_height = dataset_explorer.get_dimentions_format()
        duplicate, not_duplicated_images, hashes = dataset_explorer.problematics_data()
        class_count , image_distribution = class_and_distribution

        print(min_width, min_height)
        print(class_count)
        print(image_distribution)
        
        visualisation = Visualisation()
        visualisation.visualisation_distribution(image_distribution.keys(), image_distribution.values())
        visualisation.visualisation_count_image_per_classe(class_count.keys(), class_count.values())

        print("duplicated images display!")
        print("="*50)
        for original , duplicated in duplicate:
            print("\noriginal:", original, "and his duplicate:", duplicated)
        print("="*50)
        print("not duplicated images display!")
        print("="*50)
        for not_duplicated in not_duplicated_images:
            print("\nunique images examples:", not_duplicated)
        print("="*50)
        print("split data image to train and validation and test!")
        train , validation, test = dataset_explorer.separation_data(not_duplicated_images)
        print("="*50)

        print("="*50)
        print("pretretement de data !")
        print("="*50)

        width = min_width
        height = min_height
        data_traitement = DataTraitement()

        trained_data = data_traitement.redimensionnement(train)
        validation_data = data_traitement.redimensionnement(validation)
        test_data = data_traitement.redimensionnement(test)

        print("trained data resized : ", trained_data)
        print("\nvalidation data resized : ", validation_data)
        print("\ntest data resized : ", test_data)

        print("="*50)
        print("="*50)
        print("normalisation !")

        normlized_train_data = data_traitement.normalisation(trained_data)
        normlized_validation_data = data_traitement.normalisation(validation_data)
        normlized_test_data = data_traitement.normalisation(test_data)

        print("trained data normalized : ", normlized_train_data)
        print("\nvalidation data normalized : ", normlized_validation_data)
        print("\ntest data normalized : ", normlized_test_data)


        print("="*50)
        print("="*50)
        print("chargement du batch !")
        class_names = [
                    "buildings", "forest", "glacier",
                    "mountain", "sea", "street"
                ]
        trained_labels = []
        validation_labels = []
        test_labels = []
        for image_path in train:
            class_name = image_path.parent.name
            label = class_names.index(class_name)
            trained_labels.append(label)

        for image_path in validation:
            class_name = image_path.parent.name
            label = class_names.index(class_name)
            validation_labels.append(label)

        for image_path in test:
            class_name = image_path.parent.name
            label = class_names.index(class_name)
            test_labels.append(label)

        # print("trained labels : ", trained_labels)
        batched_trained_dataset = data_traitement.chagement_par_batch(normlized_train_data, trained_labels)
        batched_validation_dataset = data_traitement.chagement_par_batch(normlized_validation_data, validation_labels)
        batched_test_dataset = data_traitement.chagement_par_batch(normlized_test_data, test_labels)

        print("batched trained dataset : ", batched_trained_dataset)
        print("\nbatched validation dataset : ", batched_validation_dataset)
        print("\nbatched test dataset : ", batched_test_dataset)
        print("="*50)

        print("data augmentation !")

        augmented_trained_dataset = data_traitement.data_augmentation(batched_trained_dataset)

        print("augmented trained dataset : ", augmented_trained_dataset)

        visualisation.visualisation_image(augmented_trained_dataset)

        print("="*50)

        print("model building !")
        cnn_model = CNN()
        model = cnn_model.build_model()
        model.compile(
            optimizer='adam', 
            loss='sparse_categorical_crossentropy', 
            metrics=['accuracy'])

        model.summary()

        print("="*50)
        print("model training !")
        history = cnn_model.training(augmented_trained_dataset)
        print("model training history : ", history)