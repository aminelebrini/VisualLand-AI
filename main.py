from pathlib import Path
from src.DatasetExplorer import DatasetExplorer
from src.visualisation.Visualisation import Visualisation

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

        class_count , image_distribution = class_and_distribution

        print(min_width, min_height)
        print(class_count)
        print(image_distribution)

        visualisation = Visualisation()
        visualisation.visualisation_distribution(image_distribution.keys(), image_distribution.values())
        visualisation.visualisation_count_image_per_classe(class_count.keys(), class_count.values())
        

    
