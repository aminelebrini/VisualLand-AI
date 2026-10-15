import matplotlib.pyplot as plt
import seaborn as sbn
class Visualisation:

    def __init__(self):
        pass

    def visualisation_distribution(self, X, Y):
        plt.figure(figsize=(16,16))
        plt.bar(X, Y)
        plt.title("image distribution per classe")
        plt.xlabel("images classes")
        plt.ylabel("pourcentage of images")
        plt.show()

    def visualisation_count_image_per_classe(self, X, Y):

        plt.figure(figsize=(16,16))
        plt.bar(X, Y)
        plt.title("image distribution per classe")
        plt.xlabel("number of images per classe")
        plt.ylabel("classes")
        plt.show()


    def visualisation_image(self, data_images):

        for image, label in data_images.take(1):
            plt.figure(figsize=(16,16))
            for i in range(9):
                ax = plt.subplot(3, 3, i + 1)
                plt.imshow(image[i].numpy())
                plt.title(int(label[i]))
                plt.axis("off")

        plt.show()