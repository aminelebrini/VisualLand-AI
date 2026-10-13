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