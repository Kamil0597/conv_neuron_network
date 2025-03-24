from NeuralNetworkModel import NeuralNetworkModel
from filters import filters_with_names
from loadingFunctions import load_data, preprocess_data
from viewData import viewRandom10


def main():

    (train_images, train_labels), (test_images, test_labels) = load_data()

    train_images, train_labels, test_images, test_labels = preprocess_data(train_images, train_labels, test_images, test_labels)


    nn = NeuralNetworkModel(filters_with_names)

    #viewRandom10(nn, train_images, train_labels)

    nn.train(train_images[:1000], train_labels[:1000], epochs=30, learning_rate=0.001)

    nn.test_model(test_images[:1000], test_labels[:1000])


if __name__ == "__main__":
    main()
