import tensorflow as tf

def load_data():
    mnist = tf.keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    return (train_images, train_labels), (test_images, test_labels)

def preprocess_data(train_images, train_labels, test_images, test_labels):
    # Normalizacja danych - przeskalowanie wartości pikseli do zakresu [0,1]
    train_images = train_images.astype("float32") / 255.0
    test_images = test_images.astype("float32") / 255.0

    return train_images, train_labels, test_images, test_labels