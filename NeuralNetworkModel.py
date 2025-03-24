import numpy as np
from filters import *
import matplotlib.pyplot as plt

class NeuralNetworkModel:
    def __init__(self, filters, stride=1, padding=0):

        self.filters = np.array(filters[0])
        self.filters_names = filters[1]
        self.num_filters = len(filters)
        self.kernel_size = self.filters[0].shape[0]
        self.stride = stride
        self.padding = padding

        flatten_size = self.calculate_flatten_size(28, kernel_size=3, stride=1, padding=0, pool_size=2, num_layers=2)


        self.fc1_weights = np.random.randn(128, flatten_size) * 0.1
        self.fc1_biases = np.zeros(128)

        self.fc2_weights = np.random.randn(10, 128) * 0.1
        self.fc2_biases = np.zeros(10)

    def convolve2D(self, image, kernel):
        if self.padding > 0:
            image = np.pad(image, ((self.padding, self.padding), (self.padding, self.padding)), mode='constant',
                           constant_values=0)
        img_h, img_w = image.shape
        kernel_h, kernel_w = kernel.shape

        output_h = (img_h - kernel_h) // self.stride + 1
        output_w = (img_w - kernel_w) // self.stride + 1

        output = np.zeros((output_h, output_w))

        for y in range(0, output_h):
            for x in range(0, output_w):
                region = image[y * self.stride:y * self.stride + kernel_h, x * self.stride:x * self.stride + kernel_w]
                output[y, x] = np.sum(region * kernel)

        return output

    def calculate_flatten_size(self, input_size, kernel_size, stride, padding, pool_size, num_layers):

        h, w = input_size, input_size

        for _ in range(num_layers):  #Iteracja przez każdą warstwę konwolucyjną
            h = (h - kernel_size + 2 * padding) // stride + 1  #Konwolucja
            w = (w - kernel_size + 2 * padding) // stride + 1

            h //= pool_size  # Pooling
            w //= pool_size

        return h * w * len(self.filters)

    def relu(self, x):
        return np.maximum(0, x)


    def flatten(self, feature_map):
        return feature_map.flatten()

    def fully_connected(self, input_vector, weights, biases):
        return np.dot(weights, input_vector) + biases

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

    def cross_entropy_loss(self, predicted, actual):

        return -np.sum(actual * np.log(predicted + 1e-9))

    def batch_norm(self, x):
        mean = np.mean(x)
        var = np.var(x)
        return (x - mean) / np.sqrt(var + 1e-8)

    def forward(self, input_image):
        output = []
        viewOutput = []
        self.input_image = input_image
        for filter in self.filters:
            firstConvolve = self.convolve2D(input_image, filter)
            firstConvolveFirst = self.relu(self.batch_norm(firstConvolve))
            maxPooling = self.max_pooling(firstConvolveFirst)
            secondConvolve = self.convolve2D(maxPooling, filter)
            maxPooling2 = self.max_pooling(secondConvolve)
            flat = self.flatten(maxPooling2)
            output.append(flat)
            viewOutput.append(maxPooling2)

        merged_output = np.concatenate(output)

        fc1 = self.relu(self.fully_connected(merged_output, self.fc1_weights, self.fc1_biases))

        fc2 = self.softmax(self.fully_connected(fc1, self.fc2_weights, self.fc2_biases))

        #output = [self.max_pooling(self.relu(self.convolve2D(input_image, f))) for f in self.filters]
        return output, viewOutput ,self.filters_names, fc2


    def forward2(self, input_image):

        output = []
        self.input_image = input_image
        for filter in self.filters:
            firstConvolve = self.convolve2D(input_image, filter)
            firstConvolveFirst = self.relu(firstConvolve)
            maxPooling = self.max_pooling(firstConvolveFirst)
            secondConvolve = self.convolve2D(maxPooling, filter)
            maxPooling2 = self.max_pooling(secondConvolve)
            flat = self.flatten(maxPooling2)
            output.append(flat)

        merged_output = np.concatenate(output)


        self.flattened_output = merged_output
        self.fc1_output = self.relu(self.fully_connected(merged_output, self.fc1_weights, self.fc1_biases))


        self.fc2_output = self.fully_connected(self.fc1_output, self.fc2_weights, self.fc2_biases)
        fc2 = self.softmax(self.fc2_output)

        return fc2

    def backpropagation(self, predicted, actual, learning_rate=0.01):

        error = predicted - actual  # Pochodna Cross-Entropy + Softmax

        # Gradient dla FC2
        grad_fc2_weights = np.outer(error, self.fc1_output)
        grad_fc2_biases = error

        # Propagacja błędu
        error_fc1 = np.dot(self.fc2_weights.T, error) * (self.fc1_output > 0)

        grad_fc1_weights = np.outer(error_fc1, self.flattened_output)
        grad_fc1_biases = error_fc1

        # Aktualizacja wag i biasów
        self.fc2_weights -= learning_rate * grad_fc2_weights
        self.fc2_biases -= learning_rate * grad_fc2_biases
        self.fc1_weights -= learning_rate * grad_fc1_weights
        self.fc1_biases -= learning_rate * grad_fc1_biases

    def train(self, images, labels, epochs=10, learning_rate=0.01):
        for epoch in range(epochs):
            total_loss = 0

            for i, (img, label) in enumerate(zip(images, labels)):
                # print(f"\n Próbka {i + 1}/{len(images)}")
                # print(f"  - Rozmiar pojedynczego obrazu: {img.shape}")
                # print(f"  - Typ etykiety (liczba): {label}")

                # One-hot encoding dla etykiety
                actual = np.zeros(10)
                actual[label] = 1
                #print(f"  - One-hot encoding (actual): {actual.shape}, Wartość: {actual}")

                # Forward pass
                predicted = self.forward2(img)
                #print("Softmax output:", predicted)
                #print("Suma softmax:", np.sum(predicted))
                #print(f"  - Rozmiar predykcji: {predicted.shape}, Przykładowe wartości: {predicted[:5]}")  # Pierwsze 5 wartości predykcji

                # Obliczenie straty
                loss = self.cross_entropy_loss(predicted, actual)
                total_loss += loss

                # Backpropagation + aktualizacja wag
                self.backpropagation(predicted, actual, learning_rate)

            print(f"\n Epoka {epoch + 1}/{epochs}, Strata: {total_loss / len(images):.4f}")

    def test_model(self, images, labels, num_images=10):

        correct = 0
        total = len(images)
        misclassified = []

        for i, (img, label) in enumerate(zip(images, labels)):
            predicted = self.forward2(img)  # Forward pass

            #print(f"🔍 Przewidywane wartości shape: {type(predicted)}, {predicted.shape}")  # Debugging
            predicted_class = np.argmax(predicted)

            if predicted_class == label:
                correct += 1
            else:
                misclassified.append((img, label, predicted_class))  # Zapisujemy błędnie zaklasyfikowane obrazy

        accuracy = correct / total
        print(f"\nDokładność modelu na zbiorze testowym: {accuracy:.2%}")

        # Wyświetlenie kilku błędnie zaklasyfikowanych obrazów
        num_images = min(num_images, len(misclassified))  # Żeby nie wyjść poza zakres

        if num_images > 0:
            fig, axes = plt.subplots(1, num_images, figsize=(num_images * 2, 2))

            for ax, (img, actual, predicted) in zip(axes, misclassified[:num_images]):
                ax.imshow(img, cmap='gray')
                ax.set_title(f"Pred: {predicted}\nActual: {actual}", fontsize=10)
                ax.axis("off")

            plt.show()
        else:
            print("Brak błędnie sklasyfikowanych obrazów")

    def max_pooling(self, feature_map, pool_size=2, stride=2):

        h, w = feature_map.shape
        output_h = (h - pool_size) // stride + 1
        output_w = (w - pool_size) // stride + 1

        pooled_output = np.zeros((output_h, output_w))

        for i in range(0, output_h):
            for j in range(0, output_w):
                region = feature_map[i * stride:i * stride + pool_size, j * stride:j * stride + pool_size]
                pooled_output[i, j] = np.max(region)

        return pooled_output



