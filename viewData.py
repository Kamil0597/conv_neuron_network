import matplotlib.pyplot as plt
import numpy as np

def viewRandom10(model, images, labels):
    random_indices = np.random.choice(len(images), 10, replace=False)

    for idx in random_indices:
        image = images[idx]
        label = labels[idx]

        # Przepuszczenie obrazu przez sieć neuronową
        processed_images, view_images, filter_names, fc2 = model.forward(image)

        num_filters = len(processed_images)

        # Wymiary siatki
        rows = (num_filters // 4) + (1 if num_filters % 4 != 0 else 0)

        fig, axes = plt.subplots(rows + 1, 4, figsize=(10, 2 * (rows + 1)))

        # Oryginalny obraz
        axes[0, 0].imshow(image, cmap='gray')
        axes[0, 0].set_title(f"Label: {label}", fontsize=10)
        axes[0, 0].axis('off')


        for j in range(1, 4):
            axes[0, j].axis('off')

        # Przetworzone obrazy w siatce
        for i in range(rows):
            for j in range(4):
                idx = i * 4 + j
                if idx < num_filters:
                    axes[i + 1, j].imshow(view_images[idx], cmap='gray')
                    axes[i + 1, j].set_title(f"{filter_names[idx]}", fontsize=8)
                    axes[i + 1, j].axis('off')
                else:
                    axes[i + 1, j].axis('off')

        plt.tight_layout()
        plt.show()

