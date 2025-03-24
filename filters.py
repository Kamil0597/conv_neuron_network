import numpy as np
# Filtry wykrywające krawędzie, wyostrzające, itp.
sobel_x = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
])

sobel_y = np.array([
    [-1, -2, -1],
    [ 0,  0,  0],
    [ 1,  2,  1]
])

laplacian = np.array([
    [ 0,  1,  0],
    [ 1, -4,  1],
    [ 0,  1,  0]
])

sharpen = np.array([
    [ 0, -1,  0],
    [-1,  5, -1],
    [ 0, -1,  0]
])

diagonal_edge_1 = np.array([
    [-2, -1, 0],
    [-1,  0, 1],
    [ 0,  1, 2]
])

diagonal_edge_2 = np.array([
    [ 0,  1,  2],
    [ -1, 0,  1],
    [ -2, -1, 0]
])

blur = np.array([
    [1/9, 1/9, 1/9],
    [1/9, 1/9, 1/9],
    [1/9, 1/9, 1/9]
])

prewitt_x = np.array([
    [-1,  0,  1],
    [-1,  0,  1],
    [-1,  0,  1]
])

prewitt_y = np.array([
    [-1, -1, -1],
    [ 0,  0,  0],
    [ 1,  1,  1]
])

gaussian_blur = np.array([
    [1/16, 2/16, 1/16],
    [2/16, 4/16, 2/16],
    [1/16, 2/16, 1/16]
])

high_pass = np.array([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
])

emboss = np.array([
    [-2, -1,  0],
    [-1,  1,  1],
    [ 0,  1,  2]
])

filters = [
    sobel_x,  # Wykrywanie pionowych krawędzi
    sobel_y,  # Wykrywanie poziomych krawędzi
    laplacian,  # Wykrywanie konturów
    sharpen,  # Wyostrzanie cech
    prewitt_x  # wykrywanie pionowych krawędzi #2
]

filter_names = [
    "Sobel X",
    "Sobel Y",
    "Laplacian",
    "Sharpen",
    "Prewitt X"
]



filters_with_names = [filters, filter_names]

