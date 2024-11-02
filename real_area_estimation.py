import numpy as np
import cv2
import matplotlib.pyplot as plt
import time

def main():
    rows = 640
    cols = 640

    # Creazione della matrice con un gradiente lineare
    matrix = np.zeros((rows, cols), dtype = np.int16)

    # Calcolo del passo del gradiente
    grad_step = 250 / rows

    # Riempimento della matrice con il gradiente lineare
    for i in range(rows):
        matrix[i, :] = np.arange(0, 250, grad_step)[:cols]

    # Mappatura dei valori della matrice a una scala di colori
    color_matrix = cv2.applyColorMap(matrix.astype(np.uint8), cv2.COLORMAP_JET)

    # Visualizzazione dell'immagine
    cv2.imshow('Gradient Image', color_matrix)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main2():
    # Dimensioni della matrice
    size = 640

    # Creazione della matrice con un gradiente centrale
    gradient_center = np.zeros((size, size), dtype = np.float32)
    center_x = size // 2
    center_y = size // 2

    # Definizione delle dimensioni del quadrato centrale e del gradiente lineare
    square_size = 200
    gradient_width = 20

    # Creazione del gradiente lineare all'interno del quadrato centrale
    start_x = center_x - square_size // 2
    end_x = center_x + square_size // 2
    start_y = center_y - square_size // 2
    end_y = center_y + square_size // 2
    for y in range(start_y, end_y):
        gradient_center[y, start_x:end_x] = np.linspace(0, 1, square_size)

    # Generazione della matrice di rumore casuale
    noise = np.random.rand(size, size)

    # Combina il gradiente centrale con il rumore casuale solo per le aree esterne
    result = noise.copy()
    result[start_y:end_y, start_x:end_x] = gradient_center[start_y:end_y, start_x:end_x]

    # Converto la matrice in un'immagine di 8 bit
    result_img = (result * 255).astype(np.uint8)

    # Applicazione dell'operatore Sobel per individuare i bordi

    a = time.time()
    sobel_x = cv2.Sobel(result_img, cv2.CV_64F, 1, 0, ksize = 1)
    sobel_y = cv2.Sobel(result_img, cv2.CV_64F, 0, 1, ksize = 1)
    sobel_mag = np.sqrt(sobel_y ** 2)
    b = time.time()

    print(b-a)

    cv2.imshow('Sobel Magnitude', sobel_mag.astype(np.uint8))

    _, sobel_threshold = cv2.threshold(sobel_mag, 100, 255, cv2.THRESH_BINARY)

    cv2.imshow('Sobel threshold', sobel_threshold.astype(np.uint8))



    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main2()