import matplotlib.pyplot as plt
from skimage import io
from skimage.segmentation import felzenszwalb
from skimage.color import rgb2gray, label2rgb
import numpy as np

# Carica l'immagine
image = io.imread('image2.jpg')

# Esegui l'algoritmo Felzenszwalb per segmentare l'immagine
# Nota che puoi modificare scale, sigma e min_size per affinare il risultato
segments = felzenszwalb(image, scale = 100, sigma = 0.5, min_size = 50)

# Visualizza il numero di segmenti trovati
print(f"Numero di segmenti: {len(np.unique(segments))}")

# Visualizza l'immagine segmentata con i contorni dei segmenti
segmented_image = label2rgb(segments, image, kind='avg')

# Plot dell'immagine originale e di quella segmentata
fig, ax = plt.subplots(1, 2, figsize=(15, 10))
ax[0].imshow(image)
ax[0].set_title("Immagine Originale")
ax[0].axis('off')

ax[1].imshow(segmented_image)
ax[1].set_title("Immagine Segmentata (Felzenszwalb)")
ax[1].axis('off')

plt.show()