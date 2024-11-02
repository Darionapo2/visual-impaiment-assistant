import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import data

# Carica l'immagine
# image = cv2.imread('testimage.jpg')


image_rgb = data.astronaut()  # Utilizziamo l'immagine astronauta come esempio

# image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Inizializza il selettore di regioni per la Selective Search
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

# Setta l'immagine su cui fare la Selective Search
ss.setBaseImage(image_rgb)

# Scegli la modalità rapida o accurata
ss.switchToSelectiveSearchFast()  # Per una ricerca veloce
# ss.switchToSelectiveSearchQuality()  # Per una ricerca accurata

print('ciao cioa')
# Esegui la Selective Search
rects = ss.process()

print('fatto')

# Visualizza le regioni proposte (limitiamo a 100 per non sovraccaricare l'immagine)
output_image = image_rgb.copy()
for i, rect in enumerate(rects[:100]):  # Mostra solo le prime 100 regioni
    x, y, w, h = rect
    cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

plt.imshow(output_image)
plt.axis('off')
plt.savefig('region_proposal_astronaut')
plt.show()

# Ora, per mostrare l'immagine segmentata
# Utilizza un'algoritmo di segmentazione (tipo SLIC) per ottenere un'immagine segmentata
segmenter = cv2.ximgproc.createSuperpixelSLIC(image, algorithm=cv2.ximgproc.SLICO, region_size=30, ruler=10.0)
segmenter.iterate(10)  # Numero di iterazioni

# Recupera le maschere segmentate
mask = segmenter.getLabelContourMask()

# Mostra l'immagine segmentata
segmented_image = cv2.bitwise_and(image_rgb, image_rgb, mask=cv2.bitwise_not(mask))

plt.figure(figsize=(10, 10))
plt.title("Segmented Image")
plt.imshow(segmented_image)
plt.axis('off')
plt.show()
