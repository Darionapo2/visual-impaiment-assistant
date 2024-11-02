import cv2
import matplotlib.pyplot as plt
from skimage import data

# Carica l'immagine astronauta
image = data.astronaut()

# Converte l'immagine in formato BGR per OpenCV (OpenCV usa BGR invece di RGB)
image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

# Inizializza il selettore di regioni per la Selective Search
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
ss.setBaseImage(image_bgr)

# Usa la modalità veloce per la Selective Search
ss.switchToSelectiveSearchFast()

# Esegui la Selective Search
rects = ss.process()

# Crea una copia dell'immagine originale per disegnare i rettangoli
output_image = image.copy()

# Aggiungi le regioni proposte (limitiamo a 100 per non sovraccaricare l'immagine)
for i, rect in enumerate(rects[:100]):  # Mostra solo le prime 100 regioni
    x, y, w, h = rect
    cv2.rectangle(output_image, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Colore blu per i bordi

# Salva l'immagine con i region proposal senza modificare la dimensione
plt.imsave('astronaut_with_region_proposals.png', output_image)

# Mostra l'immagine
plt.figure(figsize=(10, 10))
plt.title("Astronauta con Region Proposals (512x512)")
plt.imshow(output_image)
plt.axis('off')
plt.show()
