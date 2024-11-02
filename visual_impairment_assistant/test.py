import numpy as np
import cv2

# Crea un VideoWriter per il test
test_writer = cv2.VideoWriter('test.avi', cv2.VideoWriter_fourcc(*'XVID'), 10, (1280, 720))

# Genera un frame di test con un cerchio disegnato
for i in range(100):
    test_frame = np.zeros((720, 1280, 3), dtype = np.uint8)  # Frame nero
    cv2.circle(test_frame, (640, 360), 50, (0, 255, 0), -1)  # Cerchio verde al centro
    cv2.imshow('Test Frame', test_frame)
    test_writer.write(test_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

test_writer.release()
cv2.destroyAllWindows()