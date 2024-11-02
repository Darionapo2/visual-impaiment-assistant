import math
from ultralytics import YOLO
import cv2

model = YOLO('yolov8n.pt')

class_names = ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat',
              'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
              'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
              'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
              'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
              'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
              'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',  'sofa', 'pottedplant', 'bed',
              'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
              'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
              'teddy bear', 'hair drier', 'toothbrush'
              ]


font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1.5
color = (255, 0, 0)
thickness = 3

def predict(frame):
    return model.predict(source = frame,
                         conf = 0.25,
                         classes = [0, 2, 3, 5, 7, 11, 13, 15, 39, 56, 57, 59, 61, 63],
                         verbose = False)


def apply_bounding_boxes(results, frame, obstacles_list):
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)

            confidence = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])

            # Mostra la classe dell'oggetto
            org = (x1, y1)
            cv2.putText(frame, class_names[cls], org, font, font_scale, color, thickness)

            # Calcola il centro dell'oggetto
            center_object = [(x1 + x2) / 2, (y1 + y2) / 2]

            # Verifica se l'oggetto corrisponde a un ostacolo
            for obstacle in obstacles_list:
                if obstacle.is_near(center_object, 200):  # Usa una soglia per accoppiare
                    if obstacle.distance is not None:
                        # Mostra la distanza aggiornata dell'ostacolo a schermo
                        distance_text = f'distance: {obstacle.distance / 1000:.2f}m'
                        distance_org = (x1, y1 - 30)
                        cv2.putText(frame, distance_text, distance_org, font, font_scale, (0, 255, 0), thickness)
                    else:
                        print('distance not avaiable')

    return frame

def show(queue_rgb, queue_obstacles, stop_event):
    obstacles_list = []
    # loops = 200
    obj_video_writer = cv2.VideoWriter('obj.avi', cv2.VideoWriter_fourcc(*'XVID'), 5, (1280, 720))
    while not stop_event.is_set():
        if queue_rgb.qsize() >= 1:
            frame = queue_rgb.get()

            if not queue_obstacles.empty():
                obstacles_list = queue_obstacles.get()

            results = predict(frame)
            frame_with_boxes = apply_bounding_boxes(results, frame, obstacles_list)

            # obj_video_writer.write(frame_with_boxes)

            cv2.imshow('object detection', frame_with_boxes)

            '''
            i += 1
            if i % 60 == 0:
                j += 1
                cv2.imwrite(f'frame{j}.jpg', frame_with_boxes)
                print('I just saved a frame')
            '''
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event.set()
                break

            '''
            loops -= 1
            print('object loops:', loops)

            if loops == 0:
                break
            '''

    obj_video_writer.release()
    cv2.destroyAllWindows()




# 0 person
# 2 car
# 3 motorcycle
# 5 bus
# 6 train
# 7 truck
# 11 stop sign
# 13 bench
# 15 cat
# 16 dog
# 39 bottle
# 56 chair
# 57 couch
# 59 bed
# 61 dining table
# 63 laptop