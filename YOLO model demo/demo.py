from ultralytics import YOLO


model = YOLO('../visual_impairment_assistant/yolov8n.pt')
results = model.predict(source = 'animals.jpg',
                        show = True,
                        conf = 0.35,
                        save = True)