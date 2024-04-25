import torch
import cv2
import IPython
from PIL import ImageColor
from ultralytics import YOLO


class ObjectDetection:
    def __init__(self, model_name='yolov3'):
        self.model_name = model_name
        self.model = self.load_model()
        self.classes = self.model.names
        self.device = 'cpu' 

    def load_model(self):
        if self.model_name != 'yolov8':
            model = torch.hub.load(f"ultralytics/{self.model_name}", 'custom', path=f"./weights/{self.model_name}_best.pt", force_reload=True)
        else:
            model = YOLO(f"./weights/{self.model_name}_best.pt")

        return model
    
    def score_frame(self, frame):
        results = self.model(frame)

        labels, conf, coord = results.xyxyn[0][:, -1], results.xyxyn[0][:, -2], results.xyxyn[0][:, :-1]
        return labels, conf, coord

    
    def v8_score_frame(self, frame):
        results = self.model(frame)

        labels = []
        confidences = []
        coords = []

        for result in results:
            boxes = result.boxes.cpu().numpy()
            
            label = boxes.cls
            conf = boxes.conf
            coord = boxes.xyxy

            labels.extend(label)
            confidences.extend(conf)
            coords.extend(coord)

        return labels, confidences, coords

    def get_coords(self, frame, row):

        if self.model_name != 'yolov8':
            x_shape, y_shape = frame.shape[1], frame.shape[0]
            return int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] * x_shape), int(row[3] * y_shape)
        else:
            return int(row[0]), int(row[1]), int(row[2]), int(row[3])
    
    def class_to_label(self, x):
        return self.classes[int(x)]
    
    def get_color(self, code):
        rgb = ImageColor.getcolor(code, "RGB")
        return rgb 
    
    def plot_bboxes(self, results, frame, threshold, box_color, text_color):
        labels, conf, coord = results
        n = len(labels)

        frame = frame.copy()
        box_color = self.get_color(box_color)
        text_color = self.get_color(text_color)

        for i in range(n):
            row = coord[i]
            if conf[i] >= threshold:
                x1, y1, x2, y2 = self.get_coords(frame, row)
                class_name = self.class_to_label(labels[i])

                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                cv2.putText(frame, f"{class_name} - {conf[i]*100:.2f}%", (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.5, text_color)
        
        return frame

