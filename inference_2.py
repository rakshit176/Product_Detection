from ultralytics import YOLO
import  cv2
from ultralytics.yolo.utils.plotting import Annotator
import glob
import json
import os 

class Inference():
    def __init__(self):
        self.weights_path = 'best_yolov8.pt'
        self.image_size = 1280
        self.common_confidence = 0.35
        self.common_iou = 0.45
        self.defects = ['person'] 
        self.ind_thresh = {'person':0.2} 
        self.iou = 0.1
        self.max_det = 100
        self.hide_labels = False
        self.hide_conf = True
        self.line_width = 2

    def load_model(self):
        model = YOLO(self.weights_path) 
        return model

    def get_inferenece(self,model,img):
        annotator = Annotator(img,line_width=self.line_width)
        ind_thresh=self.ind_thresh
        results = model.predict(source=img,conf=self.common_confidence,imgsz=self.image_size,iou=self.iou,max_det=self.max_det)
        boxes = results[0].boxes
        predicted_image = results[0].plot()
        coordinates = []
        detector_labels = []
        for box in boxes:
            cords = box.xyxy[0].tolist()
            label_cls = int(box.cls[0].item())
            label = model.names[label_cls]
            xmin,ymin,xmax,ymax = int(cords[0]),int(cords[1]),int(cords[2]),int(cords[3])
            if model.names[int(box.cls)] in list(ind_thresh.keys()):
                if float(box.conf) > ind_thresh[model.names[int(box.cls)]]:
                    label = None if self.hide_labels else (model.names[int(box.cls)] if self.hide_conf else f'{model.names[int(box.cls)]} {float(box.conf):.2f}')
                    annotator.box_label(box.xyxy[0],label,color=(0,0,255))
                    detector_labels.append(label)
                    coordinates.append({label:[xmin,ymin,xmax,ymax]})
            else:
                label = None if self.hide_labels else (model.names[int(box.cls)] if self.hide_conf else f'{model.names[int(box.cls)]} {float(box.conf):.2f}')
                annotator.box_label(box.xyxy[0],label,color=(0,0,255))
                detector_labels.append(label)
                coordinates.append({label:[xmin,ymin,xmax,ymax]})
        return img, detector_labels, coordinates


inf = Inference()
model = inf.load_model()

count_dict = {}
prediction_dict = {'image_name':[],'xmin': [],'ymin': [],'xmax': [],'ymax': [],'class_id': [],}
for path in glob.glob(r'D:\BERT_LABS\Train_data\test\\*.JPG'):
    frame = cv2.imread(path)
    src = frame.copy()
    file_name = os.path.basename(path)
    pre_img, detector_labels, coordinates = inf.get_inferenece(model,frame)
    cv2.imwrite('./prediction2/'+file_name,pre_img)
    if len(detector_labels) < 4:
        image = cv2.rotate(src, cv2.ROTATE_180)
        pre_img, detector_labels, coordinates = inf.get_inferenece(model,image)
        cv2.imwrite('./prediction2/'+file_name,pre_img)
    count_dict[file_name] = len(detector_labels)
    result_list = [value for data_dict in coordinates for key, value in data_dict.items()]
    for (results,labels) in zip(result_list,detector_labels):
        prediction_dict['image_name'].append(file_name)
        prediction_dict['xmin'].append(results[0])
        prediction_dict['ymin'].append(results[1])
        prediction_dict['xmax'].append(results[2])
        prediction_dict['ymax'].append(results[3])
        prediction_dict['class_id'].append(labels)

json_file_path = "image2products_2.json"

with open(json_file_path, 'w') as json_file:
    json.dump(count_dict, json_file, indent=4)
    
with open('results.json', 'w') as json_file:
    json.dump(prediction_dict, json_file, indent=4)

print(f"The dictionary has been saved to {json_file_path}") 