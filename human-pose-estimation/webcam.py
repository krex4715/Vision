import torchvision
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from torchvision import transforms as T
from PIL import Image



COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


def get_inference(img):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True) 
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    

    img_tensor = torch.tensor(np.asarray(img)).permute(2,0,1).unsqueeze(0)
    img_tensor = img_tensor.float()/255.0 
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        pred = model(img_tensor)

    pred_indices = [i for i, x in enumerate(pred[0]['labels'] )]
    pred_boxes = pred[0]['boxes'][pred_indices].int().cpu().numpy()
    pred_score = list(pred[0]['scores'][pred_indices].cpu().numpy())

    return pred_indices, pred_boxes, pred_score


def object_detector(img, threshold):
    indices, boxes, scores = get_inference(img)

    for i in range(len(boxes)):
        if scores[i] > threshold:
            class_idx = indices[i]
            class_name = COCO_INSTANCE_CATEGORY_NAMES[class_idx]
            print('Class:', class_name)
            print('Bounding box:', boxes[i])
            print('Score:', scores[i])
            x, y, w, h = boxes[i]

            img = cv2.rectangle(img, (x, y), (x + w, y + h), (10, 255, 0), 3)
            cv2.putText(img, class_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (10, 255, 0), 3)

    cv2.imshow('video', img)







if __name__ == "__main__":
    webcam = cv2.VideoCapture(cv2.CAP_DSHOW+1)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("Using {} device".format(device))
    


    if not webcam.isOpened():
        print("Could not open webcam")
        exit()

    while webcam.isOpened():
        status, frame = webcam.read()

        if status:
            img = cv2.cvtColor(frame, cv2.IMREAD_COLOR)
            object_detector(img, 0.9)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    webcam.release()
    cv2.destroyAllWindows()