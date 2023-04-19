import torchvision
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms as T
from torchvision import models
from PIL import Image




IMG_SIZE = 480

COLORS = np.array([
    (0, 0, 0),       # 0=background
    (128, 0, 0),     # 1=aeroplane
    (0, 128, 0),     # 2=bicycle
    (128, 128, 0),   # 3=bird
    (0, 0, 128),     # 4=boat
    (128, 0, 128),   # 5=bottle
    (0, 128, 128),   # 6=bus
    (128, 128, 128), # 7=car
    (255, 255, 255), # 8=cat
    (192, 0, 0),     # 9=chair
    (64, 128, 0),    # 10=cow
    (192, 128, 0),   # 11=dining table
    (64, 0, 128),    # 12=dog
    (192, 0, 128),   # 13=horse
    (64, 128, 128),  # 14=motorbike
    (192, 128, 128), # 15=person
    (0, 64, 0),      # 16=potted plant
    (128, 64, 0),    # 17=sheep
    (0, 192, 0),     # 18=sofa
    (128, 192, 0),   # 19=train
    (0, 64, 128)     # 20=tv/monitor
])



def seg_map(img, n_classes=21):
    rgb = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    for c in range(n_classes):
        idx = img == c

        rgb[idx] = COLORS[c]

    return rgb




if __name__ == "__main__":
    webcam = cv2.VideoCapture(cv2.CAP_DSHOW+1)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("Using {} device".format(device))
    
    
    deeplab = models.segmentation.deeplabv3_resnet101(pretrained=True).to(device).eval()
    trf = T.Compose([
                        T.Resize(IMG_SIZE),
                    #     T.CenterCrop(IMG_SIZE), # make square image
                        T.ToTensor(), 
                        T.Normalize(
                            mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]
                        )
                    ])

    if not webcam.isOpened():
        print("Could not open webcam")
        exit()

    while webcam.isOpened():
        status, frame = webcam.read()

        if status:
            img = cv2.cvtColor(frame, cv2.IMREAD_COLOR)
            img = Image.fromarray(img)
            input_img = trf(img).unsqueeze(0)
            input_img = input_img.to(device)
            out = deeplab(input_img)['out']
            out = torch.argmax(out.squeeze(), dim=0)
            out = out.detach().cpu().numpy()
            out_seg = seg_map(out)
            original_img = Image.fromarray(np.array(img))
            original = np.array(original_img.resize([out_seg.shape[1],out_seg.shape[0]]))
            overlap = 0.3*original + 0.7*out_seg
            seg_result = Image.fromarray(overlap.astype(np.uint8))
            
            cv2.imshow('video', np.array(img))
            cv2.imshow('segmentation', np.array(seg_result))


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    webcam.release()