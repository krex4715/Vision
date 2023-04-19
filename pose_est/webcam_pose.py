import torchvision
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from torchvision import transforms as T




def draw_skeleton_per_person(img, all_keypoints, all_scores, confs, keypoint_threshold=2, conf_threshold=0.9,linewith=2):
    
    # initialize a set of colors from the rainbow spectrum
    cmap = plt.get_cmap('rainbow')
    # create a copy of the image
    img_copy = img.copy()
    # check if the keypoints are detected
    if len(output["keypoints"])>0:
      # pick a set of N color-ids from the spectrum
      colors = np.arange(1,255, 255//len(all_keypoints)).tolist()[::-1]
      # iterate for every person detected
      for person_id in range(len(all_keypoints)):
          # check the confidence score of the detected person
          if confs[person_id]>conf_threshold:
            # grab the keypoint-locations for the detected person
            keypoints = all_keypoints[person_id, ...]

            # iterate for every limb 
            for limb_id in range(len(limbs)):
              # pick the start-point of the limb
              limb_loc1 = keypoints[limbs[limb_id][0], :2].detach().cpu().numpy().astype(np.int32)
              # pick the start-point of the limb
              limb_loc2 = keypoints[limbs[limb_id][1], :2].detach().cpu().numpy().astype(np.int32)
              # consider limb-confidence score as the minimum keypoint score among the two keypoint scores
              limb_score = min(all_scores[person_id, limbs[limb_id][0]], all_scores[person_id, limbs[limb_id][1]])
              # check if limb-score is greater than threshold
              if limb_score> keypoint_threshold:
                # pick the color at a specific color-id
                color = tuple(np.asarray(cmap(colors[person_id])[:-1])*255)
                # draw the line for the limb
                cv2.line(img_copy, tuple(limb_loc1), tuple(limb_loc2), color, linewith)

    return img_copy





def get_limbs_from_keypoints(keypoints):
    limbs = [       
        [keypoints.index('right_eye'), keypoints.index('nose')],
        [keypoints.index('right_eye'), keypoints.index('right_ear')],
        [keypoints.index('left_eye'), keypoints.index('nose')],
        [keypoints.index('left_eye'), keypoints.index('left_ear')],
        [keypoints.index('right_shoulder'), keypoints.index('right_elbow')],
        [keypoints.index('right_elbow'), keypoints.index('right_wrist')],
        [keypoints.index('left_shoulder'), keypoints.index('left_elbow')],
        [keypoints.index('left_elbow'), keypoints.index('left_wrist')],
        [keypoints.index('right_hip'), keypoints.index('right_knee')],
        [keypoints.index('right_knee'), keypoints.index('right_ankle')],
        [keypoints.index('left_hip'), keypoints.index('left_knee')],
        [keypoints.index('left_knee'), keypoints.index('left_ankle')],
        [keypoints.index('right_shoulder'), keypoints.index('left_shoulder')],
        [keypoints.index('right_hip'), keypoints.index('left_hip')],
        [keypoints.index('right_shoulder'), keypoints.index('right_hip')],
        [keypoints.index('left_shoulder'), keypoints.index('left_hip')]
        ]
    return limbs



if __name__ == "__main__":
    webcam = cv2.VideoCapture(cv2.CAP_DSHOW+1)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("Using {} device".format(device))
    
    # create a model object from the keypointrcnn_resnet50_fpn class
    model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True).to(device)
    # call the eval() method to prepare the model for inference mode.
    model.eval()
    
    # create the list of keypoints.
    keypoints = ['nose','left_eye','right_eye','left_ear','right_ear','left_shoulder','right_shoulder','left_elbow',
                    'right_elbow','left_wrist','right_wrist','left_hip','right_hip','left_knee', 'right_knee', 'left_ankle','right_ankle']

    limbs = get_limbs_from_keypoints(keypoints)

    if not webcam.isOpened():
        print("Could not open webcam")
        exit()

    while webcam.isOpened():
        status, frame = webcam.read()

        if status:
            img = cv2.cvtColor(frame, cv2.IMREAD_COLOR)
      
            img_size_ratio = img.shape[0]/img.shape[1]
            img_w = 800
            img = cv2.resize(img, (img_w, int(img_w*img_size_ratio)))
            transform = T.Compose([T.ToTensor()])
            img_tensor = transform(img)
            img_tensor = img_tensor.to(device)
            # forward-pass the model
            # the input is a list, hence the output will also be a list
            output = model([img_tensor])[0]
            skeleton_img = draw_skeleton_per_person(img, output["keypoints"], output["keypoints_scores"], output["scores"],keypoint_threshold=1.5,linewith=2)
            
            cv2.imshow('video', img)
            cv2.imshow("video", skeleton_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    webcam.release()
    cv2.destroyAllWindows()