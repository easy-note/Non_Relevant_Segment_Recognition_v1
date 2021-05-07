import os
import cv2
from PIL import Image
from torchvision import transforms
import torch

import numpy as np
import pickle

import sys
from tqdm import tqdm


data_transforms = {
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

aug = data_transforms['test']

def npy_to_tensor(npy_img):
    npy_img = cv2.cvtColor(npy_img, cv2.COLOR_BGR2RGB)
    # npy_img = npy_img[:,:,::-1]
    pil_img = Image.fromarray(npy_img)
    img = aug(pil_img)
    
    return img


def gen_data():
    video_dir = '/data/CAM_IO/robot/video/'
    video_list = os.listdir(video_dir)



    # patient_list = ['R003', 'R004', 'R006', 'R013', 'R018'] + ['R007', 'R010', 'R019', 'R056', 'R074']
    patient_list = ['R019', 'R056', 'R074']

    # patient_list = ['R017']
    target_video = []

    for patient in patient_list:
        for video in video_list:
            if patient in video:
                target_video.append(video)

    for target in target_video:
        vidcap = cv2.VideoCapture(video_dir + target)
        video_length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

        a, b = divmod(video_length, 30000)
        video_frame_list = [30000] * a
        video_frame_list.append(b)

        print('\n\n \t\t---- --- ---- ')
        print(target_video)
        print(target)
        
        for e, n in enumerate(video_frame_list):
            print(e, ', ', n)

            # success, image = vidcap.read()
            
            #cv2 image save
            # cv2.imwrite("image/cv2_img_"+str(e)+".jpg", image)
            # img = npy_to_tensor(image)

            # path = 'inference_dataset/'


            t = torch.Tensor(n, 3, 224, 224)

            for i in tqdm(range(n), desc='Gen dataset ... | {}'.format(target)) :
                success, image = vidcap.read()
                img = npy_to_tensor(image)
                t[i] = img

                if i % 10000 == 0:
                    cv2.imwrite("image/cv2_img_" + target[:-4] + str(e) + '_' + str(i) + ".jpg", image)

            
            path = '/data/CAM_IO/robot/inference_dataset/' + target[:4] + '/' + target[:-4] + '/'
            if not os.path.isdir(path):
                os.makedirs(path)

            with open(path + target[:-4] + '_' + str(e), "wb") as f:
                pickle.dump(t, f)

            t = None

def load_data():
    with open('/data/CAM_IO/robot/inference_dataset/R017/R017_ch1_video_01/R017_ch1_video_01_0', "rb") as f:
        loaded_data = pickle.load(f)

        print(loaded_data)

if __name__ == '__main__':
    gen_data()

   
