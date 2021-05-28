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
    save_root_dir = '/data/ROBOT/Inference'

    video_dir = '/data/ROBOT/Video'
    video_list = os.listdir(video_dir)




    patient_list = ['R_17', 'R_22', 'R_116', 'R_208', 'R_303'] + ['R_3', 'R_4', 'R_6', 'R_13', 'R_18'] + ['R_7', 'R_10', 'R_19', 'R_56', 'R_74']

    patient_for_parser = [patient + '_' for patient in patient_list]
    
    target_video = []

    for patient in patient_for_parser:
        for video in video_list:
            if patient in video:
                target_video.append(video)

    print('count of target_video : ', len(target_video))

    for target in target_video:
        vidcap = cv2.VideoCapture(os.path.join(video_dir, target))
        video_length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

        a, b = divmod(video_length, 30000)
        video_frame_list = [30000] * a
        video_frame_list.append(b)

        print('\n\n \t\t---- --- ---- ')
        print(target_video)
        print(target)
        
        for e, n in enumerate(video_frame_list):
            print(e, ', ', n)
            hospital, surgery_type, surgeon, op_method, patient_idx, video_channel, video_slice = os.path.splitext(target)[0].split('_')

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
                    cv2.imwrite("image/cv2_img_" + os.path.splitext(target)[0] + '_' + str(e) + '_' + str(i) + ".jpg", image)

            path = os.path.join(save_root_dir, '{}_{}'.format(op_method, patient_idx), os.path.splitext(target)[0])
            if not os.path.isdir(path):
                os.makedirs(path)

            with open(os.path.join(path, os.path.splitext(target)[0]+'_'+str(e)), "wb") as f:
                pickle.dump(t, f)

            t = None

def load_data():
    with open('/data/CAM_IO/robot/inference_dataset/R017/R017_ch1_video_01/R017_ch1_video_01_0', "rb") as f:
        loaded_data = pickle.load(f)

        print(loaded_data)

if __name__ == '__main__':
    gen_data()

   
