"""
Generate test dataset (robot, lapa).

Usage:
    gen_test_dataset.py --patient_list <patient_list> --video_dir_path <video_dir_path> --save_dir_path <save_dir_path>
"""

import os
from PIL import Image
from torchvision import transforms
import torch
import argparse

import numpy as np
import pickle
from decord import VideoReader

parser = argparse.ArgumentParser()

parser.add_argument('--patient_list', type=str, help='Select patients to generate dataset')

parser.add_argument('--video_dir_path', type=str, help='Video directory path')
parser.add_argument('--save_dir_path', type=str, help='Save directory path')

args, _ = parser.parse_known_args()


# data transforms (output: tensor)
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
    pil_img = Image.fromarray(npy_img)
    img = aug(pil_img)
    
    return img

def gen_test_dataset(video_dir_path, patient_list, save_dir_path):
    """Create test dataset in tensor form.

    Generate test datasets by videos.
    Create a 4-dimensional tensor in 30000 frames of video (Capacity issues).

    Args:
        video_dir_path: video directory path.
        patient_list: patients to generate dataset.
        save_dir_path: directory path to save test dataset.
    """

    video_dir_path = video_dir_path
    video_list = os.listdir(video_dir_path)
    patient_list = patient_list
    
    tmp_target_video = []
    target_video = []

    # Choose target video.
    for patient in patient_list:
        for video in video_list:
            if patient in video:
                tmp_target_video.append(video)

    tmp_target_video.sort()

    for i in tmp_target_video:
        if ('xx0' in i) or ('ch' in i):
            target_video.append(i)
    
    target_video.sort()

    # Generate test dataset in tensor form.
    for target in target_video:
        vr = VideoReader(video_dir_path + target)
        print(target)
        print(len(vr))

        a, b = divmod(len(vr), 30000) #30000
        video_frame_list = [30000] * a #30000
        video_frame_list.append(b)

        print(video_frame_list)

        cnt = 0
        for m, frame_idx in enumerate(video_frame_list): # [30000, 21060]
            # 4차원 텐서 (image 수, channel 수, Height, Width)
            t = torch.Tensor(frame_idx, 3, 224, 224)

            for i in range(frame_idx):
                img = npy_to_tensor(vr[cnt].asnumpy())
                t[i] = img
                cnt += 1
        
            path = save_dir_path + target[10:13] + '/' + target[:-4] + '/'
            if not os.path.isdir(path):
                os.makedirs(path)

            with open(path + target[:-4] + '_' + str(m), "wb") as f:
                pickle.dump(t, f)

            t = None

if __name__ == '__main__':
    gen_test_dataset(args.video_dir_path, args.patient_list, args.save_dir_path)