import os
import cv2
from PIL import Image
import torch
import numpy as np
import pandas as pd
import glob
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from Model import CAMIO
from torchvision import transforms


os.environ['CUDA_VISIBLE_DEVICES'] = '1'

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
def time_to_idx(time, fps):
    t_segment = time.split(':')
    idx = int(t_segment[0]) * 3600 * fps + int(t_segment[1]) * 60 * fps + int(t_segment[2]) * fps + int(t_segment[3])

    return idx


valset = ['R017', 'R022', 'R116', 'R208', 'R303']
video_dir_path = '/home2/gastrectomy'
anno_path = '/home/mkchoi/dataset/CAM'
tar_surgery = 'robot'
video_ext = '.mp4'
fps = 30


model = CAMIO()
model.cuda()
model.eval()
model = model.load_from_checkpoint(os.getcwd() + '/camIO_lightning/logs/OOB_robot_test/epoch=4-val_loss=0.2715.ckpt')

print('model loaded')

anno_list = glob.glob(anno_path + '/*csv')
dpath = os.path.join(video_dir_path, tar_surgery)

info_dict = {
    'video': [],
    'anno': [],
}

# information gathering
for tar_no in valset:
    for anno_path in anno_list:
        anno_file = anno_path.split('/')[-1]

        if tar_no in anno_file:
            tokens = anno_file.split('_')
            video_name = ''
            tk_len = len(tokens)
            for ti, token in enumerate(tokens[:-1]):
                if 'CAMIO' in token:
                    continue
                if ti < tk_len -2:
                    video_name += token + '_'
                else:
                    video_name += token + video_ext

            vpath = os.path.join(dpath, video_name)

            if os.path.exists(vpath) and not vpath in info_dict['video']:
                info_dict['video'].append(vpath)

                df = pd.read_csv(anno_path)
                d_size = len(df) - 1

                idx_list = []
                for i in range(d_size):
                    t_start = df.loc[i]['start']
                    t_end = df.loc[i]['end']

                    if not isinstance(t_start, str) or not isinstance(t_end, str):
                        break

                    idx_list.append([time_to_idx(t_start, fps), time_to_idx(t_end, fps)])

                # pass if timestamp is not existed
                if len(idx_list) < 1:
                    info_dict['video'].pop()
                    continue

                info_dict['anno'].append(idx_list)

# print(info_dict)


# inference step
plot_cnt = 1
for vpath, anno in zip(info_dict['video'], info_dict['anno']):
    video = cv2.VideoCapture(vpath)
    v_len = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    idx_list = []

    for ann in anno:
        for a in ann:
            idx_list.append(a)
    
    OOB_idx = 0
    gts = []
    preds = []
    
    for frame in range(v_len):
        if frame % 500 != 0:
            continue
        _, img = video.read()

        if idx_list[OOB_idx] <= frame and frame <= idx_list[OOB_idx+1]:
            gts.append(1)
        else:
            gts.append(0)

        _img = data_transforms['val'](Image.fromarray(img))
        _img = torch.unsqueeze(_img, 0)
        outputs = model(_img)

        idx = torch.argmax(outputs.cpu())
        preds.append(idx.data.numpy())

        if frame+1 > idx_list[OOB_idx+1]:
            # print(frame, idx_list[OOB_idx])
            if OOB_idx + 2 < len(idx_list):
                OOB_idx += 2

        if frame % 1000 == 0:
            print('Current processed [{:.4f}/100%]'.format(frame/v_len * 100))

    video.release()

    print(len(gts), len(preds))

    fig = plt.figure(figsize=(16,8))
    # plt.hold()
    plt.scatter(range(len(gts)), np.array(gts), color='green', marker='o', s=10)
    plt.scatter(range(len(gts)), np.array(preds), color='red', marker='o', s=5)
    plt.savefig(os.getcwd() + '/camIO_lightning/plot_test_{:03d}.png'.format(plot_cnt))
    plot_cnt += 1
    # break