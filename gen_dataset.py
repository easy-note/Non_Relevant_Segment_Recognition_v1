import os
import glob
import cv2
from PIL import Image
import random
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms



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


class CAMIO_Dataset(Dataset):
    def __init__(self, base_path, is_train, data_ratio=0.1):
        self.is_train = is_train
        self.img_list = []
        self.label_list = []

        if self.is_train:
            tar_path = base_path + '/train'
            self.aug = data_transforms['train']
        else:
            tar_path = base_path + '/val'
            self.aug = data_transforms['val']

        dir_list = os.listdir(tar_path)

        for dir_name in dir_list:
            dpath = os.path.join(tar_path, dir_name)
            t_img_list = glob.glob(dpath + '/*jpg')
            
            if 'non' in dir_name:
                # tar_label = np.array([1, 0])
                tar_label = 0
            else:
                tar_label = 1
                # tar_label = np.array([0, 1])
            
            self.img_list += t_img_list
            for _ in range(len(t_img_list)):
                self.label_list.append(tar_label)
        
        indices = list(range(len(self.img_list)))
        random.shuffle(indices)
        split = int(len(indices) * data_ratio)

        self.img_list = [self.img_list[i] for i in indices[:split]]
        self.label_list = [self.label_list[i] for i in indices[:split]]
        

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img_path, label = self.img_list[index], self.label_list[index]

        img = Image.open(img_path)
        img = self.aug(img)

        return img, label


# TODO 데이터 생성하는 부분(robot/lapa) 둘다 통합하는 코드로 변경 필요함

trainset = ['R001', 'R002', 'R003', 'R004', 'R005', 'R006', 'R007', 'R010', 'R013', 'R014', 'R015', 'R018', 
            'R019', 'R048', 'R056', 'R074', 'R076', 'R084', 'R094', 'R100', 'R117', 'R201', 'R202', 'R203', 
            'R204', 'R205', 'R206', 'R207', 'R209', 'R210', 'R301', 'R302', 'R304', 'R305', 'R313']

valset = ['R017', 'R022', 'R116', 'R208', 'R303']

class_name = ['camIO', 'non_camIO']
tar_surgery = 'robot'
video_ext = '.mp4'
anno_path = '/data/CAM_IO/robot/OOB'
fps = 30


def time_to_idx(time, fps):
    t_segment = time.split(':')
    idx = int(t_segment[0]) * 3600 * fps + int(t_segment[1]) * 60 * fps + int(t_segment[2]) * fps + int(t_segment[3])

    return idx


def gen_data(org_video_path, save_dir_path):
    """
        save OOB images extracted from the surgery video
    """
    anno_list = glob.glob(anno_path + '/*csv')

    train_path = os.path.join(save_dir_path, 'train')
    test_path = os.path.join(save_dir_path, 'val')

    for tar_class in class_name:
        for tpath in [train_path, test_path]:
            spath = os.path.join(tpath, tar_class)
            if not os.path.exists(spath):
                os.makedirs(spath)

    for apath in anno_list:
        csv_name = apath.split('/')[-1]
        tokens = csv_name.split('_')

        if tokens[0] in valset:
            tar_class = 'val'
        else:
            tar_class = 'train'

        video_name = ''
        tk_len = len(tokens)
        for ti, token in enumerate(tokens[:-1]):
            if 'CAMIO' in token:
                continue
            if ti < tk_len -2:
                video_name += token + '_'
            else:
                video_name += token + video_ext

        df = pd.read_csv(apath)
        d_size = len(df) - 1

        t_idx_list = []
        for i in range(d_size):
            t_start = df.loc[i]['start']
            t_end = df.loc[i]['end']

            if not isinstance(t_start, str) or not isinstance(t_end, str):
                break

            t_idx_list.append([time_to_idx(t_start, fps), time_to_idx(t_end, fps)])

        # pass if timestamp is not existed
        if len(t_idx_list) < 1:
            continue

        idx_list = []
        for li in t_idx_list:
            idx_list.append(li[0])
            idx_list.append(li[1])

        video = cv2.VideoCapture(os.path.join(org_video_path, tar_surgery, 'video', video_name))
        v_len = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        
        OOB_idx = 0
        OOB_cnt = 0
        No_OOB_cnt = 0
        
        for frame in range(v_len):
            if frame % fps:
                continue
            # video.set(1, frame)
            _, img = video.read()

            if idx_list[OOB_idx] <= frame and frame <= idx_list[OOB_idx+1]:
                OOB_cnt += 1
                sfile = os.path.join(save_dir_path, tar_class, class_name[1], '{}_{:010d}.jpg'.format(video_name[:-4], frame))
            else:
                No_OOB_cnt += 1
                sfile = os.path.join(save_dir_path, tar_class, class_name[0], '{}_{:010d}.jpg'.format(video_name[:-4], frame))

            if frame+1 > idx_list[OOB_idx+1]:
                if OOB_idx + 2 < len(idx_list)-1:
                    OOB_idx += 2

            cv2.imwrite(sfile, img)
            # print('Save file : {}'.format(sfile))

        video.release()
        print('Video processing done | OOB : {:08d}, No-OOB : {:08d}'.format(OOB_cnt, No_OOB_cnt))

