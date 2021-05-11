import os
import glob2
import glob
import cv2
from PIL import Image
import random
import numpy as np
import pandas as pd
from pandas import DataFrame as df

from tqdm import tqdm
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
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

IB_CLASS, OOB_CLASS = (0,1)


def make_robot_csv(data_dir, save_dir) : 
    
    IB_dir_name, OOB_dir_name = ['InBody', 'OutBody']

    class_dir_name = ['InBody', 'OutBody']

    print(data_dir)
    print(save_dir)
    
    img_path_list = []
    class_list = []

    # calc path, class
    for class_path in class_dir_name :
        print('processing... ', os.path.join(data_dir, class_path))

        # init
        temp_path = []
        temp_class = []

        temp_path = glob.glob(os.path.join(data_dir, class_path, '*.jpg'))

        if class_path == IB_dir_name :
            temp_class = [IB_CLASS]*len(temp_path)

        if class_path == OOB_dir_name :
            temp_class = [OOB_CLASS]*len(temp_path)

        img_path_list += temp_path
        class_list += temp_class

    # save 
    save_df = df({
        'img_path' : img_path_list,
        'class_idx' : class_list })    # 모든 이미지 path와 class 정보 저장


    save_df.to_csv(os.path.join(save_dir, 'robot_oob_assets_path.csv'), mode='w', index=False)



class CAMIO_Dataset(Dataset):
    def __init__(self, csv_path, patient_name, is_train, random_seed, IB_ratio):
        self.is_train = is_train
        self.csv_path = csv_path 
        self.aug = data_transforms['train'] if is_train else data_transforms['val']
        
        self.img_list = [] # img
        self.label_list = [] # label

        read_assets_df = pd.read_csv(csv_path) # read csv
        
        print('\n\n')
        print('==> \tPATIENT')
        print('|'.join(patient_name))
        patient_name_for_parser = [patient + '_' for patient in patient_name]
        print('|'.join(patient_name_for_parser))
        
        print('==> \tREAD_CSV')
        print(read_assets_df)
        print('\n\n')


        # select patient video
        self.assets_df = read_assets_df[read_assets_df['img_path'].str.contains('|'.join(patient_name_for_parser))]

        # seperate df by class
        self.ib_df = self.assets_df[self.assets_df['class_idx']==IB_CLASS]
        self.oob_df = self.assets_df[self.assets_df['class_idx']==OOB_CLASS]

        print('\n\n')
        print('==> \tINBODY_CSV')
        print(self.ib_df)
        print('\t'* 4)
        print('==> \tOUTBODY_CSV')
        print(self.oob_df)
        print('\n\n')

        # sort
        self.ib_df = self.ib_df.sort_values(by=['img_path'])
        self.oob_df = self.oob_df.sort_values(by=['img_path'])

        print('\n\n')
        print('==> \tSORT INBODY_CSV')
        print(self.ib_df)
        print('\t'* 4)
        print('==> \tSORT OUTBODY_CSV')
        print(self.oob_df)
        print('\n\n')

        # random_sampling and setting IB:OOB data ratio
        self.ib_df = self.ib_df.sample(n=len(self.oob_df)*IB_ratio, replace=False, random_state=random_seed) # 중복뽑기x, random seed 고정, OOB개수의 IB_ratio 개
        self.oob_df = self.oob_df.sample(frac=1, replace=False, random_state=random_seed)

        print('\n\n')
        print('==> \tRANDOM SAMPLING INBODY_CSV')
        print(self.ib_df)
        print('\t'* 4)
        print('==> \tRANDOM SAMPLING OUTBODY_CSV')
        print(self.oob_df)
        print('\n\n')

        # suffle 0,1
        self.assets_df = pd.concat([self.ib_df, self.oob_df]).sample(frac=1, random_state=random_seed).reset_index(drop=True)
        print('\n\n')
        print('==> \tFINAL ASSETS')
        print(self.assets_df)
        print('\n\n')

        print('\n\n')
        print('==> \tFINAL HEAD')
        print(self.assets_df.head(20))
        print('\n\n')

        # last processing
        self.img_list = self.assets_df.img_path.tolist()
        self.label_list = self.assets_df.class_idx.tolist()
        
        

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img_path, label = self.img_list[index], self.label_list[index]

        img = Image.open(img_path)
        img = self.aug(img)

        return img, label





# TODO 데이터 생성하는 부분(robot/lapa) 둘다 통합하는 코드로 변경 필요함
'''
trainset = ['R001', 'R002', 'R003', 'R004', 'R005', 'R006', 'R007', 'R010', 'R013', 'R014', 'R015', 'R018', 
            'R019', 'R048', 'R056', 'R074', 'R076', 'R084', 'R094', 'R100', 'R117', 'R201', 'R202', 'R203', 
            'R204', 'R205', 'R206', 'R207', 'R209', 'R210', 'R301', 'R302', 'R304', 'R305', 'R313']

valset = ['R017', 'R022', 'R116', 'R208', 'R303']

# class_name = ['camIO', 'non_camIO'] # [0(in body), 1(out of body)]
class_name = ['InBody', 'OutBody']

tar_surgery = 'robot'
video_ext = '.mp4'
anno_path = '/data/CAM_IO/robot/OOB'
fps = 30
'''


def time_to_idx(time, fps):
    t_segment = time.split(':')
    idx = (int(t_segment[0]) * 3600 * fps) + (int(t_segment[1]) * 60 * fps) + (int(t_segment[2]) * fps) # [h, m, s, ms] 

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

        # csv, video pair sampleing done.
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

        # data prosseing target info
        print('=========================')
        print('Target Video : ', os.path.join(org_video_path, 'CAM_IO', tar_surgery, 'video', video_name))
        print('OOB Frame range : ', t_idx_list)

        video = cv2.VideoCapture(os.path.join(org_video_path, 'CAM_IO', tar_surgery, 'video', video_name))
        v_len = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        # check vedio frame
        video_width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
        video_height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
        video_fps = video.get(cv2.CAP_PROP_FPS)
        print('video_width: %d, video_height: %d, video_fps: %d' %(video_width, video_height, video_fps))

        OOB_idx = 0 # out of body frame index range
        OOB_cnt = 0 # out of body (non_camIO)
        No_OOB_cnt = 0 # inbody (camIO)

        for frame in tqdm(range(v_len)):
            if frame % fps: # 0, 30, 60 -> 1fps
                continue
            
            video.set(1, frame) # frame setting
            _, img = video.read() # read img

            if idx_list[OOB_idx] <= frame and frame <= idx_list[OOB_idx+1]: # is out of body?
                OOB_cnt += 1
                print(frame)
                sfile = os.path.join(save_dir_path, tar_class, class_name[0], '{}_{:010d}.jpg'.format(video_name[:-4], frame)) # non camIO
            else:
                No_OOB_cnt += 1
                sfile = os.path.join(save_dir_path, tar_class, class_name[1], '{}_{:010d}.jpg'.format(video_name[:-4], frame)) # camio

            if frame+1 > idx_list[OOB_idx+1]:
                if OOB_idx + 2 < len(idx_list)-1:
                    OOB_idx += 2

            cv2.imwrite(sfile, img)
            # print('Save file : {}'.format(sfile))

        video.release()
        print('Video processing done | OOB : {:08d}, No-OOB : {:08d}'.format(OOB_cnt, No_OOB_cnt))


