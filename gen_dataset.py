import os
import glob
import cv2
from PIL import Image
import random
import numpy as np
import pandas as pd
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


class CAMIO_Dataset(Dataset):
    def __init__(self, base_path, is_train, test_mode=False, data_ratio=0.1):
        self.is_train = is_train
        self.test_mode = test_mode # ./test parser   
        self.img_list = []
        self.label_list = []

        '''
        if self.test_mode: # test mode
            tar_path = base_path + '/test'
            self.aug = data_transforms['test']
        '''
        if self.test_mode : # test_mode (it will be check all of data /train and /val)
            # tar_path_list = [base_path + '/test'] # if you want to other test set, try this
            tar_path_list = [base_path + '/train', base_path + '/val']
            self.aug = data_transforms['test']

        else : # train_mode : default
            if self.is_train:
                tar_path_list = [base_path + '/train']
                self.aug = data_transforms['train']
            else:
                tar_path_list = [base_path + '/val']
                self.aug = data_transforms['val']

        for tar_path in tar_path_list : # ['/train', '/val', 'test'] whatever you want
            
            dir_list = os.listdir(tar_path) # [InBody, OutBody]
            
            for dir_name in dir_list: # [InBody, OutBody]
                dpath = os.path.join(tar_path, dir_name) # [train, val, test] / [InBody, OutBody]
                t_img_list = glob.glob(dpath + '/*.jpg') # all .jpg file in [train, val, test] / [InBody, OutBody] / *.jpg
                # sorting because of apply  random seed fairly
                # t_img_list = sorted(t_img_list)

                print("Load Dataset at \t {} \t ====> {} file Exsisted ".format(dpath, len(t_img_list)))
                
                if 'In' in dir_name: # Inbody
                    # tar_label = np.array([1, 0])
                    tar_label = 0
                else: # Outbody
                    tar_label = 1
                    # tar_label = np.array([0, 1])

                # init temp ==> it will be added into self.img_list, self.label
                temp_img_list = []
                temp_label_list = []

                ###  this is for balacing train_set 3(In body) : 1(out body) ==> temp code ### 
                if 'In' in dir_name and 'train' in tar_path : # train
                    print('\t @@@ in / train')
                    # img
                    ## 중복허용하지 않고 sampling
                    random.seed(100) # seed fix
                    # temp_img_list = random.sample(t_img_list, 43518) # out body의 3배 (43518)
                    temp_img_list = t_img_list
                    

                elif 'In' in dir_name and 'val' in tar_path : # val
                    print('\t @@@ in / val')
                    # img
                    ## 중복허용하지 않고 sampling
                    random.seed(100) # seed fix
                    # temp_img_list = random.sample(t_img_list, 11151) # out body의 3배 (11151)
                    temp_img_list = t_img_list

                else : 
                    # img
                    temp_img_list = t_img_list
                    
                # label
                # label will be configurationed of temp_img_list count 
                temp_label_list = [tar_label] * len(temp_img_list)

                # update dataset (append temp data) 
                self.img_list += temp_img_list # img
                self.label_list += temp_label_list # label 
                    
                # check updataing rightly
                print('\t Dataset Updated from ==> {} | Updated File Count ==> {} ", Updated Label Count ==> {} | Label Info ==> {} '.format(dpath, len(temp_img_list), len(temp_label_list), np.unique(temp_label_list)))
                print('')


        # modify number of dataset from data_ratio parameter
        indices = list(range(len(self.img_list)))
        random.seed(10) # seed fix
        random.shuffle(indices)
        split = int(len(indices) * data_ratio)

        self.img_list = [self.img_list[i] for i in indices[:split]]
        self.label_list = [self.label_list[i] for i in indices[:split]]

        # final check dataset info
        print('\t\t ===== '*2, 'DATASET REPORT', '\t\t ===== '*2)
        print('\t\t @ Dataset composed from ==> {} | target count ==> {} ", label count ==> {} '.format(tar_path_list, len(self.img_list), len(self.label_list)))
        print('\t\t @ label composite info', np.unique(self.label_list, return_counts=True))
        print('\t\t ===== '*2, '==============', '\t\t ===== '*2)
        
        

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


