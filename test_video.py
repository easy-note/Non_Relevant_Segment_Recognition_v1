import os
import cv2
from PIL import Image
import torch
import numpy as np
import pandas as pd
import glob
import matplotlib

from pandas import DataFrame as df
from tqdm import tqdm

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from Model import CAMIO
from torchvision import transforms



# cal vedio frame
def time_to_idx(time, fps):
    t_segment = time.split(':')
    idx = int(t_segment[0]) * 3600 * fps + int(t_segment[1]) * 60 * fps + int(t_segment[2])

    return idx


def test_video() :
    parser = argparse.ArgumentParser()ß
    
    parser.add_argument('--model_path', type=str,
                        default=model.load_from_checkpoint(os.getcwd() + '/logs/robot/OOB/robot_oob_train_2/epoch=1-val_loss=0.0466.ckpt'), help='trained model_path')
    
    parser.add_argument('--data_path', type=str,
                        default='/data/CAM_IO/robot/video', help='video_path :) ')

    parser.add_argument('--anno_path', type=str,
                        default='/data/CAM_IO/robot/OOB', help='annotation_path :) ')

    parser.add_argument('--results_path', type=str,
                        default=os.path.join(os.getcwd(), 'results'), help='inference results save path')

    parser.add_argument('--mode', type=str,
                        default='robot', choice=['robot', 'lapa'], help='inference results save path')

    args, _ = parser.parse_known_args()

    
    ###  base setting for model testing ### 
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    data_transforms = {
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # dirscirbe exception, inference setting for each mode
    if args.mode == 'lapa' :        
        # test_video_for_lapa()
        pass
    
    else : # robot
        model = CAMIO()
        model.cuda()
        model.eval()
        model = model.load_from_checkpoint(args.model_path)

        print('model_loded for ROBOT')

        # starting inference
        test_video_for_robot(args.data_path, args.anno_path, args.results_path, model, data_transforms)





def test_video_for_robot(data_path, anno_path, result_path, model, data_transforms) :
    
    valset = ['R017', 'R022', 'R116', 'R208', 'R303']
    valset = ['R017']

    video_ext = '.mp4'
    fps = 30

    tar_surgery = 'robot'

    gettering_information_for_robot(data_path, anno_path, valset)


    
    

def gettering_information_for_robot (video_dir_path, anno_path, video_set) : # paring video from annotation info
    info_dict = {
        'video': [],
        'anno': [],
    }

    all_video_path = glob.glob(video_dir_path +'/mp4') # all video file list
    all_anno_path = glob.glob(anno_path + '/*csv') # all annotation file list
    
    dpath = os.path.join(video_dir_path) # video_root path


    for video_no in video_set : # get target video
        video_path = [vfile for vfile in all_video_path if vfile.start]
        

        for anno_path in anno_list: # searching in annotation list
            anno_file = os.path.basename(anno_path) #

        if video_no in anno_file:
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

print(info_dict['video'])
print(info_dict['anno'])


valset = ['R017', 'R022', 'R116', 'R208', 'R303']
valset = ['R017']
video_dir_path = '/data/CAM_IO/robot/video'
anno_path = '/data/CAM_IO/robot/OOB'
tar_surgery = 'robot'
video_ext = '.mp4'
fps = 30


model = CAMIO()
model.cuda()
model.eval()
model = model.load_from_checkpoint(os.getcwd() + '/logs/robot/OOB/robot_oob_train_1/epoch=12-val_loss=0.0303.ckpt')

print('model loaded')

anno_list = glob.glob(anno_path + '/*csv')
dpath = os.path.join(video_dir_path)

info_dict = {
    'video': [],
    'anno': [],
}

print(anno_list)
print(dpath)

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

print(info_dict['video'])
print(info_dict['anno'])


print()

# inference step
plot_cnt = 1
base_save_dir = os.path.join(os.getcwd(), 'results')


for vpath, anno in zip(info_dict['video'], info_dict['anno']):

    video = cv2.VideoCapture(vpath)
    v_len = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    ## FP frmae save
    video_name = os.path.basename(vpath).split('.')[0] # only  name

    print('Target video : ', video_name)

    # setting save dir of each video
    results_save_dir = os.path.join(base_save_dir, video_name)

    # make saving directory
    try:
        if not(os.path.isdir(results_save_dir)):
            os.makedirs(results_save_dir)
            os.makedirs(os.path.join(results_save_dir, 'frame'))
    except OSError as e:
        if e.errno != errno.EEXIST:
            print("Failed to create directory!!!!!")
            raise

    # init_variable
    idx_list = [] # init_variable, this is for use annotation info
    gts = [] # ground truth
    preds = [] # predict
    frame_idx = [] # frmae info
    

    for ann in anno: # [[start, end], [start, end]...]
        for a in ann: # [start, end]
            idx_list.append(a) # idx_list = [start, end, start, end, start, end...]
    
    OOB_idx = 0 # index for idx_list[0, 1, 2 , 3, 4..] -> idx[0] ~ idx[1] , idx[2] ~ idx[3], idx[4] ~ idx[5].... incresaing 2 steps 

    
    # cutting viedo in frame
    for frame in tqdm(range(v_len)): 
        truth = -1 # init
        idx = -1 # init

        if frame % 5000 != 0:
            continue
            # pass

        
        video.set(1, frame) # frame setting
        _, img = video.read()

        # for checking frame, is rightly input in model?
        print('')
        print(os.path.join(results_save_dir, 'frame', '{}_{:010d}.jpg'.format(video_name, frame)))
        cv2.imwrite(os.path.join(results_save_dir, 'frame', '{}_{:010d}.jpg'.format(video_name, frame)), img)
        

         # idx_list[start, end, start, end...]
        if idx_list[OOB_idx] <= frame and frame <= idx_list[OOB_idx+1]: # is oob frame?
            truth = 1 # oob
        else: # not
            truth = 0 # in body

        # truth
        gts.append(truth)

        # inferenceing frame in model
        _img = data_transforms['val'](Image.fromarray(img))
        _img = torch.unsqueeze(_img, 0)
        outputs = model(_img)

        # results of predict
        idx = torch.argmax(outputs.cpu()) # predict
        preds.append(idx.data.numpy())

        # frame
        frame_idx.append(frame)


        
        print('frame {} | truth {} | predict {}'.format(frame, truth, idx))
        # append inference results by  frame
        '''
        append_data = {
            'frame' : frame,
            'truth' : truth,
            'preds' : idx
        }
        inference_results_df.append(append_data, ignore_index=True)
        '''

        # move next idx_list index
        if frame+1 > idx_list[OOB_idx+1]: # ?????
            # print(frame, idx_list[OOB_idx])
            if OOB_idx + 2 < len(idx_list):
                OOB_idx += 2

        if frame % 1000 == 0:
            print('Current processed [{:.4f}/100%]'.format(frame/v_len * 100))

    video.release()
    
    # save step
    result_dict = {
        'frame' : frame_idx,
        'truth' : gts,
        'predict' : preds
    }
    inference_results_df = pd.DataFrame(result_dict)

    # saving inferece result
    print('Saved at ====> \t', results_save_dir)
    inference_results_df.to_csv(os.path.join(results_save_dir, '%s.csv'%video_name), mode="w")
    
    print(inference_results_df)
    print(len(gts), len(preds))


    # saving plot
    fig = plt.figure(figsize=(16,8))

    # plt.hold()
    plt.scatter(np.array(frame_idx), np.array(gts), color='blue', marker='o', s=15, label='Truth') # ground truth
    plt.scatter(np.array(frame_idx), np.array(preds), color='red', marker='o', s=5, label='Predict') # predict

    plt.title('Inference Results By frame'); plt.suptitle('%s'%video_name)
    plt.ylabel('class'); plt.xlabel('frame');
    plt.legend(loc='center right')

    plt.savefig(os.path.join(results_save_dir, '%s_plot.png'%video_name))
    
    plot_cnt += 1
    
    exit(1)