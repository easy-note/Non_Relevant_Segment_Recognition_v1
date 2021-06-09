"""
Generate train dataset (robot, lapa).

Usage:
    gen_train_dataset.py --device <divice> --patient_list <patient_list> --video_dir_path <video_dir_path> --anno_dir_path <anno_dir_path> --save_dir_path <save_dir_path>
"""

import os
import csv
import json
import argparse

from decord import VideoReader
from PIL import Image


parser = argparse.ArgumentParser()

parser.add_argument('--device', default = 'Robot', type=str, choices = ['Robot', 'Lapa'], help='Select device type') 
parser.add_argument('--patient_list', default = ['301', '303'], type=str, help='Select patients to generate dataset')

parser.add_argument('--video_dir_path', default = '/data/ROBOT/Video/', type=str, help='Video directory path')
parser.add_argument('--anno_dir_path', default = '/data/OOB/', type=str, help='Annotation file direcotry path')

parser.add_argument('--save_dir_path', default = '/data/ROBOT/_test/', type=str, help='Save directory path')

args, _ = parser.parse_known_args()


def time_to_frame(time, fps):
    t_segment = time.split(':')
    frame = (int(t_segment[0]) * 3600 * fps) + (int(t_segment[1]) * 60 * fps) + (int(t_segment[2]) * fps) # [h, m, s, ms] 

    return frame


def gen_meta_file(video_dir_path, anno_dir_path, device, patient_list, save_dir_path):
    """Creating an Video-Annotation matching file.

    Args:
        video_dir_path: video directory path.
        anno_dir_path: annotation directory path.
        device: device type (robot, lapa).
        patient_list: patients to generate dataset.
        save_dir_path:  directory path to save matching file.
    """

    video_path = video_dir_path #'/data/LAPA/Video/'
    anno_path = anno_dir_path #'/data/OOB/'

    video_list = []
    anno_list = []

    # Lapa meta file
    if device == 'Lapa':
        # 사용할 전체 비디오
        # xx0, ch1, patient
        for root, dirs, files in os.walk(video_path):
            files.sort(key=str.lower)
            
            for file in files:
                if ('Syno' not in file) and ('L' in file):
                    if ('xx0' in file) or ('ch1' in file):
                        for patient in patient_list:
                            if patient in file:
                                video_list.append(file)
        # print(video_list)

        # 사용할 전체 Annotation file
        for root, dirs, files in os.walk(anno_path):
            files.sort(key=str.lower)
            for file in files:
                if 'L' in file:
                    for patient in patient_list:
                        if patient in file:
                            anno_list.append(file)
        # print(anno_list)

    # Robot meta file
    elif device == 'Robot':
        # 사용할 전체 비디오
        for root, dirs, files in os.walk(video_path):
            files.sort(key=str.lower)
            
            for file in files:
                if ('Syno' not in file) and ('R' in file):
                    for patient in patient_list:
                        if patient in file:
                            video_list.append(file)

        # 사용할 전체 Annotation file
        for root, dirs, files in os.walk(anno_path):
            files.sort(key=str.lower)
            for file in files:
                if 'R' in file:
                    for patient in patient_list:
                        if patient in file:
                            anno_list.append(file)
    
    video_list.sort()
    anno_list.sort()


    ## [Video file - Annotaion file] matching
    total_list = []
    while anno_list:
        if not video_list :
            print('\nERROR: Video is less than annotation file \n')
            print('===== Left annotation files ===== \n {}'.format(anno_list))
            exit(0)

        temp = []
        temp.append(video_path + video_list[0])

        # 만약 비디오에 해당하는 annotation file 이 있다면. 
        if device == 'Lapa': 
            if anno_list[0][:-12] == video_list[0][:-4]:
                temp.append(root + anno_list[0])
                anno_list.pop(0)

        elif device == 'Robot':
            if anno_list[0][:-11] == video_list[0][:-4]:
                temp.append(root + anno_list[0])
                anno_list.pop(0)

        # 리스트에서 입력한 비디오 삭제하기.
        video_list.pop(0)
        total_list.append(temp)

    # print(total_list)
    
    # csv 파일 저장
    if not os.path.isdir(save_dir_path):
        os.mkdir(save_dir_path)

    output_file = save_dir_path + device + '_Video-Annotation_Matching.csv'
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)

        for i in total_list:
            writer.writerow(i)

    if device == 'Lapa':
        gen_robot_train_dataset(output_file, device, save_dir_path)

    elif divice == 'Robot':
        gen_lapa_train_dataset(output_file, device, save_dir_path)


def gen_lapa_train_dataset(meta_file_path, device, save_dir_path):
    f = open(meta_file_path, 'r', encoding='utf-8')
    rdr = csv.reader(f)

    if not os.path.isdir(save_dir_path + 'InBody/'):
        os.mkdir(save_dir_path + 'InBody/')
    if not os.path.isdir(save_dir_path + 'OutBody/'):
        os.mkdir(save_dir_path + 'OutBody/')

    for line in rdr:
        video_path = line[0]

        # if '01_G_01_L_423_xx0_01' in video_path:
        #     continue

        # Annotation file 존재
        if len(line) == 2:
            anno_path = line[1]
            print(anno_path)

            with open(anno_path) as json_file:
                time_list = []

                json_data = json.load(json_file)
                json_anno = json_data["annotations"]
                for list in json_anno:
                    temp = []
                    temp.append(int(list.get("start"))) # float -> int
                    temp.append(int(list.get("end"))) # float -> int

                    time_list.append(temp)

                oob_list = []
                for time in time_list:
                    for i in range(time[0], time[1] + 1):
                        oob_list.append(i)

            video = VideoReader(video_path)
            
            for i in range(len(video)):
                if i % 30 == 0 :
                    video_frame = video[i].asnumpy()
                    pil_image=Image.fromarray(video_frame)

                    if i in oob_list:
                        pil_image.save(save_dir_path + 'OutBody/{}_{}.jpg'.format(video_path[17:-4], str(i).zfill(10)))
                        continue
                    
                    pil_image.save(save_dir_path + 'InBody/{}_{}.jpg'.format(video_path[17:-4], str(i).zfill(10)))

            del video
        
        # Annotation file 미존재
        else:
            print(video_path)
            video = VideoReader(video_path)
            
            for i in range(len(video)):
                if i % 30 == 0 :
                    video_frame = video[i].asnumpy()
                    pil_image=Image.fromarray(video_frame)

                    pil_image.save(save_dir_path + 'InBody/{}_{}.jpg'.format(video_path[17:-4], str(i).zfill(10)))
                    # pil_image.save('/data/LAPA/Img_JH3/InBody/{}_{}.jpg'.format(video_path[17:-4], i))

            del video
        


    f.close()


def gen_robot_train_dataset(meta_file_path, device, save_dir_path):
    f = open(meta_file_path, 'r', encoding='utf-8')
    rdr = csv.reader(f)

    if not os.path.isdir(save_dir_path + 'InBody/'):
        os.mkdir(save_dir_path + 'InBody/')
    if not os.path.isdir(save_dir_path + 'OutBody/'):
        os.mkdir(save_dir_path + 'OutBody/')

    for line in rdr:
        video_path = line[0]

        # Annotation file 존재
        if len(line) == 2:
            anno_path = line[1]
            print(anno_path)


            with open(anno_path) as csv_file:
                next(csv_file)
                csv_file_reader = csv.reader(csv_file)

                time_list = []

                for l in csv_file_reader:
                    temp = []
                    temp.append(time_to_frame(l[0], 30))
                    temp.append(time_to_frame(l[1], 30))

                    time_list.append(temp)

                oob_list = []
                for time in time_list:
                    for i in range(time[0], time[1] + 1):
                        oob_list.append(i)

            video = VideoReader(video_path)
            
            for i in range(len(video)):
                if i % 30 == 0 :
                    video_frame = video[i].asnumpy()
                    pil_image=Image.fromarray(video_frame)

                    if i in oob_list:
                        pil_image.save(save_dir_path + 'OutBody/{}_{}.jpg'.format(video_path[17:-4], str(i).zfill(10)))
                        continue
                    
                    pil_image.save(save_dir_path + 'InBody/{}_{}.jpg'.format(video_path[17:-4], str(i).zfill(10)))

            del video
        
        # Annotation file 미존재
        else:
            print(video_path)
            video = VideoReader(video_path)
            
            for i in range(len(video)):
                if i % 30 == 0 :
                    video_frame = video[i].asnumpy()
                    pil_image=Image.fromarray(video_frame)
                    pil_image.save(save_dir_path + 'InBody/{}_{}.jpg'.format(video_path[17:-4], str(i).zfill(10)))

            del video

    f.close()


if __name__ == '__main__':
    gen_meta_file(args.video_dir_path, args.anno_dir_path, args.device, args.patient_list, args.save_dir_path)