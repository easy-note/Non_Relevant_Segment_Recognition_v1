"""
Generate train dataset (robot, lapa).
Usage:
    gen_train_dataset.py --patient_list <patient_list> --video_dir_path <video_dir_path> --anno_dir_path <anno_dir_path> --save_dir_path <save_dir_path>
"""
import os
import csv
import json
import argparse
from decord import VideoReader
from PIL import Image

robot_100_case = ['R_1', 'R_2', 'R_3', 'R_4', 'R_5', 'R_6', 'R_7', 'R_10', 'R_13', 'R_14', 
                  'R_15', 'R_17', 'R_18', 'R_19', 'R_22', 'R_48', 'R_56', 'R_74', 'R_76', 'R_84', 
                  'R_94', 'R_100', 'R_116', 'R_117', 'R_201', 'R_202', 'R_203', 'R_204', 'R_205', 'R_206', 
                  'R_207', 'R_208', 'R_209', 'R_210', 'R_301', 'R_302', 'R_303', 'R_304', 'R_305', 'R_313', 
                  'R_310', 'R_311', 'R_312', 'R_320', 'R_321', 'R_324', 'R_329', 'R_334', 'R_336', 'R_338', 
                  'R_339', 'R_340', 'R_342', 'R_345', 'R_346', 'R_347', 'R_348', 'R_349', 'R_355', 'R_357',
                  'R_358', 'R_362', 'R_363', 'R_369', 'R_372', 'R_376', 'R_378', 'R_379', 'R_386', 'R_391', 
                  'R_393', 'R_399', 'R_400', 'R_402', 'R_403', 'R_405', 'R_406', 'R_409', 'R_412', 'R_413', 
                  'R_415', 'R_418', 'R_419', 'R_420', 'R_423', 'R_424', 'R_427', 'R_436', 'R_445', 'R_449',
                  'R_455', 'R_480', 'R_493', 'R_501', 'R_510', 'R_522', 'R_523', 'R_526', 'R_532', 'R_533']

robot_60_case = [ 'R_310', 'R_311', 'R_312', 'R_320', 'R_321', 'R_324', 'R_329', 'R_334', 'R_336', 'R_338', 
                  'R_339', 'R_340', 'R_342', 'R_345', 'R_346', 'R_347', 'R_348', 'R_349', 'R_355', 'R_357',
                  'R_358', 'R_362', 'R_363', 'R_369', 'R_372', 'R_376', 'R_378', 'R_379', 'R_386', 'R_391', 
                  'R_393', 'R_399', 'R_400', 'R_402', 'R_403', 'R_405', 'R_406', 'R_409', 'R_412', 'R_413', 
                  'R_415', 'R_418', 'R_419', 'R_420', 'R_423', 'R_424', 'R_427', 'R_436', 'R_445', 'R_449',
                  'R_455', 'R_480', 'R_493', 'R_501', 'R_510', 'R_522', 'R_523', 'R_526', 'R_532', 'R_533']

# 아직 train dataset 생성 못함.
robot_last_case = ['R_415', 'R_418', 'R_424']

lapa_100_case = ['L_301', 'L_303', 'L_305', 'L_309', 'L_310', 'L_311', 'L_317', 'L_325', 'L_326', 'L_330', 
                 'L_333', 'L_340', 'L_346', 'L_349', 'L_367', 'L_370', 'L_377', 'L_379', 'L_385', 'L_387', 
                 'L_389', 'L_391', 'L_393', 'L_400', 'L_402', 'L_406', 'L_408', 'L_412', 'L_413', 'L_414', 
                 'L_415', 'L_418', 'L_419', 'L_421', 'L_423', 'L_427', 'L_428', 'L_430', 'L_433', 'L_434', 
                 'L_436', 'L_439', 'L_442', 'L_443', 'L_450', 'L_458', 'L_465', 'L_471', 'L_473', 'L_475', 
                 'L_477', 'L_478', 'L_479', 'L_481', 'L_482', 'L_484', 'L_491', 'L_493', 'L_496', 'L_507',
                 'L_513', 'L_514', 'L_515', 'L_517', 'L_522', 'L_534', 'L_535', 'L_537', 'L_539', 'L_542', 
                 'L_543', 'L_545', 'L_546', 'L_550', 'L_553', 'L_556', 'L_558', 'L_560', 'L_563', 'L_565', 
                 'L_568', 'L_569', 'L_572', 'L_574', 'L_575', 'L_577', 'L_580', 'L_586', 'L_595', 'L_605',
                 'L_607', 'L_625', 'L_631', 'L_647', 'L_654', 'L_659', 'L_660', 'L_661', 'L_669', 'L_676']

parser = argparse.ArgumentParser()
parser.add_argument('--patient_list', default = robot_100_case, type=str, help='Select patients to generate dataset')
parser.add_argument('--video_dir_path', default = '/data/ROBOT/Video', type=str, help='Video directory path')
parser.add_argument('--anno_dir_path', default = '/data/OOB/V2_100/ROBOT', type=str, help='Annotation file direcotry path')
parser.add_argument('--save_dir_path', default = '/data/ROBOT/V2_100/', type=str, help='Save directory path')

args, _ = parser.parse_known_args()

device = 'LAPA' if 'LAPA' in args.video_dir_path else 'ROBOT'

def time_to_frame(time, fps):
    t_segment = time.split(':')
    frame = (int(t_segment[0]) * 3600 * fps) + (int(t_segment[1]) * 60 * fps) + (int(t_segment[2]) * fps) # [h, m, s, ms] 

    return frame

def gen_meta_file(video_dir_path, anno_dir_path, origin_patient_list, save_dir_path):
    """Creating an Video-Annotation matching file.
    Args:
        video_dir_path: video directory path.
        anno_dir_path: annotation directory path.
        patient_list: patients to generate dataset.
        save_dir_path:  directory path to save matching file.
    """
    video_path = video_dir_path # '/data/LAPA/Video/'
    anno_path = anno_dir_path # '/data/OOB/'
    video_list = []
    anno_list = []
    patient_list = [patient+'_' for patient in origin_patient_list]
    # Lapa meta file
    if device == 'LAPA':
        # 사용할 전체 비디오
        # xx0, ch1, patient
        for root, dirs, files in os.walk(video_path):
            files.sort(key=str.lower)
            for file in files:
                if 'Syno' in file: 
                    continue
                if ('xx0' in file) or ('ch1' in file):
                    for patient in patient_list:
                        if patient in file:
                            video_list.append(file)
        # print(video_list)

        # 사용할 전체 Annotation file
        for root, dirs, files in os.walk(anno_path):
            files.sort(key=str.lower)
            for file in files:
                for patient in patient_list:
                    if patient in file:
                        anno_list.append(file)
        # print(anno_list)

    # Robot meta file
    elif device == 'ROBOT':
        # 사용할 전체 비디오
        for root, dirs, files in os.walk(video_path):
            files.sort(key=str.lower)
            for file in files:
                if ('Syno' in file):
                    continue
                for patient in patient_list:
                    if patient in file:
                        video_list.append(file)

        # 사용할 전체 Annotation file
        for root, dirs, files in os.walk(anno_path):
            files.sort(key=str.lower)
            for file in files:
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
        temp.append(video_path + '/' + video_list[0])
        
        # 만약 비디오에 해당하는 annotation file 이 있다면. 
        for anno_file in anno_list:
            if 'json' in anno_file:
                if anno_list[0][:-12] == video_list[0][:-4]:
                    temp.append(root + '/' + anno_list[0])
                    anno_list.pop(0)
            elif 'csv' in anno_file:
                if anno_list[0][:-11] == video_list[0][:-4]:
                    temp.append(root + '/' + anno_list[0])
                    anno_list.pop(0)
        
        # 리스트에서 입력한 비디오 삭제하기.
        video_list.pop(0)
        total_list.append(temp)
    
    # csv 파일 저장
    if not os.path.isdir(save_dir_path):
        os.mkdir(save_dir_path)
    
    output_file = save_dir_path + device + '_video-annotation_matching-V2.csv'
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        for i in total_list:
            writer.writerow(i)

def gen_train_dataset(meta_file_path, save_dir_path):
    f = open(meta_file_path, 'r', encoding='utf-8')
    rdr = csv.reader(f)
    
    if not os.path.isdir(save_dir_path + 'InBody/'):
        os.mkdir(save_dir_path + 'InBody/')
    if not os.path.isdir(save_dir_path + 'OutBody/'):
        os.mkdir(save_dir_path + 'OutBody/')
    
    for line in rdr:
        video_path = line[0]
        if len(line) > 2:
            print('ERROR: annotation file is duplicate!')
            exit(1)

        # Annotation file 존재
        if len(line) == 2:
            anno_path = line[1]
            print(anno_path)
            if 'json' in anno_path:
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

            elif 'csv' in anno_path:
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
            print('====\toob_list\t====\n {}'.format(oob_list))
            
            video = VideoReader(video_path)
            print('====\tVideoReader Complete\t====')
            
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
            print('====\tVideoReader Complete\t====')
            
            for i in range(len(video)):
                if i % 30 == 0 :
                    video_frame = video[i].asnumpy()
                    pil_image=Image.fromarray(video_frame)
                    pil_image.save(save_dir_path + 'InBody/{}_{}.jpg'.format(video_path[17:-4], str(i).zfill(10)))
            
            del video
    
    f.close()

if __name__ == '__main__':
    # gen_meta_file(args.video_dir_path, args.anno_dir_path, args.patient_list, args.save_dir_path)
    gen_train_dataset('/data/ROBOT/V2_100/ROBOT_video-annotation_matching-V2.csv', args.save_dir_path)
