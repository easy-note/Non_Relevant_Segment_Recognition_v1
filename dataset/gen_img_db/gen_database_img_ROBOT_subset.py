import os
import subprocess
import natsort

import re

'''
# 225ea & 100case
OOB_robot_list = [
    'R_1_ch1_01', 'R_1_ch1_03', 'R_1_ch1_06', 'R_2_ch1_01', 'R_2_ch1_03', 'R_3_ch1_01', 'R_3_ch1_03', 'R_3_ch1_05', 'R_4_ch1_01', 'R_4_ch1_04', 
    'R_5_ch1_01', 'R_5_ch1_03', 'R_6_ch1_01', 'R_6_ch1_03', 'R_6_ch1_05', 'R_7_ch1_01', 'R_7_ch1_04', 'R_10_ch1_01', 'R_10_ch1_03', 'R_13_ch1_01',
    'R_13_ch1_03', 'R_14_ch1_01', 'R_14_ch1_03', 'R_14_ch1_05', 'R_15_ch1_01', 'R_15_ch1_03', 'R_17_ch1_01', 'R_17_ch1_04', 'R_17_ch1_06', 'R_18_ch1_01',
    'R_18_ch1_04', 'R_19_ch1_01', 'R_19_ch1_03', 'R_19_ch1_05', 'R_22_ch1_01', 'R_22_ch1_03', 'R_22_ch1_05', 'R_48_ch1_01', 'R_48_ch1_02', 'R_56_ch1_01',
    'R_56_ch1_03', 'R_74_ch1_01', 'R_74_ch1_03', 'R_76_ch1_01', 'R_76_ch1_03', 'R_84_ch1_01', 'R_84_ch1_03', 'R_94_ch1_01', 'R_94_ch1_03', 'R_100_ch1_01',
    'R_100_ch1_03', 'R_100_ch1_05', 'R_116_ch1_01', 'R_116_ch1_03', 'R_116_ch1_06', 'R_117_ch1_01', 'R_117_ch1_03', 'R_201_ch1_01', 'R_201_ch1_03', 'R_202_ch1_01', 
    'R_202_ch1_03', 'R_202_ch1_05', 'R_203_ch1_01', 'R_203_ch1_03', 'R_204_ch1_01', 'R_204_ch1_02', 'R_205_ch1_01', 'R_205_ch1_03', 'R_205_ch1_05', 'R_206_ch1_01', 
    'R_206_ch1_03', 'R_207_ch1_01', 'R_207_ch1_03', 'R_208_ch1_01', 'R_208_ch1_03', 'R_209_ch1_01', 'R_209_ch1_03', 'R_210_ch1_01', 'R_210_ch2_04', 'R_301_ch1_01', 
    'R_301_ch1_04', 'R_302_ch1_01', 'R_302_ch1_04', 'R_303_ch1_01', 'R_303_ch1_04', 'R_304_ch1_01', 'R_304_ch1_03', 'R_305_ch1_01', 'R_305_ch1_04', 'R_310_ch1_01', 
    'R_310_ch1_03', 'R_311_ch1_01', 'R_311_ch1_03', 'R_312_ch1_02', 'R_312_ch1_03', 'R_313_ch1_01', 'R_313_ch1_03', 'R_320_ch1_01', 'R_320_ch1_03', 'R_321_ch1_01', 
    'R_321_ch1_03', 'R_321_ch1_05', 'R_324_ch1_01', 'R_324_ch1_03', 'R_329_ch1_01', 'R_329_ch1_03', 'R_334_ch1_01', 'R_334_ch1_03', 'R_336_ch1_01', 'R_336_ch1_04', 
    'R_338_ch1_01', 'R_338_ch1_03', 'R_338_ch1_05', 'R_339_ch1_01', 'R_339_ch1_03', 'R_339_ch1_05', 'R_340_ch1_01', 'R_340_ch1_03', 'R_340_ch1_05', 'R_342_ch1_01', 
    'R_342_ch1_03', 'R_342_ch1_05', 'R_345_ch1_01', 'R_345_ch1_04', 'R_346_ch1_02', 'R_346_ch1_04', 'R_347_ch1_02', 'R_347_ch1_03', 'R_347_ch1_05', 'R_348_ch1_01', 
    'R_348_ch1_03', 'R_349_ch1_01', 'R_349_ch1_04', 'R_355_ch1_02', 'R_355_ch1_04', 'R_357_ch1_01', 'R_357_ch1_03', 'R_357_ch1_05', 'R_358_ch1_01', 'R_358_ch1_03', 
    'R_358_ch1_05', 'R_362_ch1_01', 'R_362_ch1_03', 'R_362_ch1_05', 'R_363_ch1_01', 'R_363_ch1_03', 'R_369_ch1_01', 'R_369_ch1_03', 'R_372_ch1_01', 'R_372_ch1_04', 
    'R_376_ch1_01', 'R_376_ch1_03', 'R_376_ch1_05', 'R_378_ch1_01', 'R_378_ch1_03', 'R_378_ch1_05', 'R_379_ch1_02', 'R_379_ch1_04', 'R_386_ch1_01', 'R_386_ch1_03', 
    'R_391_ch1_01', 'R_391_ch1_03', 'R_391_ch2_06', 'R_393_ch1_01', 'R_393_ch1_04', 'R_399_ch1_01', 'R_399_ch1_04', 'R_400_ch1_01', 'R_400_ch1_03', 'R_402_ch1_01', 
    'R_402_ch1_03', 'R_403_ch1_01', 'R_403_ch1_03', 'R_405_ch1_01', 'R_405_ch1_03', 'R_405_ch1_05', 'R_406_ch1_02', 'R_406_ch1_04', 'R_406_ch1_06', 'R_409_ch1_01', 
    'R_409_ch1_03', 'R_412_ch1_01', 'R_412_ch1_03', 'R_413_ch1_02', 'R_413_ch1_04', 'R_415_ch1_01', 'R_415_ch1_03', 'R_415_ch1_05', 'R_418_ch1_02', 'R_418_ch1_04', 
    'R_418_ch1_06', 'R_419_ch1_01', 'R_419_ch1_04', 'R_420_ch1_01', 'R_420_ch1_03', 'R_423_ch1_01', 'R_423_ch1_03', 'R_424_ch2_02', 'R_424_ch2_04', 'R_427_ch1_01', 
    'R_427_ch1_03', 'R_436_ch1_02', 'R_436_ch1_04', 'R_436_ch1_06', 'R_436_ch1_08', 'R_436_ch1_10', 'R_445_ch1_01', 'R_445_ch1_03', 'R_449_ch1_01', 'R_449_ch1_04', 
    'R_449_ch1_06', 'R_455_ch1_01', 'R_455_ch1_03', 'R_455_ch1_05', 'R_480_ch1_01', 'R_493_ch1_01', 'R_493_ch1_03', 'R_501_ch1_01', 'R_510_ch1_01', 'R_510_ch1_03', 
    'R_522_ch1_01', 'R_523_ch1_01', 'R_526_ch1_01', 'R_532_ch1_01', 'R_533_ch1_01']

# 91ea, 40case #### 12번 서버 전체 완료
OOB_robot_40 = [
    'R_1_ch1_01', 'R_1_ch1_03', 'R_1_ch1_06', 'R_2_ch1_01', 'R_2_ch1_03', 'R_3_ch1_01', 'R_3_ch1_03', 'R_3_ch1_05', 'R_4_ch1_01', 'R_4_ch1_04', 
    'R_5_ch1_01', 'R_5_ch1_03', 'R_6_ch1_01', 'R_6_ch1_03', 'R_6_ch1_05', 'R_7_ch1_01', 'R_7_ch1_04', 'R_10_ch1_01', 'R_10_ch1_03', 'R_13_ch1_01', 
    'R_13_ch1_03', 'R_14_ch1_01', 'R_14_ch1_03', 'R_14_ch1_05', 'R_15_ch1_01', 'R_15_ch1_03', 'R_17_ch1_01', 'R_17_ch1_04', 'R_17_ch1_06', 'R_18_ch1_01', 
    'R_18_ch1_04', 'R_19_ch1_01', 'R_19_ch1_03', 'R_19_ch1_05', 'R_22_ch1_01', 'R_22_ch1_03', 'R_22_ch1_05', 'R_48_ch1_01', 'R_48_ch1_02', 'R_56_ch1_01', 
    'R_56_ch1_03', 'R_74_ch1_01', 'R_74_ch1_03', 'R_76_ch1_01', 'R_76_ch1_03', 'R_84_ch1_01', 'R_84_ch1_03', 'R_94_ch1_01', 'R_94_ch1_03', 'R_100_ch1_01', 
    'R_100_ch1_03', 'R_100_ch1_05', 'R_116_ch1_01', 'R_116_ch1_03', 'R_116_ch1_06', 'R_117_ch1_01', 'R_117_ch1_03', 'R_201_ch1_01', 'R_201_ch1_03', 'R_202_ch1_01', 
    'R_202_ch1_03', 'R_202_ch1_05', 'R_203_ch1_01', 'R_203_ch1_03', 'R_204_ch1_01', 'R_204_ch1_02', 'R_205_ch1_01', 'R_205_ch1_03', 'R_205_ch1_05', 'R_206_ch1_01', 
    'R_206_ch1_03', 'R_207_ch1_01', 'R_207_ch1_03', 'R_208_ch1_01', 'R_208_ch1_03', 'R_209_ch1_01', 'R_209_ch1_03', 'R_210_ch1_01', 'R_210_ch2_04', 'R_301_ch1_01', 
    'R_301_ch1_04', 'R_302_ch1_01', 'R_302_ch1_04', 'R_303_ch1_01', 'R_303_ch1_04', 'R_304_ch1_01', 'R_304_ch1_03', 'R_305_ch1_01', 'R_305_ch1_04', 'R_313_ch1_01', 'R_313_ch1_03']

# 134ea, 60case #### 2021.07.29.10:06 시작 
OOB_robot_60 = [
    'R_310_ch1_01', 'R_310_ch1_03', 'R_311_ch1_01', 'R_311_ch1_03', 'R_312_ch1_02', 'R_312_ch1_03', 'R_320_ch1_01', 'R_320_ch1_03', 'R_321_ch1_01', 'R_321_ch1_03', 
    'R_321_ch1_05', 'R_324_ch1_01', 'R_324_ch1_03', 'R_329_ch1_01', 'R_329_ch1_03', 'R_334_ch1_01', 'R_334_ch1_03', 'R_336_ch1_01', 'R_336_ch1_04', 'R_338_ch1_01', 
    'R_338_ch1_03', 'R_338_ch1_05', 'R_339_ch1_01', 'R_339_ch1_03', 'R_339_ch1_05', 'R_340_ch1_01', 'R_340_ch1_03', 'R_340_ch1_05', 'R_342_ch1_01', 'R_342_ch1_03', 
    'R_342_ch1_05', 'R_345_ch1_01', 'R_345_ch1_04', 'R_346_ch1_02', 'R_346_ch1_04', 'R_347_ch1_02', 'R_347_ch1_03', 'R_347_ch1_05', 'R_348_ch1_01', 'R_348_ch1_03', 
    'R_349_ch1_01', 'R_349_ch1_04', 'R_355_ch1_02', 'R_355_ch1_04', 'R_357_ch1_01', 'R_357_ch1_03', 'R_357_ch1_05', 'R_358_ch1_01', 'R_358_ch1_03', 'R_358_ch1_05', 
    'R_362_ch1_01', 'R_362_ch1_03', 'R_362_ch1_05', 'R_363_ch1_01', 'R_363_ch1_03', 'R_369_ch1_01', 'R_369_ch1_03', 'R_372_ch1_01', 'R_372_ch1_04', 'R_376_ch1_01', 
    'R_376_ch1_03', 'R_376_ch1_05', 'R_378_ch1_01', 'R_378_ch1_03', 'R_378_ch1_05', 'R_379_ch1_02', 'R_379_ch1_04', 'R_386_ch1_01', 'R_386_ch1_03', 'R_391_ch1_01', 
    'R_391_ch1_03', 'R_391_ch2_06', 'R_393_ch1_01', 'R_393_ch1_04', 'R_399_ch1_01', 'R_399_ch1_04', 'R_400_ch1_01', 'R_400_ch1_03', 'R_402_ch1_01', 'R_402_ch1_03', 
    'R_403_ch1_01', 'R_403_ch1_03', 'R_405_ch1_01', 'R_405_ch1_03', 'R_405_ch1_05', 'R_406_ch1_02', 'R_406_ch1_04', 'R_406_ch1_06', 'R_409_ch1_01', 'R_409_ch1_03', 
    'R_412_ch1_01', 'R_412_ch1_03', 'R_413_ch1_02', 'R_413_ch1_04', 'R_415_ch1_01', 'R_415_ch1_03', 'R_415_ch1_05', 'R_418_ch1_02', 'R_418_ch1_04', 'R_418_ch1_06', 
    'R_419_ch1_01', 'R_419_ch1_04', 'R_420_ch1_01', 'R_420_ch1_03', 'R_423_ch1_01', 'R_423_ch1_03', 'R_424_ch2_02', 'R_424_ch2_04', 'R_427_ch1_01', 'R_427_ch1_03', 
    'R_436_ch1_02', 'R_436_ch1_04', 'R_436_ch1_06', 'R_436_ch1_08', 'R_436_ch1_10', 'R_445_ch1_01', 'R_445_ch1_03', 'R_449_ch1_01', 'R_449_ch1_04', 'R_449_ch1_06', 
    'R_455_ch1_01', 'R_455_ch1_03', 'R_455_ch1_05', 'R_480_ch1_01', 'R_493_ch1_01', 'R_493_ch1_03', 'R_501_ch1_01', 'R_510_ch1_01', 'R_510_ch1_03', 'R_522_ch1_01', 
    'R_523_ch1_01', 'R_526_ch1_01', 'R_532_ch1_01', 'R_533_ch1_01']
'''

OOB_robot_40 = [
    'R_116_ch1_03', 'R_205_ch1_03', 'R_304_ch1_03'
]

OOB_robot_60 = [
    'R_403_ch1_03', 'R_415_ch1_03', 'R_415_ch1_05', 'R_510_ch1_01'
]

robot_40_video_base_path = '/data1/HuToM/Video_Robot_cordname'
robot_60_video_base_path = '/data2/Video/Robot/Dataset2_60case'

img_base_path = '/raid/img_db_subset'

def convert_patient_num(patient_num):
    return ''.join(re.findall('[1-9]\d*', patient_num))

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('ERROR: Creating directory {}'.format(directory))

def gen_dataset_using_ffmpeg(input_video_path, output_dir_path):
    # ffmpeg -i 'input_vidio_path' 'output_dir_path/01_G_01_R_1_ch1_3-%010d.jpg'
    print('\nProcessing ====>\t {}\n'.format(os.path.join(output_dir_path, '{}'.format(output_dir_path.split('/')[-1]))))
    output_img_path = os.path.join(output_dir_path, '{}-%010d.jpg'.format(output_dir_path.split('/')[-1]))
    # cmd = ['ffmpeg', '-i', input_video_path, '-start_number', '0', '-vsync', '0', '-vf', 'scale=512:512']
    # cmd = ['ffmpeg', '-i', input_video_path, '-start_number', '0', '-vsync', '0']
    cmd = ['ffmpeg', '-i', input_video_path, '-start_number', '0', '-vf', 'scale=512:512']
    cmd += [output_img_path]

    print('Running: ', " ".join(cmd))
    subprocess.run(cmd)

def save_log(save_path, log_txt):
    with open(save_path, 'a') as f:
        f.write(log_txt)

def main_40case():
    for (root, dirs, files) in os.walk(robot_40_video_base_path):
        files = natsort.natsorted(files)

        for file in files:
            if re.search('r\d{6}/ch\d_video_\d{2}[.]mp4', os.path.join(root, file).lower()): # ./R000001/ch1_video_03.mp4
                patient_num = os.path.join(root, file).split('/')[-2] # R000001
                patient_num = convert_patient_num(patient_num) # R000001 -> 1

                channel = os.path.join(root, file).split('/')[-1][:3] # ch1
                video_num = os.path.join(root, file).split('/')[-1].split('_')[2][:2] # 03

                # 파일 이름 재정의.
                rename_file = 'R_{}_{}_{}'.format(patient_num, channel, video_num) # R_1_ch1_03
                full_rename_file = '01_G_01_{}'.format(rename_file) # 01_G_01_R_1_ch1_03
                
                
                if rename_file in OOB_robot_40:
                    output_dir_path = os.path.join(img_base_path, 'ROBOT_sub', 'R_{}'.format(patient_num), full_rename_file)
                    
                    createFolder(output_dir_path)
                    gen_dataset_using_ffmpeg(os.path.join(root, file), output_dir_path)
                    save_log(os.path.join(img_base_path, 'ROBOT_sub_database_log.txt'), 'Robot_40case | Origin_video: {}\t|\t Rename_file: {}\n'.format(os.path.join(root, file), full_rename_file))
                    print('Robot_40case | Origin_video: {}\t|\t Rename_file: {}\n'.format(os.path.join(root, file), full_rename_file))

            
            # 76번 예외 비디오
            elif re.search('r000076/ch1_video_01_6915320_rdg.mp4', os.path.join(root, file).lower()):
                patient_num = os.path.join(root, file).split('/')[-2] 
                patient_num = convert_patient_num(patient_num)

                channel = os.path.join(root, file).split('/')[-1][:3] 
                video_num = os.path.join(root, file).split('/')[-1].split('_')[2][:2] 

                # 파일 이름 재정의.
                rename_file = 'R_{}_{}_{}'.format(patient_num, channel, video_num)
                full_rename_file = '01_G_01_{}'.format(rename_file) # 01_G_01_R_1_ch1_03
                
                if rename_file in OOB_robot_40:
                    output_dir_path = os.path.join(img_base_path, 'ROBOT_sub', 'R_{}'.format(patient_num), full_rename_file)
                    
                    createFolder(output_dir_path)
                    gen_dataset_using_ffmpeg(os.path.join(root, file), output_dir_path)
                    save_log(os.path.join(img_base_path, 'ROBOT_sub_database_log.txt'), 'Robot_40case | Origin_video: {}\t|\t Rename_file: {}\n'.format(os.path.join(root, file), full_rename_file))
                    print('Robot_40case | Origin_video: {}\t|\t Rename_file: {}\n'.format(os.path.join(root, file), full_rename_file))

            # 84번 예외 비디오
            elif re.search('r000084/ch1_video_01_8459178_robotic subtotal.mp4', os.path.join(root, file).lower()):
                patient_num = os.path.join(root, file).split('/')[-2] 
                patient_num = convert_patient_num(patient_num) 

                channel = os.path.join(root, file).split('/')[-1][:3]
                video_num = os.path.join(root, file).split('/')[-1].split('_')[2][:2] 

                # 파일 이름 재정의.
                rename_file = 'R_{}_{}_{}'.format(patient_num, channel, video_num) 
                full_rename_file = '01_G_01_{}'.format(rename_file) # 01_G_01_R_1_ch1_03
                
                if rename_file in OOB_robot_40:
                    output_dir_path = os.path.join(img_base_path, 'ROBOT_sub', 'R_{}'.format(patient_num), full_rename_file)
                    
                    createFolder(output_dir_path)
                    gen_dataset_using_ffmpeg(os.path.join(root, file), output_dir_path)
                    save_log(os.path.join(img_base_path, 'ROBOT_sub_database_log.txt'), 'Robot_40case | Origin_video: {}\t|\t Rename_file: {}\n'.format(os.path.join(root, file), full_rename_file))
                    print('Robot_40case | Origin_video: {}\t|\t Rename_file: {}\n'.format(os.path.join(root, file), full_rename_file))


def main_60case():
    for (root, dirs, files) in os.walk(robot_60_video_base_path):
        files = natsort.natsorted(files)

        for file in files:
            if re.search('r\d{6}/ch\d_video_\d{2}[.]mp4', os.path.join(root, file).lower()): # ./R000001/ch1_video_03.mp4
                patient_num = os.path.join(root, file).split('/')[-2] # R000001
                patient_num = convert_patient_num(patient_num) # R000001 -> 1

                channel = os.path.join(root, file).split('/')[-1][:3] # ch1
                video_num = os.path.join(root, file).split('/')[-1].split('_')[2][:2] # 03

                # 파일 이름 재정의.
                rename_file = 'R_{}_{}_{}'.format(patient_num, channel, video_num) # R_1_ch1_03
                full_rename_file = '01_G_01_{}'.format(rename_file)

                if rename_file in OOB_robot_60: # R_391_ch2_06 비디오 없음.
                    output_dir_path = os.path.join(img_base_path, 'ROBOT_sub', 'R_{}'.format(patient_num), full_rename_file)
                    createFolder(output_dir_path)

                    gen_dataset_using_ffmpeg(os.path.join(root, file), output_dir_path)
                    save_log(os.path.join(img_base_path, 'ROBOT_sub_database_log.txt'), 'Robot_60case | Origin_video: {}\t|\t Rename_file: {}\n'.format(os.path.join(root, file), full_rename_file))
                    print('Robot_60case | Origin_video: {}\t|\t Rename_file: {}\n'.format(os.path.join(root, file), full_rename_file))

           # R_391 예외 비디오.
            elif re.search('r000391/01_g_01_r_391_ch2_06.mp4', os.path.join(root, file).lower()):
                full_rename_file = os.path.join(root, file).split('/')[-1].split('.')[0]
                patient_num = full_rename_file.split('_')[4]

                output_dir_path = os.path.join(img_base_path, 'ROBOT_sub' , 'R_{}'.format(patient_num), full_rename_file)
                createFolder(output_dir_path)

                gen_dataset_using_ffmpeg(os.path.join(root, file), output_dir_path)
                save_log(os.path.join(img_base_path, 'ROBOT_sub_database_log.txt'), 'Robot_60case | Origin_video: {}\t|\t Rename_file: {}\n'.format(os.path.join(root, file), full_rename_file))
                print('Robot_60case | Origin_video: {}\t|\t Rename_file: {}\n'.format(os.path.join(root, file), rename_file))


def get_occurrence_count(target_list):
    new_list = {}
    for i in target_list:
        try: new_list[i] += 1
        except: new_list[i] = 1
    
    return new_list

if __name__ == '__main__':
    # main_40case()
    # main_60case()

