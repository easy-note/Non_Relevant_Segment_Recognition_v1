def make_etc24():

    import os
    import glob
    from tqdm import tqdm

    from core.config.assets_info import video_path, annotation_path, img_db_path
    from core.utils.ffmpegHelper import ffmpegHelper
    from core.utils.parser import InfoParser
    from core.utils.logger import LogHelper

    '''
    data_path = '/data3/DATA/IMPORT/211220/12_14/TEST_24case'
    save_path = '/raid/img_db/ETC24/'
    '''

    data_dir = video_path['etc24'][0]
    save_dir = img_db_path['etc24']

    '''
    original ROBOT 100
    @ ~R_7/01_G_01_R_7_ch1_01/01_G_01_R_7_ch1_01-0000111032.jpg

    no channel info on etc24 dataset
    @ ~/01_GS1_03/R_16/01_GS1_03_R_16_01.mpg
    @ ~/01_GS1_03/R_16/01_GS1_03_R_16_02.mpg ...
    
    @ ~/01_GS3_06/R_6/01_GS3_06_R_6_01.mp4
    @ ~/01_GS3_06/R_6/01_GS3_06_R_6_02.mp4 ...
    '''
    
    # 0. data parsing and extract only video file
    data_path = []
    all_data_path = glob.glob(os.path.join(data_dir, '*', '*', '*'))
    
    for path in all_data_path:
        f_name, ext = os.path.splitext(path)
        if ext.lower() in ['.mp4', '.mpg']:
            data_path.append(path)

    info_parser = InfoParser('ETC_VIDEO_1')
    log_helper = LogHelper(os.path.join(save_dir, 'ETC24-frame_info_log.txt'))
    # log_helper.writeln('{} | {}\t\t ======> \t\t{} '.format('no', 'origin path', 'target path'))
    log_helper.writeln('{} | {}\t\t ======> \t\t{} \t | {} | {} | {}'.format('no', 'origin path', 'target path', 'cutted_frame_len', 'ffmpeg_video_len', 'ffmpeg_video_fps'))
    
    for i, path in tqdm(enumerate(data_path)): # video paths        

        # 1. redefine video name
        info_parser.write_file_name(path)
        info = info_parser.get_info()

        save_file_name = '_'.join([info['hospital'], info['surgery_type'], info['surgeon'], info['op_method'], info['patient_idx'], info['video_channel'], info['video_slice_no']])
        patient_no = info_parser.get_patient_no()
        video_name = info_parser.get_video_name()

        # 2. set save path
        target_dir = os.path.join(save_dir, patient_no, video_name)

        os.makedirs(target_dir, exist_ok=True)

        # 3. cut frame
        ffmpeg_helper = ffmpegHelper(video_path=path, results_dir=target_dir)
        # ffmpeg_helper.cut_frame_total(save_name=save_file_name)
        ffmpeg_video_len = ffmpeg_helper.get_video_length()
        ffmpeg_video_fps = ffmpeg_helper.get_video_fps()      
        cutted_frame_len = len(glob.glob(os.path.join(target_dir, '*.jpg')))

        # 4. loggin
        # log_txt = '{} | {}\t\t ======> \t\t{}'.format(i, path, target_dir)
        log_txt = '{} | {}\t\t ======> \t\t{} \t | {} | {} | {}'.format(i, path, target_dir, cutted_frame_len, ffmpeg_video_len, ffmpeg_video_fps)
        print(log_txt)
        log_helper.writeln(log_txt)


if __name__ == '__main__':
    if __package__ is None:
        import sys
        import os
        from os import path    
        base_path = path.dirname(path.dirname(path.abspath(__file__)))
        sys.path.append(base_path)
        sys.path.append(base_path+'/core/accessory/RepVGG')
        print(base_path)
        

        make_etc24()