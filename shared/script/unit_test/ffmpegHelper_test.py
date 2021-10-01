def main():
    print('main')

    import os
    import glob
    from core.utils.ffmpegHelper import ffmpegHelper

    from natsort import natsorted
    from shutil import copy
    from tqdm import tqdm

    video_path = "/data2/oob_module_test_results/2021-09-28/3b079065-48b3-443b-8156-b391ff0659c4/01_G_01_R_526_ch1_01.mp4"
    results_path = "/data2/oob_module_test_ffmpeg/assets/R_526/01_G_01_R_526_ch1_01"
    
    total_cutted_frame = os.path.join(results_path, 'frame_all')
    os.makedirs(total_cutted_frame, exist_ok=True)

    ffmepg_helper = ffmpegHelper(video_path, total_cutted_frame) # ffmpegHelper Module
    ffmepg_helper.cut_frame_total()

    img_list = glob.glob(os.path.join(total_cutted_frame, '*.jpg')) # ALL img into DB path
    img_list = natsorted(img_list) # sorting

    cutted_frame_1fps = os.path.join(results_path, 'frame_pick')
    os.makedirs(cutted_frame_1fps, exist_ok=True)

    for idx in tqdm(range(0, len(img_list), 30)) :
        copy(img_list[idx], cutted_frame_1fps)
    

if __name__ == '__main__':
    
    if __package__ is None:
        import sys
        from os import path
        sys.path.append(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))

    main()