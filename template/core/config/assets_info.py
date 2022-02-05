
"""
    annotation, img_db path
    oob_assets save path
"""

video_path = {
    # dataset 1 (40 case) + dataset 2 (60 case)
    'robot': ['/data1/HuToM/Video_Robot_cordname', '/data2/Video/Robot/Dataset2_60case'],
    'etc24': ['/data3/DATA/IMPORT/211220/12_14/TEST_24case'],
    'gangbuksamsung': ['/data3/DATA/IMPORT/211220/12_14/gangbuksamsung_127case'],
    'severance_1st' :['/data3/DATA/IMPORT/211220/12_14/severance_1st'],
    'severance_2nd': ['/data3/DATA/IMPORT/211220/12_14/severance_2nd'],
}

annotation_path = {
    'annotation_v1_base_path': '/data2/Public/IDC_21.06.25/ANNOTATION/Gastrectomy/Event/OOB/V1',
    'annotation_v2_base_path': '/data2/Public/IDC_21.06.25/ANNOTATION/Gastrectomy/Event/OOB/V2',
    'annotation_v3_base_path': '/data2/Public/IDC_21.06.25/ANNOTATION/Gastrectomy/Event/OOB/V3/TBE',
    'annotation_etc24_base_path': '/data2/Public/OOB_Recog/annotation/ROBOT/etc/etc24',
    'annotation_gangbuksamsung_base_path': '/data3/Public/NRS_Recog/annotation/Gastrectomy/Lapa/v3/gangbuksamsung_127case/NRS',
    'annotation_severance_1st_base_path': '/data3/Public/NRS_Recog/annotation/Gastrectomy/Lapa/v3/severance_1st/NRS',
    'annotation_severance_2nd_base_path': '/data3/Public/NRS_Recog/annotation/Gastrectomy/Lapa/v3/severance_2nd/NRS',
}

img_db_path = {
    'robot': '/raid/img_db/ROBOT', ### cuaton @@@ changed key @@@ 12 -> robot
    'etc24': '/raid/img_db/ETC24',
    'gangbuksamsung': '/raid/img_db/VIHUB/gangbuksamsung_127case',
    'severance_1st': '/raid/img_db/VIHUB/severance_1st',
    'severance_2nd': '/raid/img_db/VIHUB/severance_2nd',
}

oob_assets_save_path = {
    'oob_assets_v1_robot_save_path':  '/data2/Public/OOB_Recog/oob_assets/V1/ROBOT',
    'oob_assets_v1_lapa_save_path':'/data2/Public/OOB_Recog/oob_assets/V1/LAPA',

    'oob_assets_v2_robot_save_path':'/raid/img_db/oob_assets/V2/ROBOT',
    'oob_assets_v2_lapa_save_path': '/raid/img_db/oob_assets/V2/LAPA',

    'oob_assets_v3_robot_save_path': '/raid/img_db/oob_assets/V3/ROBOT',

    'theator-oob_assets_v3_robot_save_path': '/raid/img_db/oob_assets/V3/theator-ROBOT',

    'vihub_assets_v3_lapa_save_path': '/raid/img_db/oob_assets/V3/LAPA',
}

mc_assets_save_path = {
    'robot': '/data2/Public/OOB_Recog/offline',
}