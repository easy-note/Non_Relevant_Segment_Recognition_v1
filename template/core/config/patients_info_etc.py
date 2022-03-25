"""
    patients information file (fold)
"""

val_videos = {
    '01_GS1_03': ['R_16', 'R_20', 'R_21', 'R_30'],
    '01_GS3_06': ['R_6', 'R_46', 'R_154', 'R_155', 'R_156', 'R_157', 'R_158', 'R_160', 'R_161', 'R_162', 'R_163', 'R_164', 'R_165', 'R_166', 'R_167', 'R_168', 'R_169', 'R_170', 'R_171', 'R_172'],
    'all': ['R_16', 'R_20', 'R_21', 'R_30'] + ['R_6', 'R_46', 'R_154', 'R_155', 'R_156', 'R_157', 'R_158', 'R_160', 'R_161', 'R_162', 'R_163', 'R_164', 'R_165', 'R_166', 'R_167', 'R_168', 'R_169', 'R_170', 'R_171', 'R_172'],
    'free': ['R_6', 'R_46', 'R_154', 'R_155', 'R_156', 'R_157', 'R_158', 'R_160', 'R_161', 'R_162', 'R_163', 'R_164', 'R_165', 'R_166', 'R_167', 'R_168', 'R_169', 'R_170', 'R_171', 'R_172'], # 20 case
}

# img_db naming 기준 R_16_empty_01
# video, anno는 R_16_01
video_details = {
    '01_GS1_03': [ # 63ea # 4case
        'R_16_empty_01', 'R_16_empty_02', 'R_16_empty_03', 'R_16_empty_04', 'R_16_empty_05', 'R_16_empty_06', 'R_16_empty_07', 'R_16_empty_08', 'R_16_empty_09', 'R_16_empty_10',
        'R_16_empty_11', 'R_16_empty_12', 'R_16_empty_13', 'R_16_empty_14', 'R_16_empty_15', 'R_16_empty_16', 'R_16_empty_17', 'R_16_empty_18', 'R_16_empty_19', 'R_16_empty_20', 'R_16_empty_21',
        'R_16_empty_22', 'R_16_empty_23', 'R_16_empty_24', 'R_16_empty_25', 'R_16_empty_26', 'R_16_empty_27', 'R_16_empty_28', 'R_20_empty_01', 'R_20_empty_02', 'R_20_empty_03', 'R_20_empty_04',
        'R_20_empty_05', 'R_20_empty_06', 'R_20_empty_07', 'R_20_empty_08', 'R_20_empty_09', 'R_21_empty_01', 'R_21_empty_02', 'R_21_empty_03', 'R_21_empty_04', 'R_21_empty_05', 'R_21_empty_06',
        'R_21_empty_07', 'R_21_empty_08', 'R_21_empty_09', 'R_21_empty_10', 'R_21_empty_11', 'R_30_empty_01', 'R_30_empty_02', 'R_30_empty_03', 'R_30_empty_04', 'R_30_empty_05', 'R_30_empty_06',
        'R_30_empty_07', 'R_30_empty_08', 'R_30_empty_09', 'R_30_empty_10', 'R_30_empty_11', 'R_30_empty_12', 'R_30_empty_13', 'R_30_empty_14', 'R_30_empty_15'],


    '01_GS3_06': [ # 62ea (video) => 31ea(annotaion) # 20case
        'R_154_empty_01', 'R_154_empty_02', 'R_154_empty_03', 'R_154_empty_04', 'R_154_empty_05',
        'R_155_empty_01', 'R_155_empty_02', 'R_156_empty_01', 'R_156_empty_02', 'R_157_empty_01',
        'R_158_empty_01', 'R_160_empty_01', 'R_160_empty_02', 'R_161_empty_01', 'R_162_empty_01',
        'R_163_empty_01', 'R_164_empty_01', 'R_165_empty_01', 'R_166_empty_01', 'R_167_empty_01', 'R_167_empty_02', 'R_167_empty_03',
        'R_168_empty_01', 'R_169_empty_01', 'R_170_empty_01', 'R_171_empty_01', 'R_172_empty_01', 'R_46_empty_01', 'R_46_empty_02',
        'R_6_empty_01', 'R_6_empty_02'],

    'robot': [
        'R_16_empty_01', 'R_16_empty_02', 'R_16_empty_03', 'R_16_empty_04', 'R_16_empty_05', 'R_16_empty_06', 'R_16_empty_07', 'R_16_empty_08', 'R_16_empty_09', 'R_16_empty_10',
        'R_16_empty_11', 'R_16_empty_12', 'R_16_empty_13', 'R_16_empty_14', 'R_16_empty_15', 'R_16_empty_16', 'R_16_empty_17', 'R_16_empty_18', 'R_16_empty_19', 'R_16_empty_20', 'R_16_empty_21',
        'R_16_empty_22', 'R_16_empty_23', 'R_16_empty_24', 'R_16_empty_25', 'R_16_empty_26', 'R_16_empty_27', 'R_16_empty_28', 'R_20_empty_01', 'R_20_empty_02', 'R_20_empty_03', 'R_20_empty_04',
        'R_20_empty_05', 'R_20_empty_06', 'R_20_empty_07', 'R_20_empty_08', 'R_20_empty_09', 'R_21_empty_01', 'R_21_empty_02', 'R_21_empty_03', 'R_21_empty_04', 'R_21_empty_05', 'R_21_empty_06',
        'R_21_empty_07', 'R_21_empty_08', 'R_21_empty_09', 'R_21_empty_10', 'R_21_empty_11', 'R_30_empty_01', 'R_30_empty_02', 'R_30_empty_03', 'R_30_empty_04', 'R_30_empty_05', 'R_30_empty_06',
        'R_30_empty_07', 'R_30_empty_08', 'R_30_empty_09', 'R_30_empty_10', 'R_30_empty_11', 'R_30_empty_12', 'R_30_empty_13', 'R_30_empty_14', 'R_30_empty_15'] +
            [
        'R_154_empty_01', 'R_154_empty_02', 'R_154_empty_03', 'R_154_empty_04', 'R_154_empty_05',
        'R_155_empty_01', 'R_155_empty_02', 'R_156_empty_01', 'R_156_empty_02', 'R_157_empty_01',
        'R_158_empty_01', 'R_160_empty_01', 'R_160_empty_02', 'R_161_empty_01', 'R_162_empty_01',
        'R_163_empty_01', 'R_164_empty_01', 'R_165_empty_01', 'R_166_empty_01', 'R_167_empty_01', 'R_167_empty_02', 'R_167_empty_03',
        'R_168_empty_01', 'R_169_empty_01', 'R_170_empty_01', 'R_171_empty_01', 'R_172_empty_01', 'R_46_empty_01', 'R_46_empty_02',
        'R_6_empty_01', 'R_6_empty_02'],
}