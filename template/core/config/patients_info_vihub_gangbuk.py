"""
    patients information file (fold)
"""

val_videos = {
    # 117 case (115, 2case) // L_79, L_118 not contained all procedure
    'all': ['L_1', 'L_2', 'L_3', 'L_4', 'L_5', 'L_6', 'L_7', 'L_8', 'L_9', 'L_10', 'L_11', 'L_12', 'L_13', 'L_14', 'L_16', 'L_17', 'L_18', 'L_19', 'L_20',
            'L_21', 'L_22', 'L_25', 'L_26', 'L_27', 'L_28', 'L_29', 'L_30', 'L_32', 'L_34', 'L_36', 'L_37', 'L_38', 'L_39', 'L_40', 'L_41', 'L_42', 'L_43',
            'L_44', 'L_45', 'L_46', 'L_47', 'L_48', 'L_49', 'L_50', 'L_51', 'L_52', 'L_53', 'L_55', 'L_56', 'L_57', 'L_58', 'L_59', 'L_60', 'L_61', 'L_62',
            'L_63', 'L_64', 'L_65', 'L_66', 'L_67', 'L_68', 'L_69', 'L_70', 'L_71', 'L_72', 'L_73', 'L_74', 'L_75', 'L_76', 'L_77', 'L_79', 'L_80', 'L_81',
            'L_82', 'L_83', 'L_84', 'L_85', 'L_86', 'L_87', 'L_88', 'L_89', 'L_90', 'L_91', 'L_92', 'L_94', 'L_95', 'L_96', 'L_97', 'L_98', 'L_99', 'L_100',
            'L_101', 'L_102', 'L_103', 'L_104', 'L_105', 'L_106', 'L_107', 'L_108', 'L_109', 'L_110', 'L_111', 'L_112', 'L_113', 'L_114', 'L_115', 'L_116',
            'L_117', 'L_118', 'L_120', 'L_121', 'L_122', 'L_123', 'L_124', 'L_125', 'L_126', 'L_127'],
    'free': ['L_20', 'L_21'],
}


# img_db naming 기준 R_16_empty_01
# video, anno는 R_16_01
video_details = {
    'lapa': ['L_1_empty_01', 'L_1_empty_02', 'L_2_empty_01', 'L_3_empty_01', 'L_4_empty_01', 'L_4_empty_02', 'L_4_empty_03', 'L_4_empty_04', 'L_5_empty_01', 'L_5_empty_02',
    'L_6_empty_01', 'L_7_empty_01', 'L_8_empty_01', 'L_9_empty_01', 'L_10_empty_01', 'L_11_empty_01', 'L_12_empty_01', 'L_13_empty_01', 'L_14_empty_01', 'L_16_empty_01',
    'L_17_empty_01', 'L_17_empty_02', 'L_18_empty_01', 'L_19_empty_01', 'L_20_empty_01', 'L_20_empty_02', 'L_21_empty_01', 'L_21_empty_02', 'L_21_empty_03', 'L_22_empty_01', 'L_22_empty_02',
    'L_25_empty_01', 'L_26_empty_01', 'L_26_empty_02', 'L_27_empty_01', 'L_28_empty_01', 'L_29_empty_01', 'L_30_empty_01', 'L_30_empty_02', 'L_32_empty_01', 'L_34_empty_01', 'L_36_empty_01',
    'L_37_empty_01', 'L_38_empty_01', 'L_39_empty_01', 'L_40_empty_01', 'L_40_empty_02', 'L_41_empty_01', 'L_42_empty_01', 'L_43_empty_01', 'L_44_empty_01', 'L_45_empty_01', 'L_46_empty_01',
    'L_47_empty_01', 'L_47_empty_02', 'L_47_empty_03', 'L_48_empty_01', 'L_49_empty_01', 'L_49_empty_02', 'L_49_empty_03', 'L_49_empty_04', 'L_50_empty_01',
    'L_51_empty_01', 'L_51_empty_02', 'L_51_empty_03', 'L_52_empty_01', 'L_52_empty_02', 'L_53_empty_01', 'L_55_empty_01', 'L_55_empty_02', 'L_55_empty_03',
    'L_56_empty_01', 'L_56_empty_02', 'L_57_empty_01', 'L_58_empty_01', 'L_58_empty_02', 'L_58_empty_03', 'L_59_empty_01', 'L_60_empty_01', 'L_60_empty_02',
    'L_61_empty_01', 'L_61_empty_02', 'L_61_empty_03', 'L_62_empty_01', 'L_63_empty_01', 'L_64_empty_01', 'L_65_empty_01', 'L_65_empty_02', 'L_65_empty_03', 'L_65_empty_04', 'L_65_empty_05',
    'L_66_empty_01', 'L_67_empty_01', 'L_67_empty_02', 'L_67_empty_03', 'L_68_empty_01', 'L_69_empty_01', 'L_70_empty_01', 'L_70_empty_02', 'L_70_empty_03', 'L_71_empty_01', 'L_72_empty_01',
    'L_73_empty_01', 'L_74_empty_01', 'L_74_empty_02', 'L_74_empty_03', 'L_75_empty_01', 'L_76_empty_01', 'L_77_empty_01', 'L_77_empty_02', 'L_79_empty_02', 'L_79_empty_03', 'L_80_empty_01',
    'L_81_empty_01', 'L_81_empty_02', 'L_82_empty_01', 'L_83_empty_01', 'L_83_empty_02', 'L_83_empty_03', 'L_84_empty_01', 'L_85_empty_01', 'L_86_empty_01', 'L_87_empty_01', 'L_88_empty_01',
    'L_89_empty_01', 'L_90_empty_01', 'L_91_empty_01', 'L_92_empty_01', 'L_94_empty_01', 'L_95_empty_01', 'L_96_empty_01', 'L_96_empty_02', 'L_96_empty_03', 'L_96_empty_04', 'L_97_empty_01',
    'L_98_empty_01', 'L_99_empty_01', 'L_100_empty_01', 'L_101_empty_01', 'L_102_empty_01', 'L_103_empty_01', 'L_104_empty_01', 'L_105_empty_01', 'L_106_empty_01', 'L_107_empty_01',
    'L_108_empty_01', 'L_109_empty_01', 'L_110_empty_01', 'L_111_empty_01', 'L_112_empty_01', 'L_113_empty_01', 'L_114_empty_01', 'L_115_empty_01', 'L_116_empty_01', 'L_117_empty_01',
    'L_118_empty_02', 'L_120_empty_01', 'L_121_empty_01', 'L_122_empty_01', 'L_123_empty_01', 'L_124_empty_01', 'L_125_empty_01', 'L_126_empty_01', 'L_127_empty_01'],
}