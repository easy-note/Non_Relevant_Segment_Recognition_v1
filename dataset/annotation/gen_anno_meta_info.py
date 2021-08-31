"""
Parsing Annotation info and aggregate for patient level
"""
### for setting import mobule ###
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
print(sys.path)

from OOB_RECOG.test.test import idx_to_time
from OOB_RECOG.test.test_info_dict import parsing_patient_video, pateint_video_sort, return_idx_is_str_in_list, check_anno_over_frame, load_yaml_to_dict
from OOB_RECOG.evaluation.visual_gradcam import img_seq_to_gif
### for setting import mobule ###

import cv2
import torch
import numpy as np
import pandas as pd
import glob
import matplotlib
import time
import json

import math

from pandas import DataFrame as df

import torch
import pytorch_lightning as pl

import re
import copy

import natsort
import yaml

matplotlib.use('Agg')

EXCEPTION_NUM = -100

###### PATINET CASE ########

LAPA_CASE = ['L_301', 'L_303', 'L_305', 'L_309', 'L_317', 'L_325', 'L_326', 'L_340', 'L_346', 'L_349', 'L_412', 'L_421', 'L_423', 'L_442', 'L_443',
                'L_450', 'L_458', 'L_465', 'L_491', 'L_493', 'L_496', 'L_507', 'L_522', 'L_534', 'L_535', 'L_550', 'L_553', 'L_586', 'L_595', 'L_605', 'L_607', 'L_625',
                'L_631', 'L_647', 'L_654', 'L_659', 'L_660', 'L_661', 'L_669', 'L_676', 'L_310', 'L_311', 'L_330', 'L_333', 'L_367', 'L_370', 'L_377', 'L_379', 'L_385',
                'L_387', 'L_389', 'L_391', 'L_393', 'L_400', 'L_402', 'L_406', 'L_408', 'L_413', 'L_414', 'L_415', 'L_418', 'L_419', 'L_427', 'L_428', 'L_430', 'L_433',
                'L_434', 'L_436', 'L_439', 'L_471', 'L_473', 'L_475', 'L_477', 'L_478', 'L_479', 'L_481', 'L_482', 'L_484', 'L_513', 'L_514', 'L_515', 'L_517', 'L_537',
                'L_539', 'L_542', 'L_543', 'L_545', 'L_546', 'L_556', 'L_558', 'L_560', 'L_563', 'L_565', 'L_568', 'L_569', 'L_572', 'L_574', 'L_575', 'L_577', 'L_580']

ROBOT_CASE = ['R_1', 'R_2', 'R_3', 'R_4', 'R_5', 'R_6', 'R_7', 'R_10', 'R_13', 'R_14', 'R_15', 'R_17', 'R_18', 'R_19', 'R_22', 'R_48', 'R_56', 'R_74',
                'R_76', 'R_84', 'R_94', 'R_100', 'R_116', 'R_117', 'R_201', 'R_202', 'R_203', 'R_204', 'R_205', 'R_206', 'R_207', 'R_208', 'R_209', 'R_210', 'R_301',
                'R_302', 'R_303', 'R_304', 'R_305', 'R_310', 'R_311', 'R_312', 'R_313', 'R_320', 'R_321', 'R_324', 'R_329', 'R_334', 'R_336', 'R_338', 'R_339', 'R_340',
                'R_342', 'R_345', 'R_346', 'R_347', 'R_348', 'R_349', 'R_355', 'R_357', 'R_358', 'R_362', 'R_363', 'R_369', 'R_372', 'R_376', 'R_378', 'R_379', 'R_386',
                'R_391', 'R_393', 'R_399', 'R_400', 'R_402', 'R_403', 'R_405', 'R_406', 'R_409', 'R_412', 'R_413', 'R_415', 'R_418', 'R_419', 'R_420', 'R_423', 'R_424',
                'R_427', 'R_436', 'R_445', 'R_449', 'R_455', 'R_480', 'R_493', 'R_501', 'R_510', 'R_522', 'R_523', 'R_526', 'R_532', 'R_533']


###### OOB VIDEO LIST ########
# ANNOTATION PATH - /NAS/DATA2/Public/IDC_21.06.25/ANNOTATION/Gastrectomy/Event/OOB/V1 # /data2/Public/IDC_21.06.25/ANNOTATION/Gastrectomy/Event/OOB/V1
# ANNOTATION PATH - /NAS/DATA2/Public/IDC_21.06.25/ANNOTATION/Gastrectomy/Event/OOB/V2 # /data2/Public/IDC_21.06.25/ANNOTATION/Gastrectomy/Event/OOB/V2

# 91ea, 40case # /NAS/DATA/HuToM/Video_Robot_cordname # /data1/HuToM/Video_Robot_cordname
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

# 134ea, 60case # /NAS/DATA2/Video/Robot/Dataset2_60case # /data2/Video/Robot/Dataset2_60case
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

# 225ea & 100case 
OOB_robot_list = OOB_robot_40 + OOB_robot_60

# 350ea, 40case # /NAS/DATA2/Public/IDC_21.06.25/Dataset1 # /data2/Public/IDC_21.06.25/Dataset1
OOB_lapa_40 = [
    'L_301_xx0_01', 'L_301_xx0_02', 'L_301_xx0_03', 'L_301_xx0_04', 'L_301_xx0_05', 'L_301_xx0_06', 'L_303_xx0_01', 'L_303_xx0_02', 'L_303_xx0_03', 'L_303_xx0_04', 
    'L_303_xx0_05', 'L_303_xx0_06', 'L_305_xx0_01', 'L_305_xx0_02', 'L_305_xx0_03', 'L_305_xx0_04', 'L_305_xx0_05', 'L_305_xx0_06', 'L_305_xx0_07', 'L_305_xx0_08', 
    'L_305_xx0_09', 'L_305_xx0_10', 'L_305_xx0_11', 'L_305_xx0_12', 'L_305_xx0_13', 'L_305_xx0_14', 'L_305_xx0_15', 'L_309_xx0_01', 'L_309_xx0_02', 'L_309_xx0_03', 
    'L_309_xx0_04', 'L_309_xx0_05', 'L_309_xx0_06', 'L_309_xx0_07', 'L_317_xx0_01', 'L_317_xx0_02', 'L_317_xx0_03', 'L_317_xx0_04', 'L_325_xx0_01', 'L_325_xx0_02', 
    'L_325_xx0_03', 'L_325_xx0_04', 'L_325_xx0_05', 'L_325_xx0_06', 'L_325_xx0_07', 'L_325_xx0_08', 'L_325_xx0_09', 'L_325_xx0_10', 'L_325_xx0_11', 'L_325_xx0_12', 
    'L_326_xx0_01', 'L_326_xx0_02', 'L_326_xx0_03', 'L_326_xx0_04', 'L_326_xx0_05', 'L_326_xx0_06', 'L_340_xx0_01', 'L_340_xx0_02', 'L_340_xx0_03', 'L_340_xx0_04', 
    'L_340_xx0_05', 'L_340_xx0_06', 'L_340_xx0_07', 'L_340_xx0_08', 'L_340_xx0_09', 'L_340_xx0_10', 'L_346_xx0_01', 'L_346_xx0_02', 'L_349_ch1_01', 'L_349_ch1_02', 
    'L_349_ch1_03', 'L_349_ch1_04', 'L_412_xx0_01', 'L_412_xx0_02', 'L_412_xx0_03', 'L_421_xx0_01', 'L_421_xx0_02', 'L_423_xx0_01', 'L_423_xx0_02', 'L_423_xx0_03', 
    'L_423_xx0_04', 'L_423_xx0_05', 'L_442_xx0_01', 'L_442_xx0_02', 'L_442_xx0_03', 'L_442_xx0_04', 'L_442_xx0_05', 'L_442_xx0_06', 'L_442_xx0_07', 'L_442_xx0_08', 
    'L_442_xx0_09', 'L_442_xx0_10', 'L_442_xx0_11', 'L_442_xx0_12', 'L_442_xx0_13', 'L_442_xx0_14', 'L_443_xx0_01', 'L_443_xx0_02', 'L_443_xx0_03', 'L_443_xx0_04', 
    'L_443_xx0_05', 'L_443_xx0_06', 'L_443_xx0_07', 'L_443_xx0_08', 'L_443_xx0_09', 'L_443_xx0_10', 'L_443_xx0_11', 'L_443_xx0_12', 'L_443_xx0_13', 'L_443_xx0_14', 
    'L_443_xx0_15', 'L_443_xx0_16', 'L_450_xx0_01', 'L_450_xx0_02', 'L_450_xx0_03', 'L_450_xx0_04', 'L_450_xx0_05', 'L_450_xx0_06', 'L_450_xx0_07', 'L_450_xx0_08', 
    'L_450_xx0_09', 'L_450_xx0_10', 'L_450_xx0_11', 'L_450_xx0_12', 'L_450_xx0_13', 'L_450_xx0_14', 'L_450_xx0_15', 'L_450_xx0_16', 'L_450_xx0_17', 'L_450_xx0_18', 
    'L_450_xx0_19', 'L_450_xx0_20', 'L_450_xx0_21', 'L_450_xx0_22', 'L_458_xx0_01', 'L_458_xx0_02', 'L_458_xx0_03', 'L_458_xx0_04', 'L_458_xx0_05', 'L_458_xx0_06', 
    'L_458_xx0_07', 'L_458_xx0_08', 'L_458_xx0_09', 'L_458_xx0_10', 'L_458_xx0_11', 'L_458_xx0_12', 'L_458_xx0_13', 'L_458_xx0_14', 'L_458_xx0_15', 'L_465_xx0_01', 
    'L_465_xx0_02', 'L_465_xx0_03', 'L_465_xx0_04', 'L_465_xx0_05', 'L_465_xx0_06', 'L_465_xx0_07', 'L_465_xx0_08', 'L_465_xx0_09', 'L_465_xx0_10', 'L_465_xx0_11', 
    'L_465_xx0_12', 'L_465_xx0_13', 'L_465_xx0_14', 'L_465_xx0_15', 'L_465_xx0_16', 'L_465_xx0_17', 'L_465_xx0_18', 'L_465_xx0_19', 'L_465_xx0_20', 'L_465_xx0_21', 
    'L_491_xx0_01', 'L_491_xx0_02', 'L_491_xx0_03', 'L_491_xx0_04', 'L_491_xx0_05', 'L_491_xx0_06', 'L_491_xx0_07', 'L_491_xx0_08', 'L_491_xx0_09', 'L_491_xx0_10', 
    'L_491_xx0_11', 'L_491_xx0_12', 'L_493_ch1_01', 'L_493_ch1_02', 'L_493_ch1_03', 'L_493_ch1_04', 'L_496_ch1_01', 'L_496_ch1_02', 'L_496_ch1_03', 'L_507_xx0_01', 
    'L_507_xx0_02', 'L_507_xx0_03', 'L_507_xx0_04', 'L_507_xx0_05', 'L_507_xx0_06', 'L_507_xx0_07', 'L_522_xx0_01', 'L_522_xx0_02', 'L_522_xx0_03', 'L_522_xx0_04', 
    'L_522_xx0_05', 'L_522_xx0_06', 'L_522_xx0_07', 'L_522_xx0_08', 'L_522_xx0_09', 'L_522_xx0_10', 'L_522_xx0_11', 'L_534_xx0_01', 'L_534_xx0_02', 'L_534_xx0_03', 
    'L_534_xx0_04', 'L_534_xx0_05', 'L_534_xx0_06', 'L_534_xx0_07', 'L_535_xx0_01', 'L_535_xx0_02', 'L_535_xx0_03', 'L_535_xx0_04', 'L_535_xx0_05', 'L_550_xx0_01', 
    'L_550_xx0_02', 'L_550_xx0_03', 'L_550_xx0_04', 'L_550_xx0_05', 'L_550_xx0_06', 'L_550_xx0_07', 'L_550_xx0_08', 'L_550_xx0_09', 'L_550_xx0_10', 'L_550_xx0_11', 
    'L_550_xx0_12', 'L_553_ch1_01', 'L_553_ch1_02', 'L_553_ch1_03', 'L_553_ch1_04', 'L_586_xx0_01', 'L_586_xx0_02', 'L_586_xx0_03', 'L_586_xx0_04', 'L_586_xx0_05', 
    'L_586_xx0_06', 'L_586_xx0_07', 'L_586_xx0_08', 'L_586_xx0_09', 'L_586_xx0_10', 'L_586_xx0_11', 'L_586_xx0_12', 'L_586_xx0_13', 'L_586_xx0_14', 'L_586_xx0_15', 
    'L_586_xx0_16', 'L_586_xx0_17', 'L_586_xx0_18', 'L_586_xx0_19', 'L_586_xx0_20', 'L_595_xx0_01', 'L_595_xx0_02', 'L_595_xx0_03', 'L_595_xx0_04', 'L_595_xx0_05', 
    'L_595_xx0_06', 'L_595_xx0_07', 'L_595_xx0_08', 'L_605_xx0_01', 'L_605_xx0_02', 'L_605_xx0_03', 'L_605_xx0_04', 'L_605_xx0_05', 'L_605_xx0_06', 'L_605_xx0_07', 
    'L_605_xx0_08', 'L_605_xx0_09', 'L_605_xx0_10', 'L_605_xx0_11', 'L_605_xx0_12', 'L_605_xx0_13', 'L_605_xx0_14', 'L_605_xx0_15', 'L_605_xx0_16', 'L_605_xx0_17', 
    'L_605_xx0_18', 'L_607_xx0_01', 'L_607_xx0_02', 'L_607_xx0_03', 'L_607_xx0_04', 'L_625_xx0_01', 'L_625_xx0_02', 'L_625_xx0_03', 'L_625_xx0_04', 'L_625_xx0_05', 
    'L_625_xx0_06', 'L_625_xx0_07', 'L_625_xx0_08', 'L_625_xx0_09', 'L_631_xx0_01', 'L_631_xx0_02', 'L_631_xx0_03', 'L_631_xx0_04', 'L_631_xx0_05', 'L_631_xx0_06', 
    'L_631_xx0_07', 'L_631_xx0_08', 'L_647_xx0_01', 'L_647_xx0_02', 'L_647_xx0_03', 'L_647_xx0_04', 'L_654_xx0_01', 'L_654_xx0_02', 'L_654_xx0_03', 'L_654_xx0_04', 
    'L_654_xx0_05', 'L_654_xx0_06', 'L_654_xx0_07', 'L_654_xx0_08', 'L_654_xx0_09', 'L_654_xx0_10', 'L_654_xx0_11', 'L_659_xx0_01', 'L_659_xx0_02', 'L_659_xx0_03', 
    'L_659_xx0_04', 'L_659_xx0_05', 'L_660_xx0_01', 'L_660_xx0_02', 'L_660_xx0_03', 'L_660_xx0_04', 'L_661_xx0_01', 'L_661_xx0_02', 'L_661_xx0_03', 'L_661_xx0_04', 
    'L_661_xx0_05', 'L_661_xx0_06', 'L_661_xx0_07', 'L_661_xx0_08', 'L_661_xx0_09', 'L_661_xx0_10', 'L_661_xx0_11', 'L_661_xx0_12', 'L_661_xx0_13', 'L_661_xx0_14', 
    'L_661_xx0_15', 'L_669_xx0_01', 'L_669_xx0_02', 'L_669_xx0_03', 'L_669_xx0_04', 'L_676_xx0_01', 'L_676_xx0_02', 'L_676_xx0_03', 'L_676_xx0_04', 'L_676_xx0_05']

# 521ea, 60case # /NAS/DATA2/Public/IDC_21.06.25/Dataset2 # /data2/Public/IDC_21.06.25/Dataset2
OOB_lapa_60 = [
    'L_310_xx0_01', 'L_310_xx0_02', 'L_310_xx0_03', 'L_310_xx0_04', 'L_310_xx0_05', 'L_310_xx0_06', 'L_310_xx0_07', 'L_310_xx0_08', 'L_310_xx0_09', 'L_310_xx0_10', 
    'L_310_xx0_11', 'L_310_xx0_12', 'L_311_xx0_01', 'L_311_xx0_02', 'L_311_xx0_03', 'L_311_xx0_04', 'L_311_xx0_05', 'L_330_ch1_01', 'L_333_xx0_01', 'L_333_xx0_02', 
    'L_333_xx0_03', 'L_333_xx0_04', 'L_333_xx0_05', 'L_333_xx0_06', 'L_333_xx0_07', 'L_333_xx0_08', 'L_333_xx0_09', 'L_333_xx0_10', 'L_333_xx0_11', 'L_367_ch1_01', 
    'L_370_ch1_01', 'L_377_ch1_01', 'L_379_xx0_01', 'L_379_xx0_02', 'L_379_xx0_03', 'L_379_xx0_04', 'L_379_xx0_05', 'L_379_xx0_06', 'L_379_xx0_07', 'L_379_xx0_08', 
    'L_379_xx0_09', 'L_379_xx0_10', 'L_379_xx0_11', 'L_385_xx0_01', 'L_385_xx0_02', 'L_385_xx0_03', 'L_385_xx0_04', 'L_385_xx0_05', 'L_385_xx0_06', 'L_385_xx0_07', 
    'L_385_xx0_08', 'L_385_xx0_09', 'L_385_xx0_10', 'L_385_xx0_11', 'L_385_xx0_12', 'L_385_xx0_13', 'L_385_xx0_14', 'L_385_xx0_15', 'L_387_xx0_01', 'L_387_xx0_02', 
    'L_387_xx0_03', 'L_387_xx0_04', 'L_387_xx0_05', 'L_387_xx0_06', 'L_387_xx0_07', 'L_387_xx0_08', 'L_389_xx0_01', 'L_389_xx0_02', 'L_389_xx0_03', 'L_389_xx0_04', 
    'L_389_xx0_05', 'L_389_xx0_06', 'L_389_xx0_07', 'L_389_xx0_08', 'L_389_xx0_09', 'L_389_xx0_10', 'L_389_xx0_11', 'L_389_xx0_12', 'L_389_xx0_13', 'L_391_xx0_01', 
    'L_391_xx0_02', 'L_391_xx0_03', 'L_391_xx0_04', 'L_391_xx0_05', 'L_391_xx0_06', 'L_391_xx0_07', 'L_391_xx0_08', 'L_391_xx0_09', 'L_393_xx0_01', 'L_393_xx0_02', 
    'L_393_xx0_03', 'L_393_xx0_04', 'L_393_xx0_05', 'L_393_xx0_06', 'L_393_xx0_07', 'L_393_xx0_08', 'L_393_xx0_09', 'L_393_xx0_10', 'L_400_xx0_01', 'L_400_xx0_02', 
    'L_400_xx0_03', 'L_400_xx0_04', 'L_400_xx0_05', 'L_400_xx0_06', 'L_400_xx0_07', 'L_400_xx0_08', 'L_400_xx0_09', 'L_400_xx0_10', 'L_400_xx0_11', 'L_400_xx0_12', 
    'L_402_xx0_01', 'L_402_xx0_02', 'L_402_xx0_03', 'L_402_xx0_04', 'L_406_xx0_01', 'L_406_xx0_02', 'L_406_xx0_03', 'L_406_xx0_04', 'L_406_xx0_05', 'L_406_xx0_06', 
    'L_406_xx0_07', 'L_406_xx0_08', 'L_406_xx0_09', 'L_406_xx0_10', 'L_406_xx0_11', 'L_406_xx0_12', 'L_406_xx0_13', 'L_408_ch1_01', 'L_413_xx0_01', 'L_413_xx0_02', 
    'L_413_xx0_03', 'L_413_xx0_04', 'L_413_xx0_05', 'L_413_xx0_06', 'L_413_xx0_07', 'L_413_xx0_08', 'L_413_xx0_09', 'L_413_xx0_10', 'L_414_xx0_01', 'L_414_xx0_02', 
    'L_414_xx0_03', 'L_414_xx0_04', 'L_414_xx0_05', 'L_414_xx0_06', 'L_414_xx0_07', 'L_414_xx0_08', 'L_415_xx0_01', 'L_415_xx0_02', 'L_415_xx0_03', 'L_415_xx0_04', 
    'L_415_xx0_05', 'L_415_xx0_06', 'L_415_xx0_07', 'L_415_xx0_08', 'L_415_xx0_09', 'L_415_xx0_10', 'L_415_xx0_11', 'L_415_xx0_12', 'L_418_xx0_01', 'L_418_xx0_02', 
    'L_418_xx0_03', 'L_418_xx0_04', 'L_418_xx0_05', 'L_418_xx0_06', 'L_418_xx0_07', 'L_418_xx0_08', 'L_419_xx0_01', 'L_419_xx0_02', 'L_419_xx0_03', 'L_419_xx0_04', 
    'L_419_xx0_05', 'L_419_xx0_06', 'L_427_xx0_01', 'L_427_xx0_02', 'L_427_xx0_03', 'L_427_xx0_04', 'L_427_xx0_05', 'L_427_xx0_06', 'L_427_xx0_07', 'L_427_xx0_08', 
    'L_427_xx0_09', 'L_427_xx0_10', 'L_427_xx0_11', 'L_427_xx0_12', 'L_427_xx0_13', 'L_427_xx0_14', 'L_427_xx0_15', 'L_428_ch1_01', 'L_430_ch1_01', 'L_433_xx0_01', 
    'L_433_xx0_02', 'L_433_xx0_03', 'L_433_xx0_04', 'L_433_xx0_05', 'L_433_xx0_06', 'L_433_xx0_07', 'L_433_xx0_08', 'L_433_xx0_09', 'L_434_xx0_01', 'L_434_xx0_02', 
    'L_434_xx0_03', 'L_434_xx0_04', 'L_434_xx0_05', 'L_434_xx0_06', 'L_434_xx0_07', 'L_434_xx0_08', 'L_434_xx0_09', 'L_434_xx0_10', 'L_436_xx0_01', 'L_436_xx0_02', 
    'L_436_xx0_03', 'L_436_xx0_04', 'L_436_xx0_05', 'L_436_xx0_06', 'L_436_xx0_07', 'L_436_xx0_08', 'L_436_xx0_09', 'L_436_xx0_10', 'L_436_xx0_11', 'L_436_xx0_12', 
    'L_439_xx0_01', 'L_439_xx0_02', 'L_439_xx0_03', 'L_439_xx0_04', 'L_439_xx0_05', 'L_439_xx0_06', 'L_439_xx0_07', 'L_439_xx0_08', 'L_439_xx0_09', 'L_439_xx0_10', 
    'L_439_xx0_11', 'L_439_xx0_12', 'L_439_xx0_13', 'L_439_xx0_14', 'L_439_xx0_15', 'L_439_xx0_16', 'L_471_xx0_01', 'L_471_xx0_02', 'L_471_xx0_03', 'L_471_xx0_04', 
    'L_471_xx0_05', 'L_471_xx0_06', 'L_471_xx0_07', 'L_471_xx0_08', 'L_471_xx0_09', 'L_471_xx0_10', 'L_471_xx0_11', 'L_473_xx0_01', 'L_473_xx0_02', 'L_473_xx0_03', 
    'L_473_xx0_04', 'L_473_xx0_05', 'L_473_xx0_06', 'L_473_xx0_07', 'L_475_ch1_01', 'L_475_ch1_02', 'L_477_ch1_01', 'L_478_xx0_01', 'L_478_xx0_02', 'L_478_xx0_03', 
    'L_478_xx0_04', 'L_478_xx0_05', 'L_478_xx0_06', 'L_478_xx0_07', 'L_478_xx0_08', 'L_478_xx0_09', 'L_478_xx0_10', 'L_479_xx0_01', 'L_479_xx0_02', 'L_479_xx0_03', 
    'L_479_xx0_04', 'L_479_xx0_05', 'L_479_xx0_06', 'L_479_xx0_07', 'L_479_xx0_08', 'L_479_xx0_09', 'L_481_xx0_01', 'L_481_xx0_02', 'L_481_xx0_03', 'L_481_xx0_04', 
    'L_481_xx0_05', 'L_481_xx0_06', 'L_481_xx0_07', 'L_481_xx0_08', 'L_481_xx0_09', 'L_481_xx0_10', 'L_481_xx0_11', 'L_481_xx0_12', 'L_481_xx0_13', 'L_482_xx0_01', 
    'L_482_xx0_02', 'L_482_xx0_03', 'L_482_xx0_04', 'L_482_xx0_05', 'L_482_xx0_06', 'L_482_xx0_07', 'L_482_xx0_08', 'L_482_xx0_09', 'L_482_xx0_10', 'L_482_xx0_11', 
    'L_482_xx0_12', 'L_482_xx0_13', 'L_482_xx0_14', 'L_482_xx0_15', 'L_484_xx0_01', 'L_484_xx0_02', 'L_484_xx0_03', 'L_484_xx0_04', 'L_484_xx0_05', 'L_484_xx0_06', 
    'L_484_xx0_07', 'L_484_xx0_08', 'L_484_xx0_09', 'L_484_xx0_10', 'L_484_xx0_11', 'L_513_xx0_01', 'L_513_xx0_02', 'L_513_xx0_03', 'L_513_xx0_04', 'L_513_xx0_05', 
    'L_513_xx0_06', 'L_513_xx0_07', 'L_513_xx0_08', 'L_513_xx0_09', 'L_513_xx0_10', 'L_513_xx0_11', 'L_513_xx0_12', 'L_514_xx0_01', 'L_514_xx0_02', 'L_514_xx0_03', 
    'L_514_xx0_04', 'L_514_xx0_05', 'L_514_xx0_06', 'L_514_xx0_07', 'L_514_xx0_08', 'L_514_xx0_09', 'L_514_xx0_10', 'L_515_xx0_01', 'L_515_xx0_02', 'L_515_xx0_03', 
    'L_515_xx0_04', 'L_515_xx0_05', 'L_515_xx0_06', 'L_515_xx0_07', 'L_515_xx0_08', 'L_517_xx0_01', 'L_517_xx0_02', 'L_517_xx0_03', 'L_517_xx0_04', 'L_517_xx0_05', 
    'L_517_xx0_06', 'L_517_xx0_07', 'L_517_xx0_08', 'L_537_xx0_01', 'L_537_xx0_02', 'L_537_xx0_03', 'L_537_xx0_04', 'L_537_xx0_05', 'L_537_xx0_06', 'L_537_xx0_07', 
    'L_537_xx0_08', 'L_537_xx0_09', 'L_537_xx0_10', 'L_537_xx0_11', 'L_537_xx0_12', 'L_539_ch1_01', 'L_542_xx0_01', 'L_542_xx0_02', 'L_542_xx0_03', 'L_542_xx0_04', 
    'L_542_xx0_05', 'L_542_xx0_06', 'L_542_xx0_07', 'L_542_xx0_08', 'L_542_xx0_09', 'L_543_xx0_01', 'L_543_xx0_02', 'L_543_xx0_03', 'L_543_xx0_04', 'L_543_xx0_05', 
    'L_543_xx0_06', 'L_543_xx0_07', 'L_543_xx0_08', 'L_543_xx0_09', 'L_543_xx0_10', 'L_543_xx0_11', 'L_543_xx0_12', 'L_543_xx0_13', 'L_543_xx0_14', 'L_545_xx0_01', 
    'L_545_xx0_02', 'L_545_xx0_03', 'L_545_xx0_04', 'L_545_xx0_05', 'L_545_xx0_06', 'L_545_xx0_07', 'L_545_xx0_08', 'L_545_xx0_09', 'L_546_xx0_01', 'L_546_xx0_02', 
    'L_546_xx0_03', 'L_546_xx0_04', 'L_546_xx0_05', 'L_546_xx0_06', 'L_546_xx0_07', 'L_546_xx0_08', 'L_546_xx0_09', 'L_546_xx0_10', 'L_556_xx0_01', 'L_556_xx0_02', 
    'L_556_xx0_03', 'L_556_xx0_04', 'L_556_xx0_05', 'L_556_xx0_06', 'L_556_xx0_07', 'L_556_xx0_08', 'L_556_xx0_09', 'L_556_xx0_10', 'L_556_xx0_11', 'L_556_xx0_12', 
    'L_556_xx0_13', 'L_556_xx0_14', 'L_558_xx0_01', 'L_560_xx0_01', 'L_560_xx0_02', 'L_560_xx0_03', 'L_560_xx0_04', 'L_560_xx0_05', 'L_560_xx0_06', 'L_560_xx0_07', 
    'L_560_xx0_08', 'L_560_xx0_09', 'L_560_xx0_10', 'L_560_xx0_11', 'L_560_xx0_12', 'L_560_xx0_13', 'L_560_xx0_14', 'L_560_xx0_15', 'L_563_xx0_01', 'L_563_xx0_02', 
    'L_563_xx0_03', 'L_563_xx0_04', 'L_563_xx0_05', 'L_563_xx0_06', 'L_563_xx0_07', 'L_563_xx0_08', 'L_563_xx0_09', 'L_563_xx0_10', 'L_563_xx0_11', 'L_563_xx0_12', 
    'L_565_xx0_01', 'L_565_xx0_02', 'L_565_xx0_03', 'L_565_xx0_04', 'L_565_xx0_05', 'L_565_xx0_06', 'L_565_xx0_07', 'L_565_xx0_08', 'L_565_xx0_09', 'L_568_xx0_01', 
    'L_568_xx0_02', 'L_568_xx0_03', 'L_568_xx0_04', 'L_568_xx0_05', 'L_568_xx0_06', 'L_568_xx0_07', 'L_568_xx0_08', 'L_568_xx0_09', 'L_569_xx0_01', 'L_569_xx0_02', 
    'L_569_xx0_03', 'L_569_xx0_04', 'L_569_xx0_05', 'L_569_xx0_06', 'L_569_xx0_07', 'L_569_xx0_08', 'L_569_xx0_09', 'L_569_xx0_10', 'L_569_xx0_11', 'L_569_xx0_12', 
    'L_572_ch1_01', 'L_574_xx0_01', 'L_574_xx0_02', 'L_574_xx0_03', 'L_574_xx0_04', 'L_574_xx0_05', 'L_574_xx0_06', 'L_574_xx0_07', 'L_574_xx0_08', 'L_574_xx0_09', 
    'L_574_xx0_10', 'L_574_xx0_11', 'L_575_xx0_01', 'L_575_xx0_02', 'L_575_xx0_03', 'L_575_xx0_04', 'L_577_xx0_01', 'L_577_xx0_02', 'L_577_xx0_03', 'L_577_xx0_04', 
    'L_577_xx0_05', 'L_577_xx0_06', 'L_577_xx0_07', 'L_577_xx0_08', 'L_577_xx0_09', 'L_577_xx0_10', 'L_580_xx0_01', 'L_580_xx0_02', 'L_580_xx0_03', 'L_580_xx0_04', 
    'L_580_xx0_05', 'L_580_xx0_06', 'L_580_xx0_07', 'L_580_xx0_08', 'L_580_xx0_09', 'L_580_xx0_10', 'L_580_xx0_11', 'L_580_xx0_12', 'L_580_xx0_13', 'L_580_xx0_14', 'L_580_xx0_15']

# 871ea, 100case
OOB_lapa_list = OOB_lapa_40 + OOB_lapa_60







##### ##### FOR OOB EVENT META INFO ##### #####
def parser_anno_meta_info(anno_path) :
    assert os.path.splitext(os.path.basename(anno_path))[-1] in ['.json'], 'NOT SUPPORT ANNOTATION FORMAT'

    with open(anno_path) as json_file :
            json_data = json.load(json_file)

    anno_meta_info_dict = {
        'totalFrame': None,
        'fps': None,
        'IB_count': None,
        'OOB_count': None,
        'OOB_event_cnt': None,
        'OOB_event_duration': None,
        'annotation_start_point': None,
        'annotation_end_point': None
    }

    # meta_info
    total_frame = json_data['totalFrame']
    fps = json_data['frameRate']

    oob_event_cnt = 0
    oob_event_duration = []
    ib_count = 0
    oob_count = 0
    annotation_start_point = EXCEPTION_NUM
    annotation_end_point = EXCEPTION_NUM
    start_frame_idx = EXCEPTION_NUM
    end_frame_idx = EXCEPTION_NUM
    
    anno_info = []

    # annotation frame    
    for anno_data in json_data['annotations'] :
        start_frame_idx = anno_data['start'] # frame
        end_frame_idx = anno_data['end'] # frame

        anno_info.append([start_frame_idx, end_frame_idx])
    
    # sanity check (over frame)
    if anno_info : # not empty annotation
        _, anno_info = check_anno_over_frame(anno_info, total_frame)
    
    # check count and  duration
    for start_frame_idx, end_frame_idx in anno_info :
        oob_event_cnt += 1

        oob_duration = (end_frame_idx - start_frame_idx) + 1
        oob_count += oob_duration
        oob_event_duration.append(frame_to_sec(oob_duration, fps)) # 1-5 == 1,2,3,4,5 frame, duration = 5 frame

        if oob_event_cnt == 1: # init annotation
            annotation_start_point = start_frame_idx
        
        if oob_event_cnt == len(anno_info): # last annotation
            annotation_end_point = end_frame_idx

    ib_count = total_frame - oob_count
    
    # set parsing info
    anno_meta_info_dict['totalFrame'] = total_frame
    anno_meta_info_dict['fps'] = fps
    anno_meta_info_dict['IB_count'] = ib_count
    anno_meta_info_dict['OOB_count'] = oob_count
    anno_meta_info_dict['OOB_event_cnt'] = oob_event_cnt
    anno_meta_info_dict['OOB_event_duration'] = oob_event_duration # list
    anno_meta_info_dict['annotation_start_point'] = annotation_start_point
    anno_meta_info_dict['annotation_end_point'] = annotation_end_point
    anno_meta_info_dict['anno_info'] = anno_info # list
    
    return anno_meta_info_dict
    


def frame_to_sec(frame, fps):
    sec = frame/fps
    return sec




def gen_anno_meta_info(anno_root_path, video_list, patient_list, results_dir) :
    print('\n{}'.format('=====' * 10))
    print('\t ===== GET ANNOTATION INFO =====')
    print('{}\n'.format('=====' * 10))

    # init df
    total_anno_meta_info_per_patient_df = pd.DataFrame(index=range(0, 0), columns=['Patient', 'Method', 'totalFrame', 'fps', 'IB_count', 'OOB_count', 'total_time', 'IB_event_time', 'OOB_event_time', 'OOB_event_cnt', 'annotation_start_point', 'annotation_end_point', 'start_frame_idx', 'end_frame_idx', 'OOB_event_duration'])
    total_anno_meta_info_df = pd.DataFrame(index=range(0, 0), columns=['Video_name', 'Method', 'totalFrame', 'fps', 'IB_count', 'OOB_count', 'total_time', 'IB_event_time', 'OOB_event_time', 'OOB_event_cnt', 'annotation_start_point', 'annotation_end_point', 'start_frame_idx', 'end_frame_idx', 'OOB_event_duration'])

    # set USE VIDEO LIST
    USE_VIDEO_LIST = video_list.copy()
    
    # 1. SET patient video
    patient_video_dict = parsing_patient_video(patient_list, video_list) # parsing pateint video

    # 2. patient video sorting
    for patient, video_name_list in patient_video_dict.items() : 
        patient_video_dict[patient] = pateint_video_sort(video_name_list)
    
    # print patinet video
    print('\n----- SORTED ------\n')
    for patient, video_name_list in patient_video_dict.items() : 
        print(patient, video_name_list) 

    
    # 3. parsing all annotation path
    all_anno_path = []
    anno_ext_list = ['json']

    for ext in anno_ext_list :
        all_anno_path.extend(glob.glob(os.path.join(anno_root_path, '*.{}'.format(ext))))
    
    # 4. check which idx is included parser_str in all_anno_path
    cnt = 0 # PROCESSED CNT
    FAILD_VIDEO_NAME = []
    NO_ANNOTATION_VIDEO = []

    # per patient loop
    for patient, video_name_list in patient_video_dict.items() : 

        patient_total_frame = 0
        patient_fps = 0
        patient_oob_event_cnt = 0
        patient_oob_event_duration = []
        patient_ib_count = 0
        patient_oob_count = 0
        patient_annotation_start_point = EXCEPTION_NUM
        patient_annotation_end_point = EXCEPTION_NUM
        patient_start_frame_idx = EXCEPTION_NUM
        patient_end_frame_idx = EXCEPTION_NUM
        patient_anno_info = []


        # for patient aggregation info 
        patient_total_frame_list = []
        patient_annotation_info_list = []
        patient_fps_list = []

        # only loop for patient video
        for video_name in video_name_list : # per video loop
            idx = return_idx_is_str_in_list(video_name, all_anno_path)

            if idx == -1 : # file is not exist
                FAILD_VIDEO_NAME.append(video_name)
                pass

            if video_name in ['R_14_ch1_05', 'R_17_ch1_06']: # exception video (file is not exist)
                # per video info (each row)
                if video_name == 'R_14_ch1_05':
                    anno_meta_info_per_each_video_dict = {
                        'Video_name':'R_14_ch1_05',
                        'Method': 'R',
                        'totalFrame':4060,
                        'fps':30,
                        'IB_count':0,
                        'OOB_count':4060,
                        
                        'total_time': 135.3333333,
                        'IB_event_time': 0,
                        'OOB_event_time': 135.3333333,
                        
                        'OOB_event_cnt': 1,

                        'annotation_start_point': 0,
                        'annotation_end_point': 4059,
                        'OOB_event_duration': 135.3333333,
                        'start_frame_idx': 0,
                        'end_frame_idx': 4059,
                        'start_frame_time':0,
                        'end_frame_time':135.3
                    }

                    # add for patinets
                    anno_meta_info_dict = {
                        'totalFrame': 4060,
                        'fps': 30,
                        'anno_info': [[0, 4059]]
                    }

                elif video_name == 'R_17_ch1_06':
                    anno_meta_info_per_each_video_dict = {
                        'Video_name':'R_17_ch1_06',
                        'Method': 'R',
                        'totalFrame':1476,
                        'fps':30,
                        'IB_count':0,
                        'OOB_count':1476,
                        
                        'total_time': 49.2,
                        'IB_event_time': 0,
                        'OOB_event_time': 49.2,
                        
                        'OOB_event_cnt': 1,

                        'annotation_start_point': 0,
                        'annotation_end_point': 1475,
                        'OOB_event_duration': 49.2,
                        'start_frame_idx': 0,
                        'end_frame_idx': 1475,
                        'start_frame_time':0,
                        'end_frame_time':49.166666666666667
                    }
                    
                    # add for patinets
                    anno_meta_info_dict = {
                        'totalFrame': 1476,
                        'fps': 30,
                        'anno_info': [[0, 1475]]
                    }

                # change to per video info in df
                print(anno_meta_info_per_each_video_dict)
                anno_meta_info_per_each_video_df = pd.DataFrame.from_records([anno_meta_info_per_each_video_dict])
                print(anno_meta_info_per_each_video_df)
            
                # append metric per video
                # columns=['Video_name', 'totalFrame', 'fps', 'IB_count', 'OOB_count', 'total_time', 'OOB_event_time', 'IB_event_time', 'OOB_event_cnt', 'annotation_start_point', 'annotation_end_point', 'start_frame_idx', 'end_frame_idx', 'OOB_event_duration']
                total_anno_meta_info_df = pd.concat([total_anno_meta_info_df, anno_meta_info_per_each_video_df], ignore_index=True) # should synk with col name
                print(total_anno_meta_info_df)
                    

            else :
        
                cnt += 1
                target_anno_path = all_anno_path[idx]
                
                # 5. parsing anno meta info & sanity check over frame
                anno_meta_info_dict = parser_anno_meta_info(target_anno_path)

                # per video info (each row)
                anno_meta_info_per_each_video_dict = {
                    'Video_name':video_name,
                    'Method': video_name.split('_')[0],
                    'totalFrame':EXCEPTION_NUM,
                    'fps':EXCEPTION_NUM,
                    'IB_count':EXCEPTION_NUM,
                    'OOB_count':EXCEPTION_NUM,
                    
                    'total_time': EXCEPTION_NUM,
                    'IB_event_time': EXCEPTION_NUM,
                    'OOB_event_time': EXCEPTION_NUM,
                    
                    'OOB_event_cnt': EXCEPTION_NUM,

                    'annotation_start_point': EXCEPTION_NUM,
                    'annotation_end_point': EXCEPTION_NUM,
                    'OOB_event_duration': EXCEPTION_NUM,
                    'start_frame_idx': EXCEPTION_NUM,
                    'end_frame_idx': EXCEPTION_NUM
                }

                

                anno_meta_info_per_each_video_dict['totalFrame'] = anno_meta_info_dict['totalFrame']
                anno_meta_info_per_each_video_dict['fps'] = anno_meta_info_dict['fps']
                anno_meta_info_per_each_video_dict['IB_count'] = anno_meta_info_dict['IB_count']
                anno_meta_info_per_each_video_dict['OOB_count']= anno_meta_info_dict['OOB_count']
                anno_meta_info_per_each_video_dict['OOB_event_cnt'] = anno_meta_info_dict['OOB_event_cnt']
                anno_meta_info_per_each_video_dict['annotation_start_point'] = anno_meta_info_dict['annotation_start_point']
                anno_meta_info_per_each_video_dict['annotation_end_point'] = anno_meta_info_dict['annotation_end_point']

                # 6. add more info
                anno_meta_info_dict['Video_name'] = video_name

                oob_duration_list = anno_meta_info_dict['OOB_event_duration'] # per video anno meta info
                anno_meta_info_per_each_video_dict['OOB_event_time'] = frame_to_sec(anno_meta_info_per_each_video_dict['OOB_count'], anno_meta_info_per_each_video_dict['fps'])
                anno_meta_info_per_each_video_dict['IB_event_time'] = frame_to_sec(anno_meta_info_per_each_video_dict['IB_count'], anno_meta_info_per_each_video_dict['fps'])
                anno_meta_info_per_each_video_dict['total_time'] = frame_to_sec(anno_meta_info_per_each_video_dict['totalFrame'], anno_meta_info_per_each_video_dict['fps']) 

                if not oob_duration_list: # empty (not exist annotation info)
                    NO_ANNOTATION_VIDEO.append(video_name)
                    anno_meta_info_per_each_video_dict['OOB_event_duration'] = 0
                    anno_meta_info_per_each_video_dict['start_frame_idx'] = EXCEPTION_NUM
                    anno_meta_info_per_each_video_dict['end_frame_idx'] = EXCEPTION_NUM

                    anno_meta_info_per_each_video_dict['start_frame_time'] = EXCEPTION_NUM
                    anno_meta_info_per_each_video_dict['end_frame_time'] = EXCEPTION_NUM

                    # 6. change to per video info in df
                    anno_meta_info_per_each_video_df = pd.DataFrame.from_records([anno_meta_info_per_each_video_dict])
                    print(anno_meta_info_per_each_video_df)
                
                    # 7. append metric per video
                    # columns=['Video_name', 'Method', 'totalFrame', 'fps', 'IB_count', 'OOB_count', 'total_time', 'OOB_event_time', 'IB_event_time', 'OOB_event_cnt', 'annotation_start_point', 'annotation_end_point', 'start_frame_idx', 'end_frame_idx', 'OOB_event_duration']
                    total_anno_meta_info_df = pd.concat([total_anno_meta_info_df, anno_meta_info_per_each_video_df], ignore_index=True) # should synk with col name
                    print(total_anno_meta_info_df)
                
                else : # not empty, save row until finish OOB event duration
                    event_duration = 0
                    start_frame_idx = EXCEPTION_NUM
                    end_frame_idx = EXCEPTION_NUM
                    for event_duration, (start_frame_idx, end_frame_idx) in zip(anno_meta_info_dict['OOB_event_duration'], anno_meta_info_dict['anno_info']):
                        anno_meta_info_per_each_video_dict['OOB_event_duration'] = event_duration
                        anno_meta_info_per_each_video_dict['start_frame_idx'] = start_frame_idx
                        anno_meta_info_per_each_video_dict['end_frame_idx'] = end_frame_idx

                        anno_meta_info_per_each_video_dict['start_frame_time'] = frame_to_sec(start_frame_idx, anno_meta_info_per_each_video_dict['fps'])
                        anno_meta_info_per_each_video_dict['end_frame_time'] = frame_to_sec(end_frame_idx, anno_meta_info_per_each_video_dict['fps'])
                
                        # 6. change to per video info in df
                        print(anno_meta_info_per_each_video_dict)
                        anno_meta_info_per_each_video_df = pd.DataFrame.from_records([anno_meta_info_per_each_video_dict])
                        print(anno_meta_info_per_each_video_df)
                    
                        # 7. append metric per video
                        # columns=['Video_name', 'Method', 'totalFrame', 'fps', 'IB_count', 'OOB_count', 'total_time', 'OOB_event_time', 'IB_event_time', 'OOB_event_cnt', 'annotation_start_point', 'annotation_end_point', 'start_frame', 'end_frame', 'OOB_event_duration']
                        total_anno_meta_info_df = pd.concat([total_anno_meta_info_df, anno_meta_info_per_each_video_df], ignore_index=True) # should synk with col name
                        print(total_anno_meta_info_df)



            # 8. for aggregation patinet info
            patient_total_frame_list.append(anno_meta_info_dict['totalFrame'])
            patient_annotation_info_list.append(anno_meta_info_dict['anno_info'])
            patient_fps_list.append(anno_meta_info_dict['fps'])

        ####### 여기 #####
        # 9. aggretation anno info for patinet
        print('PATIENT : {}'.format(patient))
        patient_anno_meta_info_dict = aggregation_annotation_info_for_patient(patient_total_frame_list, patient_annotation_info_list, patient_fps_list)

        anno_meta_info_per_each_patient_dict = {
            'Patient':patient,
            'Method': patient.split('_')[0],
            'totalFrame':EXCEPTION_NUM,
            'fps':EXCEPTION_NUM,
            'IB_count':EXCEPTION_NUM,
            'OOB_count':EXCEPTION_NUM,
            
            'total_time': EXCEPTION_NUM,
            'IB_event_time': EXCEPTION_NUM,
            'OOB_event_time': EXCEPTION_NUM,
            
            'OOB_event_cnt': EXCEPTION_NUM,

            'annotation_start_point': EXCEPTION_NUM,
            'annotation_end_point': EXCEPTION_NUM,
            'OOB_event_duration': EXCEPTION_NUM,
            'start_frame_idx': EXCEPTION_NUM,
            'end_frame_idx': EXCEPTION_NUM,

            'start_frame_time': EXCEPTION_NUM,
            'end_frame_time': EXCEPTION_NUM,
        }

        anno_meta_info_per_each_patient_dict['totalFrame'] = patient_anno_meta_info_dict['totalFrame']
        anno_meta_info_per_each_patient_dict['fps'] = patient_anno_meta_info_dict['fps']
        anno_meta_info_per_each_patient_dict['IB_count'] = patient_anno_meta_info_dict['IB_count']
        anno_meta_info_per_each_patient_dict['OOB_count'] = patient_anno_meta_info_dict['OOB_count']
        anno_meta_info_per_each_patient_dict['OOB_event_cnt'] = patient_anno_meta_info_dict['OOB_event_cnt']
        anno_meta_info_per_each_patient_dict['annotation_start_point'] = patient_anno_meta_info_dict['annotation_start_point']
        anno_meta_info_per_each_patient_dict['annotation_end_point'] = patient_anno_meta_info_dict['annotation_end_point']
        anno_meta_info_per_each_patient_dict['OOB_event_cnt'] = patient_anno_meta_info_dict['OOB_event_cnt']

        # add more info
        patient_anno_meta_info_dict['Patient'] = patient

        oob_duration_list = patient_anno_meta_info_dict['OOB_event_duration'] # per patient anno meta info
        anno_meta_info_per_each_patient_dict['OOB_event_time'] = frame_to_sec(anno_meta_info_per_each_patient_dict['OOB_count'], anno_meta_info_per_each_patient_dict['fps'])
        anno_meta_info_per_each_patient_dict['IB_event_time'] = frame_to_sec(anno_meta_info_per_each_patient_dict['IB_count'], anno_meta_info_per_each_patient_dict['fps'])
        anno_meta_info_per_each_patient_dict['total_time'] = frame_to_sec(anno_meta_info_per_each_patient_dict['totalFrame'], anno_meta_info_per_each_patient_dict['fps'])



        if not oob_duration_list: # empty (not exist annotation info)
            anno_meta_info_per_each_patient_dict['OOB_event_duration'] = 0
            anno_meta_info_per_each_patient_dict['start_frame_idx'] = EXCEPTION_NUM
            anno_meta_info_per_each_patient_dict['end_frame_idx'] = EXCEPTION_NUM

            anno_meta_info_per_each_patient_dict['start_frame_time'] = EXCEPTION_NUM
            anno_meta_info_per_each_patient_dict['end_frame_time'] = EXCEPTION_NUM

            # 6. change to per patient info in df
            anno_meta_info_per_each_patient_df = pd.DataFrame.from_records([anno_meta_info_per_each_patient_dict])
            print(anno_meta_info_per_each_patient_df)
        
            # 7. append metric per patient
            # columns=['Patient', 'Method', 'totalFrame', 'fps', 'IB_count', 'OOB_count', 'total_time', 'IB_event_time', 'OOB_event_time', 'OOB_event_cnt', 'annotation_start_point', 'annotation_end_point', 'start_frame_idx', 'end_frame_idx', 'start_frame_time', 'end_frame_time', 'OOB_event_duration']
            total_anno_meta_info_per_patient_df = pd.concat([total_anno_meta_info_per_patient_df, anno_meta_info_per_each_patient_df], ignore_index=True) # should synk with col name
            print(total_anno_meta_info_per_patient_df)
        
        else : # not empty, save row until finish OOB event duration
            event_duration = 0
            start_frame_idx = EXCEPTION_NUM
            end_frame_idx = EXCEPTION_NUM

            for event_duration, (start_frame_idx, end_frame_idx) in zip(patient_anno_meta_info_dict['OOB_event_duration'], patient_anno_meta_info_dict['anno_info']):
                anno_meta_info_per_each_patient_dict['OOB_event_duration'] = event_duration
                anno_meta_info_per_each_patient_dict['start_frame_idx'] = start_frame_idx
                anno_meta_info_per_each_patient_dict['end_frame_idx'] = end_frame_idx

                anno_meta_info_per_each_patient_dict['start_frame_time'] = frame_to_sec(start_frame_idx, anno_meta_info_per_each_patient_dict['fps'])
                anno_meta_info_per_each_patient_dict['end_frame_time'] = frame_to_sec(end_frame_idx, anno_meta_info_per_each_patient_dict['fps'])
        
                # 6. change to per patient info in df
                anno_meta_info_per_each_patient_df = pd.DataFrame.from_records([anno_meta_info_per_each_patient_dict])
                print(anno_meta_info_per_each_patient_df)
            
                # 7. append metric per patient
                # columns=['Patient', 'Method', 'totalFrame', 'fps', 'IB_count', 'OOB_count', 'total_time', 'OOB_event_time', 'IB_event_time', 'OOB_event_cnt', 'annotation_start_point', 'annotation_end_point', 'start_frame_idx', 'end_frame_idx', 'start_frame_time', 'end_frame_time','OOB_event_duration']
                total_anno_meta_info_per_patient_df = pd.concat([total_anno_meta_info_per_patient_df, anno_meta_info_per_each_patient_df], ignore_index=True) # should synk with col name
                print(total_anno_meta_info_per_patient_df)

            
    # save total df
    total_anno_meta_info_df.to_csv(os.path.join(results_dir, 'TOTAL_V2_anno_meta_info_per_video.csv'))
    total_anno_meta_info_per_patient_df.to_csv(os.path.join(results_dir, 'TOTAL_V2_anno_meta_info_per_patient.csv'))

    print('\n----- PROCESSED DATA CNT : {} | FAILED PROCESSED DATA CNT : {} ------\n'.format(cnt, len(USE_VIDEO_LIST)-cnt))
    print('FALIED DATA LIST : {}'.format(FAILD_VIDEO_NAME))
    print('NO_ANNOTATION_VIDEO LIST : {}'.format(NO_ANNOTATION_VIDEO))

def aggregation_annotation_info_for_patient(patient_total_frame_list, patient_annotation_info_list, fps_list):

    if len(np.unique(np.array(fps_list))) != 1 :
        print('\n\t--- --- fps list--- --- \n')
        print('BASE FPS : {}'.format(fps_list))
        print('UNIQUE FPS : {}'.format(np.unique(np.array(fps_list))))
        print('FPS UN CORRESPONDANCE {}'.format(fps_list))
        print('SET FPS : {}'.format(sum(fps_list) / len(fps_list)))
    

    patient_total_frame = 0
    patient_fps = sum(fps_list) / len(fps_list)
    patient_oob_event_cnt = 0
    patient_oob_event_duration = []
    patient_ib_count = 0
    patient_oob_count = 0
    patient_annotation_start_point = EXCEPTION_NUM
    patient_annotation_end_point = EXCEPTION_NUM
    patient_aggregation_anno_info = []

    
    # change annotation info for patinet level
    patient_total_frame = 0
    patient_annotation_info = []

    for video_len, anno_info in zip(patient_total_frame_list, patient_annotation_info_list):
        for start, end in anno_info :
            start_idx = start + patient_total_frame
            end_idx = end + patient_total_frame

            idx_list = [start_idx, end_idx]
            patient_annotation_info += idx_list

        patient_total_frame += video_len   

    for idx in range(0, len(patient_annotation_info), 2) :
        start_frame = patient_annotation_info[idx]
        end_frame = patient_annotation_info[idx+1]
        
        if idx == 0:
            patient_aggregation_anno_info.append([start_frame, end_frame])
    
        else : # 2~
            pre_start_frame = patient_annotation_info[idx-2]
            pre_end_frame = patient_annotation_info[idx-1]

            if pre_end_frame + 1 < start_frame :
                patient_aggregation_anno_info.append([start_frame, end_frame])
            else :
                print('AGGREGATE | (pre start : {}, pre end : {}) | (start : {}, end : {})'.format(pre_start_frame, pre_end_frame, start_frame, end_frame))
                patient_aggregation_anno_info.append([pre_start_frame, end_frame])
    
    # check count and  duration
    for start_frame_idx, end_frame_idx in patient_aggregation_anno_info :
        patient_oob_event_cnt += 1

        oob_duration = (end_frame_idx - start_frame_idx) + 1
        patient_oob_count += oob_duration
        patient_oob_event_duration.append(frame_to_sec(oob_duration, patient_fps)) # 1-5 == 1,2,3,4,5 frame, duration = 5 frame

        if patient_oob_event_cnt == 1: # init annotation
            patient_annotation_start_point = start_frame_idx

        if patient_oob_event_cnt == len(patient_aggregation_anno_info): # last anno
            patient_annotation_end_point = end_frame_idx


    patient_ib_count = patient_total_frame - patient_oob_count

    # set parsing info
    patient_anno_meta_info_dict = {}
    patient_anno_meta_info_dict['totalFrame'] = patient_total_frame
    patient_anno_meta_info_dict['fps'] = patient_fps
    patient_anno_meta_info_dict['IB_count'] = patient_ib_count
    patient_anno_meta_info_dict['OOB_count'] = patient_oob_count
    patient_anno_meta_info_dict['OOB_event_cnt'] = patient_oob_event_cnt
    patient_anno_meta_info_dict['OOB_event_duration'] = patient_oob_event_duration # list
    patient_anno_meta_info_dict['annotation_start_point'] = patient_annotation_start_point
    patient_anno_meta_info_dict['annotation_end_point'] = patient_annotation_end_point
    patient_anno_meta_info_dict['anno_info'] = patient_aggregation_anno_info # list

    

    print(patient_anno_meta_info_dict['OOB_event_duration'])
    print(patient_anno_meta_info_dict['anno_info'])
    
    return patient_anno_meta_info_dict



#### for save still cut of short oob event
# from subprocess import Popen, PIPE
def still_cut_of_oob_event(video_path, fps, start_idx, end_idx, margin, result_dir):

    cmds_list = []

    stdin_r, stdin_w = os.pipe()
    stdout_r, stdout_w = os.pipe()

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    os.makedirs(result_dir, exist_ok=True)

    # make all cmd
    '''
    for target_idx in range(start_idx-margin, end_idx+margin):

        out_path = os.path.join(result_dir, '{}-fps_{}-frame_{:010d}-time_{}.jpg'.format(video_name, fps, target_idx, idx_to_time(target_idx, fps)))

        meta_info = '{} | fps = {} | frame = {:010d} | sec = {:.2f}'.format(video_name, fps, target_idx, frame_to_sec(target_idx, fps))
        # meta_info = 'HELLO'

        # ffmpeg -i /data2/Public/IDC_21.06.25/Dataset1/01_G_01_L_301_xx0_01.MP4 -vframes 1 -vf "fps=30,select=eq(n\,222)" -vsync 0 ./ffmpeg/x.jpg
        if (target_idx >= start_idx) and (target_idx <= end_idx) : # emphasis
            cmd = 'ffmpeg -i {} -vframes 1 -vf "drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf: text=\[OOB\] - ({}) : x=(w-tw)/2: y=h-(2*lh): fontcolor=black: fontsize=30: box=1: boxcolor=red: boxborderw=5,fps={},select=eq(n\,{})" -vsync 0 {} -y'.format(video_path, meta_info, fps, target_idx, out_path)
            pass
        
        else :
            # cmd = 'ffmpeg -i {} -vframes 1 -vf "fps={},select=eq(n\,{})" -vsync 0 {}'.format(video_path, fps, target_idx, out_path)
            cmd = 'ffmpeg -i {} -vframes 1 -vf "drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf: text=\[IB\] - ({}) : x=(w-tw)/2: y=h-(2*lh): fontcolor=black: fontsize=30: box=1: boxcolor=white: boxborderw=5,fps={},select=eq(n\,{})" -vsync 0 {} -y'.format(video_path, meta_info, fps, target_idx, out_path)
            pass
            
        cmds_list.append([cmd])
        print(cmd)
    '''

    cmds_list = []

    pre_frame_idx = (start_idx - margin, start_idx - 1)
    post_frame_idx = (end_idx + 1, end_idx + margin)
    pre_frame_cnt = pre_frame_idx[1] - pre_frame_idx[0] + 1
    post_frame_cnt = post_frame_idx[1] - post_frame_idx[0] + 1
    target_frame_cnt = end_idx - start_idx + 1

    # make cmd (pre)
    text = "'" + '[{}]-({} | {}fps | {} | {})'.format('IB', video_name, fps, '%{'+'frame_num'+'}', '%{'+'pts \:hms'+'}' + "'")
    out_path = os.path.join(result_dir, '{}-frame_'.format(video_name) + '%d.jpg')
    cmd = ['ffmpeg', '-i', video_path, '-vframes', str(pre_frame_cnt), '-vf', '"' + 'drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf: text={}:x=(w-tw)/2: y=h-(2*lh): fontcolor=black: fontsize=30: box=1: boxcolor=white: boxborderw=5'.format(text) + ',' + 'fps=30' + ',' + "select='between(n\,{},{})'".format(pre_frame_idx[0], pre_frame_idx[1]) + '"', '-vsync', '0', '-start_number', str(pre_frame_idx[0]), out_path, '-y']
    cmd = ' '.join(cmd)
    cmd = cmd.replace("\'", "'")
    cmd = cmd.replace("\\\\", "\\")
    cmds_list.append([cmd])

    print(cmd)

    # make cmd (target)
    text = "'" + '[{}]-({} | {}fps | {} | {})'.format('OOB', video_name, fps, '%{'+'frame_num'+'}', '%{'+'pts \:hms'+'}' + "'")
    out_path = os.path.join(result_dir, '{}-frame_'.format(video_name) + '%d.jpg')
    cmd = ['ffmpeg', '-i', video_path, '-vframes', str(target_frame_cnt), '-vf', '"' + 'drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf: text={}:x=(w-tw)/2: y=h-(2*lh): fontcolor=black: fontsize=30: box=1: boxcolor=red: boxborderw=5'.format(text) + ',' + 'fps=30' + ',' + "select='between(n\,{},{})'".format(start_idx, end_idx) + '"', '-vsync', '0', '-start_number', str(start_idx), out_path, '-y']
    cmd = ' '.join(cmd)
    cmd = cmd.replace("\'", "'")
    cmd = cmd.replace("\\\\", "\\")
    cmds_list.append([cmd])

    print(cmd)

    # make cmd (post)
    text = "'" + '[{}]-({} | {}fps | {} | {})'.format('IB', video_name, fps, '%{'+'frame_num'+'}', '%{'+'pts \:hms'+'}' + "'")
    out_path = os.path.join(result_dir, '{}-frame_'.format(video_name) + '%d.jpg')
    cmd = ['ffmpeg', '-i', video_path, '-vframes', str(post_frame_cnt), '-vf', '"' + 'drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf: text={}:x=(w-tw)/2: y=h-(2*lh): fontcolor=black: fontsize=30: box=1: boxcolor=white: boxborderw=5'.format(text) + ',' + 'fps=30' + ',' + "select='between(n\,{},{})'".format(post_frame_idx[0], post_frame_idx[1]) + '"', '-vsync', '0', '-start_number', str(post_frame_idx[0]), out_path, '-y']
    cmd = ' '.join(cmd)
    cmd = cmd.replace("\'", "'")
    cmd = cmd.replace("\\\\", "\\")
    cmds_list.append([cmd])

    print(cmd)
    
    # procs_list = [subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE  , shell=True) for cmd in cmds_list]
    procs_list = [Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE, shell=True) for cmd in cmds_list]
    for proc in procs_list:
        print('Cutting ...')
        # proc.wait() # communicate() # 자식 프로세스들이 I/O를 마치고 종료하기를 기다림
        out = proc.communicate()
        print(out)
        '''
        try : 
            out, err = proc.communicate(timeout = 10)
            # print(out.strip())
            # print(err.strip())
        
        except subprocess.TimeoutExpired: 
            # print(out.strip())
            # print(err.strip())
            proc.terminate()
            proc.wait()

        '''

    
        print("Processes are done")
    


def collect_oob_event(anno_meta_info_csv_path, VIDEO_PATH_SHEET_path, results_dir, min_event_duration = 0.0, max_event_duration = 2.0):
    # columns=[Video_name, totalFrame, fps, IB_count, OOB_count, total_time, IB_event_time, OOB_event_time, OOB_event_cnt, annotation_start_point, annotation_end_point, start_frame_idx,end_frame_idx, OOB_event_duration, start_frame_time, end_frame_time]

    VIDEO_PATH_SHEET = load_yaml_to_dict(VIDEO_PATH_SHEET_path)
    os.makedirs(results_dir, exist_ok=True)

    anno_meta_info_df = pd.read_csv(anno_meta_info_csv_path)

    # check oob duration
    over_min_event_duration = anno_meta_info_df['OOB_event_duration'] > min_event_duration
    under_max_event_duration = anno_meta_info_df['OOB_event_duration'] < max_event_duration


    # extract target df
    target_df = anno_meta_info_df[over_min_event_duration & under_max_event_duration]

    target_df = target_df.sort_values(['OOB_event_duration'], ascending=[True])
    print(target_df)

    for idx, row in target_df.iterrows():
        print(row)
        
        video_name = row['Video_name']
        fps = row['fps']
        start_frame_idx = row['start_frame_idx']
        end_frame_idx = row['end_frame_idx']
        
        oob_event_duration = row['OOB_event_duration']
        margin = int(fps * 2) # 2 sec
        
        target_video_path = VIDEO_PATH_SHEET.get(video_name, '0')
        target_result_dir = os.path.join(results_dir, video_name, '{}-{}'.format(idx_to_time(start_frame_idx, fps), idx_to_time(end_frame_idx, fps)))

        os.makedirs(target_result_dir, exist_ok=True)
        

        print('\n--- --- --- ---\n')
        print('VIDEO_NAME : {} | TARGET VIDEO : {}'.format(video_name, target_video_path))
        print('\n--- --- --- ---\n')

        # still cut oob event
        still_cut_of_oob_event(target_video_path, fps, start_frame_idx, end_frame_idx, margin, target_result_dir)

        # make img to seqeuence gif
        # all_results_img_path = natsorted(glob.glob(target_result_dir +'/*.{}'.format('jpg')), key=lambda x : os.path.splitext(os.path.basename(x))[0].split('-')[2], alg=natsort.ns.INT) # 위에서 저장한 img 모두 parsing
        all_results_img_path = natsort.natsorted(glob.glob(target_result_dir +'/*.{}'.format('jpg')), key=lambda x : os.path.splitext(os.path.basename(x))[0].split('-')[-1], alg=natsort.ns.INT) # 위에서 저장한 img 모두 parsing

        img_seq_to_gif(all_results_img_path, os.path.join(target_result_dir, '{}-start-{}-end-{}-duration-{}.gif'.format(video_name, start_frame_idx, end_frame_idx, oob_event_duration))) # seqence 이므로 sort 하여 append

        print('\n\n=== === === === ===\n\n')

def main():
    
    # make_data_sheet('./DATA_SHEET')
    '''
    ANNOTATION_V2_ROOT_PATH = '/data2/Public/IDC_21.06.25/ANNOTATION/Gastrectomy/Event/OOB/V2'
    results_dir = './DATA_SHEET'
    
    print(len(OOB_robot_list + OOB_lapa_list), len(OOB_robot_list), len(OOB_lapa_list))
    
    gen_anno_meta_info(ANNOTATION_V2_ROOT_PATH, OOB_robot_list + OOB_lapa_list, ROBOT_CASE + LAPA_CASE, results_dir)
    '''

    # anno_meta_info_csv_path = './DATA_SHEET/ROBOT_V2_anno_meta_info_per_video.csv'
    # results_dir = './OOB_EVENT_NEW_2_5'
    # VIDEO_PATH_SHEET_path = './DATA_SHEET/VIDEO_PATH_SHEET.yaml'

    # collect_oob_event(anno_meta_info_csv_path, VIDEO_PATH_SHEET_path, results_dir, min_event_duration=2.0, max_event_duration=5.0)


    

if __name__ == "__main__":
	main()