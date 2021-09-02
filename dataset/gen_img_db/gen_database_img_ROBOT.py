import os
import subprocess
import natsort

import re

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

# 871ea, 100case
OOB_lapa_list = [
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
    'L_661_xx0_15', 'L_669_xx0_01', 'L_669_xx0_02', 'L_669_xx0_03', 'L_669_xx0_04', 'L_676_xx0_01', 'L_676_xx0_02', 'L_676_xx0_03', 'L_676_xx0_04', 'L_676_xx0_05', 
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


# list = []
# for i in OOB_lapa_list:
#     if i.split('_')[1] not in list:
#         list.append(i.split('_')[1])
# print(list)
# print(len(list))
# print(len(OOB_lapa_list))

robot_40_video_base_path = '/data1/HuToM/Video_Robot_cordname'
robot_60_video_base_path = '/data2/Video/Robot/Dataset2_60case'

img_base_path = '/raid/img_db'

def convert_patient_num(patient_num):
    return ''.join(re.findall('[1-9]\d*', patient_num))

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('ERROR: Creating directory {}'.format(directory))

def gen_dataset_using_ffmpeg(input_video_path, output_dir_path):
    ## ffmpeg -i 'input_vidio_path' 'output_dir_path/01_G_01_R_1_ch1_3-%010d.jpg'
    print('\nProcessing ====>\t {}\n'.format(os.path.join(output_dir_path, '{}'.format(output_dir_path.split('/')[-1]))))
    output_img_path = os.path.join(output_dir_path, '{}-%010d.jpg'.format(output_dir_path.split('/')[-1]))
    cmd = ['ffmpeg', '-i', input_video_path, '-start_number', '0', '-vsync', '0', '-vf', 'scale=512:512']
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
                    output_dir_path = os.path.join(img_base_path, 'ROBOT', 'R_{}'.format(patient_num), full_rename_file)
                    
                    createFolder(output_dir_path)
                    gen_dataset_using_ffmpeg(os.path.join(root, file), output_dir_path)
                    save_log(os.path.join(img_base_path, 'ROBOT_database_log.txt'), 'Robot_40case | Origin_video: {}\t|\t Rename_file: {}\n'.format(os.path.join(root, file), full_rename_file))
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
                    output_dir_path = os.path.join(img_base_path, 'ROBOT', 'R_{}'.format(patient_num), full_rename_file)
                    
                    createFolder(output_dir_path)
                    gen_dataset_using_ffmpeg(os.path.join(root, file), output_dir_path)
                    save_log(os.path.join(img_base_path, 'ROBOT_database_log.txt'), 'Robot_40case | Origin_video: {}\t|\t Rename_file: {}\n'.format(os.path.join(root, file), full_rename_file))
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
                    output_dir_path = os.path.join(img_base_path, 'ROBOT', 'R_{}'.format(patient_num), full_rename_file)
                    
                    createFolder(output_dir_path)
                    gen_dataset_using_ffmpeg(os.path.join(root, file), output_dir_path)
                    save_log(os.path.join(img_base_path, 'ROBOT_database_log.txt'), 'Robot_40case | Origin_video: {}\t|\t Rename_file: {}\n'.format(os.path.join(root, file), full_rename_file))
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
                    output_dir_path = os.path.join(img_base_path, 'ROBOT', 'R_{}'.format(patient_num), full_rename_file)
                    createFolder(output_dir_path)

                    gen_dataset_using_ffmpeg(os.path.join(root, file), output_dir_path)
                    save_log(os.path.join(img_base_path, 'ROBOT_database_log.txt'), 'Robot_60case | Origin_video: {}\t|\t Rename_file: {}\n'.format(os.path.join(root, file), full_rename_file))
                    print('Robot_60case | Origin_video: {}\t|\t Rename_file: {}\n'.format(os.path.join(root, file), full_rename_file))

           # R_391 예외 비디오.
            elif re.search('r000391/01_g_01_r_391_ch2_06.mp4', os.path.join(root, file).lower()):
                full_rename_file = os.path.join(root, file).split('/')[-1].split('.')[0]
                patient_num = full_rename_file.split('_')[4]

                output_dir_path = os.path.join(img_base_path, 'ROBOT' , 'R_{}'.format(patient_num), full_rename_file)
                createFolder(output_dir_path)

                gen_dataset_using_ffmpeg(os.path.join(root, file), output_dir_path)
                save_log(os.path.join(img_base_path, 'ROBOT_database_log.txt'), 'Robot_60case | Origin_video: {}\t|\t Rename_file: {}\n'.format(os.path.join(root, file), full_rename_file))
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

