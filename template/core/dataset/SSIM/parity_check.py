import os
import pandas as pd

pd.set_option('display.max_rows', None)

origin_f = '/raid/SSIM_RESULT/0.997-SSIM_RESULT/severance_2nd/01_VIHUB1.2_B4_L_148/01_VIHUB1.2_B4_L_148-5FPS.csv'
pp_f = '/raid/SSIM_RESULT/0.997-SSIM_RESULT/severance_2nd/01_VIHUB1.2_B4_L_148/pp-01_VIHUB1.2_B4_L_148-5FPS.csv'

origin_f = '/raid/SSIM_RESULT/0.997-SSIM_RESULT/gangbuksamsung_127case/04_GS4_99_L_4/04_GS4_99_L_4-5FPS.csv'
pp_f = '/raid/SSIM_RESULT/0.997-SSIM_RESULT/gangbuksamsung_127case/04_GS4_99_L_4/pp-04_GS4_99_L_4-5FPS.csv'

origin_data = pd.read_csv(origin_f)
origin_class = origin_data['class'].to_list()

print(origin_class.count(0))
print(origin_class.count(1))
print(origin_class.count(2))
print(origin_class.count(3))

print('\n')

pp_data = pd.read_csv(pp_f)
pp_class = pp_data['class'].to_list()

print(pp_class.count(0))
print(pp_class.count(1))
print(pp_class.count(2))
print(pp_class.count(3))