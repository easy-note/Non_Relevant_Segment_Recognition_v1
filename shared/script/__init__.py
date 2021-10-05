
print('hi')

import sys
from os import path

root_path = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
sys.path.append(root_path)
sys.path.append(root_path, 'core'))
sys.path.append(root_path, 'core', 'api'))
sys.path.append(root_path, 'core', 'api', 'Evaluation_vihub'))
sys.path.append(root_path, 'core', 'api', 'Inference_vihub'))
sys.path.append(root_path, 'core', 'api', 'Inference_vihub', 'VIHUB_pro_QA_v2'))

sys.path.append(root_path, 'core', 'utils'))

print(sys.path)