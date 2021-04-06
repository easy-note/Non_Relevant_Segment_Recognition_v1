import gen_dataset

save_dir_path = '/data/CAM_IO/robot/new_images'
org_video_path = '/data'

### demo_images
# trainset = ['R001', 'R002', 'R003', 'R004', 'R005', 'R006', 'R007']
# valset = ['R017']
gen_dataset.gen_data('/data', '/data/CAM_IO/robot/demo_images')

#### new_images
# trainset = ['R001', 'R002', 'R003', 'R004', 'R005', 'R006', 'R007', 'R010', 'R013', 'R014', 'R015', 'R018', 
#            'R019', 'R048', 'R056', 'R074', 'R076', 'R084', 'R094', 'R100', 'R117', 'R201', 'R202', 'R203', 
#            'R204', 'R205', 'R206', 'R207', 'R209', 'R210', 'R301', 'R302', 'R304', 'R305', 'R313']

# valset = ['R017', 'R022', 'R116', 'R208', 'R303']
# gen_dataset.gen_data('/data', '/data/CAM_IO/robot/new_images')