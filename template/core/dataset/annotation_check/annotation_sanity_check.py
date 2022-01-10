import os
import glob
import csv


severance_1st_path = '/data3/Public/NRS_Recog/annotation/Gastrectomy/Lapa/v3/severance_1st'
severance_2nd_path = '/data3/Public/NRS_Recog/annotation/Gastrectomy/Lapa/v3/severance_2nd'
gangbuksamsunt_path = '/data3/Public/NRS_Recog/annotation/Gastrectomy/Lapa/v3/gangbuksamsung_127case'


def anno_sanity_check(target_anno_path):
    nrs_anno_path = os.path.join(target_anno_path, 'NRS')
    nrs_anno_list = glob.glob(os.path.join(nrs_anno_path, '*'))

    nrs_anno_list_sort = sorted(nrs_anno_list)
    to_csv_list = []
    dummy_list = []
    
    for nrs_anno in nrs_anno_list_sort:
        if '_'.join(nrs_anno.split('/')[-1].split('_')[:5]) in dummy_list:
            patient_no = ''
        else:
            patient_no = '_'.join(nrs_anno.split('/')[-1].split('_')[:5])
            dummy_list.append(patient_no)
        
        to_csv_list.append([patient_no, nrs_anno.split('/')[-1]])

    
    with open('assets/{}.csv'.format(target_anno_path.split('/')[-1]+'-video_list'), 'w') as f:
        writer = csv.writer(f)
        writer.writerows(to_csv_list)

def compare_oob_nrs(target_anno_path):
    nrs_anno_path = os.path.join(target_anno_path, 'NRS')
    nrs_anno_list = glob.glob(os.path.join(nrs_anno_path, '*'))

    oob_anno_path = os.path.join(target_anno_path, 'OOB')
    oob_anno_list = glob.glob(os.path.join(nrs_anno_path, '*'))

    print(len(nrs_anno_list))
    print(len(oob_anno_list))



if __name__ == '__main__':
    # anno_sanity_check(gangbuksamsunt_path)
    compare_oob_nrs(gangbuksamsunt_path)
    