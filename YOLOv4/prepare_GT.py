import json
import os, shutil
import csv
import numpy as np
import yaml


# ========================== MY CONFIG =========================== #

# read path from yolov4.yaml
yml = yaml.safe_load(open('yolov4.yaml'))

VAL_path = os.path.expanduser(yml['VAL_image_path'])
VAL_image_path = os.path.join(VAL_path, 'images_wo_border')
VAL_label_path = os.path.join(VAL_path, 'labels_wo_border')

TEST_path = os.path.expanduser(yml['TEST_image_path'])
TEST_image_path = os.path.join(TEST_path, 'images_wo_border')
TEST_label_path = os.path.join(TEST_path, 'labels_wo_border')

RETRAIN_FN_path = os.path.expanduser(yml['RETRAIN_crop_image_path'])
RETRAIN_FN_crop_image_path = os.path.join(RETRAIN_FN_path, 'images_random_crop')
RETRAIN_FN_crop_label_path = os.path.join(RETRAIN_FN_path, 'labels_random_crop')

Yolo_config_path = yml['YOLO_config_path']

# output valid.csv
test_FN_csv = os.path.join(Yolo_config_path, yml['test_FN_csv'])
test_GT_csv = os.path.join(Yolo_config_path, yml['test_GT_csv'])
valid_GT_100_csv = os.path.join(Yolo_config_path, yml['valid_GT_100_csv'])


defects = ["appearance",
		   "appearance_hole",
		   "appearance_less",
		   "bridge",
           "empty",
		   "excess_solder",
           "solder_ball",
           "other",
           "kneel"]
          
######################################################################

def write_inference_data(data,result_csv,method):
    defect_types_count_list = np.zeros(len(defects))
    # CSV
    with open(result_csv, method, newline='') as csvFile:
        #
        writer = csv.writer(csvFile)
        for d in data:
            defect_type = defects.index(d[1])
            defect_types_count_list[defect_type] += 1
            writer.writerow(d)
    return defect_types_count_list
    
def get_json_data(file_list,label_path):

    bbox_data = []
    for img in file_list:
    
        # read json array
        json_array = json.load(open (os.path.join(label_path, img+".json")))
        
        g = 0
        for item in json_array["shapes"]:
            g = g+1
            
            #print(item)
    
            defect_type = defects.index(item['label'])
            if defect_type >= 6: continue
            x = int(item['points'][0][0])
            y = int(item['points'][0][1])
            x2 = int(item['points'][1][0])
            y2 = int(item['points'][1][1])
    
            err_w = x2 - x
            err_h = y2 - y
    
            wrt_row = [img+'.jpg', item['label'], str(x), str(y), str(err_w), str(err_h)]
            bbox_data.append(wrt_row)
            #print(wrt_row)


    return bbox_data



# validation 100 pics
file_list = []
for img in os.listdir(VAL_image_path):
    if img.endswith(".jpg"):
        file_list.append(img.split(".")[0])
file_list.sort()

valid_100_data = get_json_data(file_list,VAL_label_path)
valid_defect_count = write_inference_data(valid_100_data,valid_GT_100_csv,'w')
print("valid_defect_count: ",valid_defect_count)


# test 1057 pics?
file_list = []
for img in os.listdir(TEST_image_path):
    if img.endswith(".jpg"):
        file_list.append(img.split(".")[0])
file_list.sort()

test_GT_data = get_json_data(file_list,TEST_label_path)
test_defect_count = write_inference_data(test_GT_data,test_GT_csv,'w')
print("test_defect_count: ", test_defect_count)


# FN ? pics
file_list = []
for img in os.listdir(RETRAIN_FN_crop_image_path):
    if img.endswith(".jpg"):
        file_list.append(img.split(".")[0])
file_list.sort()

test_FN_data = get_json_data(file_list,RETRAIN_FN_crop_label_path)
FN_defect_count = write_inference_data(test_FN_data,test_FN_csv,'w')
print("FN_defect_count: ", FN_defect_count)
