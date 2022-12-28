import os, sys, shutil
import json
import yaml

aoi_dir_name = os.getenv('AOI_DIR_NAME')
assert aoi_dir_name != None, "Environment variable AOI_DIR_NAME is None"

current_dir = os.path.dirname(os.path.abspath(__file__))
idx = current_dir.find(aoi_dir_name) + len(aoi_dir_name)
aoi_dir = current_dir[:idx]

# ========================== MY CONFIG =========================== #

# read path from yolov4.yaml
yml = yaml.safe_load(open('yolov4.yaml'))

# Original train dataset
ORI_train_crop_image_path = os.path.expanduser(yml['TRAIN_crop_image_path'])
ORI_train_crop_label_file = os.path.expanduser(yml['TRAIN_crop_label_file'])

# Original valid dataset
ORI_valid_crop_image_path = os.path.expanduser(yml['VALID_crop_image_path'])
ORI_valid_crop_label_file = os.path.expanduser(yml['VALID_crop_label_file'])

# yolo config file for load model
Yolo_config_path = yml['YOLO_config_path']
Yolo_data_file = os.path.join(Yolo_config_path, 'pcb.data')
Yolo_names_file = os.path.join(Yolo_config_path, 'pcb.names')
Yolo_cfg_file = os.path.join(Yolo_config_path, 'yolov4-pcb.cfg')
Yolo_train_file =  os.path.join(Yolo_config_path, 'train.txt')
Yolo_valid_file =  os.path.join(Yolo_config_path, 'valid.txt')

# previous model
now_best_model = yml['yolov4_old_model_file_path']

# yolo dataset path
Yolo_dataset_path = yml['YOLO_dataset_path']
Yolo_train_set_path = os.path.join(Yolo_dataset_path, 'train') 
Yolo_valid_set_path = os.path.join(Yolo_dataset_path, 'valid') 

# training models folder
Yolo_weights_path = yml['YOLO_weight_path']

# result folder
before_retrain_path = yml['before_retrain_path']
after_retrain_path = yml['after_retrain_path']
final_weight_path = yml['final_weight_path']

# input data detail
window_size = yml['window_size']
margin = yml['margin']

# ================================================================ #

def generate_yolo_dataset(input_image_path, input_json, target_path):
    
    #example: {'id': 0, 'category_id': 1, 'bbox': [237.0, 71.0, 38.0, 56.0],..., 'area': 2128.0, 'iscrowd': 0, 'image_id': 13}
    
    images_list = []
    defects = []
    
    # read input json file
    json_array = json.load(open (input_json))  
    
    # set the yolo label file
    for item in json_array['images']:
        images_list.append(item['file_name'])
        filename = item['file_name'].split('.', 1 )[0]
        
        fp = open(os.path.join(target_path, filename+".txt"), "a")
        fp.close()

    # get the defects categories
    for item in json_array['categories']:
        defects.append(item['name'])
    # print("defect categories:", len(defects))    
    
    # write label into label file
    for item in json_array['annotations']:
        id = item['id']
        img_name = images_list[int(item['image_id'])]
        category_id = int(item['category_id'])-1
        category_name = defects[category_id]
        [bbox_x,bbox_y,bbox_w,bbox_h] = item['bbox'][:]
        

        yolo_x = (float(bbox_x)+(float(bbox_w)/2)) / window_size
        yolo_y = (float(bbox_y)+(float(bbox_h)/2)) / window_size
        yolo_w = float(bbox_w) / window_size
        yolo_h = float(bbox_h) / window_size

        filename = img_name.split('.', 1 )[0]
        fp = open(os.path.join(target_path, filename+".txt"), "a")
        fp.write("%s %f %f %f %f\n" % (category_id, yolo_x, yolo_y, yolo_w, yolo_h))
        fp.close()  
    # print("label counts:", str(id))     
        
    # copy train image crop data
    for img_name in os.listdir(input_image_path):
        shutil.copyfile(os.path.join(input_image_path, img_name), os.path.join(target_path, img_name))


    return len(images_list), len(defects)




# 1. check previous best training model is exists or not
if not os.path.exists(now_best_model):
    print("pre-train model not exists! find previous model from retrain folder")
    if not os.path.exists(final_weight_path):
        print("Can not find the pre-train model!")
        exit()
    else:
        weights_list = []
        for weight in os.listdir(final_weight_path):
            if weight.endswith(".weights"):
                weights_list.append(weight.split(".")[0])
                shutil.copyfile(os.path.join(final_weight_path, weight), now_best_model)
    
        
        if len(weights_list)< 1:
            print("Can not find the pre-train model!")
            exit()        
else:
    print("old best yolov4 model:", now_best_model)


# 2. empty previous training model
if os.path.exists(Yolo_weights_path):
    shutil.rmtree(os.path.abspath(Yolo_weights_path), ignore_errors=True)
os.makedirs(Yolo_weights_path)



# 3. prepare yolov4 dataset
if os.path.exists(Yolo_train_set_path):
    print("Yolo_train_set_path: ",Yolo_train_set_path)
    shutil.rmtree(os.path.abspath(Yolo_train_set_path), ignore_errors=True)
if os.path.exists(Yolo_valid_set_path):
    print("Yolo_valid_set_path: ",Yolo_valid_set_path)
    shutil.rmtree(os.path.abspath(Yolo_valid_set_path), ignore_errors=True)
if os.path.exists(Yolo_dataset_path):
    print("Yolo_dataset_path: ",Yolo_dataset_path)
    shutil.rmtree(os.path.abspath(Yolo_dataset_path), ignore_errors=True)
os.makedirs(Yolo_dataset_path)
os.makedirs(Yolo_train_set_path)
os.makedirs(Yolo_valid_set_path)



# 4. generate yolov4 data file
train_dataset_size, train_dataset_categories_size = generate_yolo_dataset(ORI_train_crop_image_path, ORI_train_crop_label_file, Yolo_train_set_path)
print("train dataset size:", train_dataset_size)
print("train dataset categories:", train_dataset_categories_size)

valid_dataset_size, valid_dataset_categories_size = generate_yolo_dataset(ORI_valid_crop_image_path, ORI_valid_crop_label_file, Yolo_valid_set_path)
print("valid dataset size:", valid_dataset_size)
print("valid dataset categories:", valid_dataset_categories_size)



# 6. generate train.txt & valid.txt
fp = open(Yolo_train_file, "w")
for img in os.listdir(Yolo_train_set_path):
    abs_path = os.path.abspath(Yolo_train_set_path)
    if img.endswith(".jpg"):
        fp.write("%s\n" % os.path.join(abs_path, img))
fp.close()

fp = open(Yolo_valid_file, "w")
for img in os.listdir(Yolo_valid_set_path):
    abs_path = os.path.abspath(Yolo_valid_set_path)
    if img.endswith(".jpg"):
        fp.write("%s\n" % os.path.join(abs_path, img))
fp.close()


