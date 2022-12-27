import os, sys
import csv
import time
import numpy as np
import darknet
import cv2

aoi_dir_name = os.getenv('AOI_DIR_NAME')
assert aoi_dir_name != None, "Environment variable AOI_DIR_NAME is None"

current_dir = os.path.dirname(os.path.abspath(__file__))
idx = current_dir.find(aoi_dir_name) + len(aoi_dir_name)
aoi_dir = current_dir[:idx]

sys.path.append(os.path.join(aoi_dir, "config"))
from crop_small_image import crop_sliding_window

###################### MY CONFIG #######################################
Result_folder = './result/'
Test_dataset = './test_v3/'
Image_folder = 'images_wo_border/'
Label_folder = 'labels_wo_border/'

ORI_image_path = Test_dataset+Image_folder

Yolo_result_label_json_dir = Result_folder+'label_json_dir/'


Window_size = 512
Margin = 100

Score_threshold = 0.01

Edge_limit = 20 #pixels

Batch_size = 5

# yolo config file for load model
Yolo_config_path = './cfg/'
Yolo_data_file = Yolo_config_path+'pcb.data'
Yolo_cfg_file = Yolo_config_path+'yolov4-pcb.cfg'
Yolo_weights_file = Yolo_config_path+'yolov4-pcb_best.weights'
Yolo_result_csv = "20T_yolo_result.csv"

#######################################################################

def check_batch_shape(images, batch_size):
    """
        Image sizes should be the same width and height
    """
    shapes = [image.shape for image in images]
    if len(set(shapes)) > 1:
        raise ValueError("Images don't have same shape")
    if len(shapes) > batch_size:
        raise ValueError("Batch size higher than number of images")
    return shapes[0]


def prepare_batch(images, network, channels=3):
    width = darknet.network_width(network)
    height = darknet.network_height(network)

    darknet_images = []
    for image in images:
        image_resized = cv2.resize(image, (width, height),interpolation=cv2.INTER_LINEAR)
        custom_image = image_resized.transpose(2, 0, 1)
        darknet_images.append(custom_image)

    batch_array = np.concatenate(darknet_images, axis=0)
    batch_array = np.ascontiguousarray(batch_array.flat, dtype=np.float32)/255.0
    darknet_images = batch_array.ctypes.data_as(darknet.POINTER(darknet.c_float))
    return darknet.IMAGE(width, height, channels, darknet_images)

def batch_detection(patches_list, patches_position, filename, network, class_names, class_colors, \
                    thresh,edge_limit,hier_thresh, nms, batch_size):

    window_size = patches_list[0].shape[0]
    images_list = patches_list.copy()
    i = 0
    YOLO_detections = []

    images_cnt = len(images_list)
    
    while images_cnt:
        patches = []
        now_patch_idx = i
        now_size = min(images_cnt,batch_size)
        for idx in range(now_size):
            patches.append(images_list[i])
            i += 1
            images_cnt -= 1
            
        image_height, image_width, _ = check_batch_shape(patches, now_size)
        darknet_images = prepare_batch(patches, network)

        batch_detections = darknet.network_predict_batch(network, darknet_images, now_size, window_size, window_size, thresh, hier_thresh, None, 0, 0)
        i = now_patch_idx
        for idx in range(now_size):
            num = batch_detections[idx].num
            detections = batch_detections[idx].dets
            if nms:
                darknet.do_nms_obj(detections, num, len(class_names), nms)
            predictions = darknet.remove_negatives(detections, class_names, num)

            YOLO_detections += detection_process(predictions,filename,patches_position[i],edge_limit,window_size)
            i += 1
                
        darknet.free_batch_detections(batch_detections, now_size)
    
    return YOLO_detections


def image_detection(patches_list, patches_position, filename, network, class_names, class_colors, thresh,edge_limit):
    # Darknet doesn't accept numpy images.
    # Create one with image we reuse for each detect
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    darknet_image = darknet.make_image(width, height, 3)

    YOLO_detections = []


    for i, patch in enumerate(patches_list):
        
        patch_resized = cv2.resize(patch, (width, height), interpolation=cv2.INTER_LINEAR)
        darknet.copy_image_from_bytes(darknet_image, patch_resized.tobytes())
        
        #detect patch
        detections = darknet.detect_image(network, class_names, darknet_image, thresh=thresh)

        #detection result process
        window_size = patch.shape[0]
        YOLO_detections += detection_process(detections,filename,patches_position[i],edge_limit,window_size)
        

    darknet.free_image(darknet_image)
    return  YOLO_detections

def detection_process(detections,filename,position,edge_limit,window_size):
    selected_detections = []
    edge_limit_max = window_size-edge_limit
    edge_limit_min = edge_limit

    # data = [type, score, [l,t,r,b]]
    for data in detections:
        if data[2][0] < edge_limit_min or data[2][1] < edge_limit_min or data[2][0] > edge_limit_max or data[2][1] > edge_limit_max :continue
        if data[2][2] > window_size or data[2][3] > window_size: continue

        new_w = int(data[2][2])
        new_h = int(data[2][3])
        new_l = int(data[2][0] - (new_w/2)) + position[0]
        new_t = int(data[2][1] - (new_h/2)) + position[1]
        new_r = int(data[2][0] + (new_w/2)) + position[0]
        new_b = int(data[2][1] + (new_h/2)) + position[1]
        confidence = str(round(data[1],4))

        new_data = [filename,data[0],new_l,new_t,new_r,new_b,confidence]
        selected_detections.append(new_data)

    return selected_detections


def write_data_to_YOLO_csv( yolo_data,yolo_result_csv,result_folder,method):
	# 開啟輸出的 CSV 檔案
	with open(result_folder + yolo_result_csv, method, newline='') as csvFile:
		# 建立 CSV 檔寫入器
		writer = csv.writer(csvFile)
		for data in yolo_data:
			writer.writerow(data)
	return

if __name__ == '__main__':
	

	# load model
	network, class_names, class_colors = darknet.load_network(
		Yolo_cfg_file,
		Yolo_data_file,
		Yolo_weights_file,
		Batch_size
	)
	

	#load all test board images
	images_list = []
	for img in os.listdir(ORI_image_path):
		if img.endswith(".jpg"):
			images_list.append(img)
	images_list.sort()
    

	max_image_name = ""
	max_image_time = 0
	min_image_name = ""
	min_image_time = 30

	img_cnt = 0

	# detection for all test board images
	for img in images_list:
		img_cnt +=1
        # read big image for board
		I = cv2.imread(ORI_image_path+img)
		I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
		I = I.astype(np.uint8)
		print("now process:",img_cnt,img)

		
		# crop big img to patches
		crop_rect_list, crop_image_list = crop_sliding_window(I)
		print("total patches: ",len(crop_image_list))
		
		# start detect time
		start = time.time()

		# detection	
		if Batch_size == 1 :
			yolo_data = image_detection(crop_image_list, crop_rect_list, img, network, class_names, class_colors, \
                                        Score_threshold,Edge_limit)
		else : 
			yolo_data = batch_detection(crop_image_list, crop_rect_list, img, network, class_names, class_colors, \
                                        Score_threshold, Edge_limit, .5, .45, Batch_size)

		detect = time.time()

		# end detect time
		end = time.time()
        
        
		# 輸出結果
		detect_time = detect-start
		process_time = end - start
		print("bbox count: ",len(yolo_data))
		print("detect time : %f s" % (detect_time))
		print("each big img processing time : %f s" % (process_time))
		print("*" * 100)

		
		if process_time >= max_image_time:
			max_image_time = process_time
			max_image_name = img
		
		if process_time <= min_image_time:
			min_image_time = process_time
			min_image_name = img
		

	print("range of process_time: ", max_image_time," ~ ", min_image_time, " s")
	print("max_image: ", max_image_name)
	print("min_image: ", min_image_name)