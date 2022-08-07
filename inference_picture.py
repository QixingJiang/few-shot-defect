'''
antohr: JQX
date: 2022/08/07      1:53 PM
Introduce:
    Using the config file and trained model to inference/detect image and save the bbox information and result image.
'''
from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv
import os
import pandas as pd
import json
import torch
import numpy as np
import matplotlib
import cv2

config_file = './configs/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco.py'
checkpoint_file = 'work_dirs/cascade_rcnn_r50_fpn_1x_coco/latest.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

d = {}
# the path of images to be inferenced/detected
image_path = "/opt/data/private/jqx/yolov5/test"
# the path to save the inference result (bbox txt) according to the few-shot competition form
save_path = '/opt/data/private/jqx/yolov5/test_result'
# the path to save the inference result images
path_to_save_result_img ='/opt/data/private/jqx/yolov5/test_result_img'

# get the test image list !
piclist = os.listdir(image_path)

# create the save path


def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]  # bbox´ò·Ö

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # ´ò·Ö´Ó´óµ½Ð¡ÅÅÁÐ£¬È¡index
    order = scores.argsort()[::-1]
    # keepÎª×îºó±£ÁôµÄ±ß¿ò
    keep = []
    while order.size > 0.0:
        # order[0]ÊÇµ±Ç°·ÖÊý×î´óµÄ´°¿Ú£¬¿Ï¶¨±£Áô
        i = order[0]
        keep.append(i)
        # ¼ÆËã´°¿ÚiÓëÆäËûËùÓÐ´°¿ÚµÄ½»µþ²¿·ÖµÄÃæ»ý
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        # ½»/²¢µÃµ½iouÖµ
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        # indsÎªËùÓÐÓë´°¿ÚiµÄiouÖµÐ¡ÓÚthresholdÖµµÄ´°¿ÚµÄindex£¬ÆäËû´°¿Ú´Ë´Î¶¼±»´°¿ÚiÎüÊÕ
        inds = np.where(ovr <= thresh)[0]
        # orderÀïÃæÖ»±£ÁôÓë´°¿Úi½»µþÃæ»ýÐ¡ÓÚthresholdµÄÄÇÐ©´°¿Ú£¬ÓÉÓÚovr³¤¶È±Èorder³¤¶ÈÉÙ1(²»°üº¬i)£¬ËùÒÔinds+1¶ÔÓ¦µ½±£ÁôµÄ´°¿Ú
        order = order[inds + 1]

    return dets[keep]

def write_result_txt():
    for pic_name in piclist:
        pic_path = os.path.join(image_path, pic_name)
        # print(pic_name + ':')
        result = inference_detector(model, pic_path)
        # print(result[0])
        show_result_pyplot(model,pic_path,result)
        result = [py_cpu_nms(result[0], 0.1)]

        # print(keep)

        boxes = []
        # print(result)
        for i in range(1):
            for box in result[i]:
                # ×ª»»³ÉÁÐ±í
                cbox = []
                copybox = box.tolist()

                if i == 0:
                    copybox.append('defect')

                # print(copybox)
                cbox.append('defect')
                cbox.append(copybox[4])
                cbox.extend(copybox[:4])

                # ÖÃÐÅ¶È
                if copybox[-2] >= 0.75:
                    boxes.append(cbox)
                #print(copybox[-2])
        boxes.sort(key=lambda x: x[0])
        # print(boxes)

        f_name = pic_name.split(".")[0] + ".txt"
        # print(os.path.join(save_path, f_name))
        f = open(os.path.join(save_path, f_name), 'w')
        for i in range(len(boxes)):
            for j in range(len(boxes[i])):
                if j == 0:
                    f.write(str(boxes[i][j]) + " ")
                elif j == 1:
                    f.write(str(round(boxes[i][j], 6)) + " ")
                elif j != 5:
                    f.write(str(int(boxes[i][j])) + " ")
                else:
                    f.write(str(int(boxes[i][j])))
            f.write('\n')
        f.close()

def show_result_pyplot(model, img, result, score_thr=0.5, fig_size=(15, 10)):
    """Visualize the detection results on the image.
    Args:
        model (nn.Module): The loaded detector.
        img (str or np.ndarray): Image filename or loaded image.
        result (tuple[list] or list): The detection result, can be either
            (bbox, segm) or just bbox.
        score_thr (float): The threshold to visualize the bboxes and masks.
        fig_size (tuple): Figure size of the pyplot figure.
    """
    if hasattr(model, 'module'):
        model = model.module
    img = model.show_result(img, result, score_thr=score_thr, show=False)
    # not show the picture (otherwise you have to close the window every image)
    return img
    # plt.figure(figsize=fig_size)
    # plt.imshow(mmcv.bgr2rgb(img))
    # plt.show()
def save_result_img():
    # save the picture
    for pic_name in piclist:
        pic_path = os.path.join(image_path, pic_name)
        result = inference_detector(model, pic_path)
        img = show_result_pyplot(model, pic_path, result, score_thr=0.5)
        cv2.imwrite("{}/{}.jpg".format(path_to_save_result_img, pic_name), img)
        print("finish saving the picture:{}".format(pic_name))

if __name__ =="__main__":


    # create the file to put the save result 
    if not os.path.isdir(save_path):
    	os.mkdir(save_path)
    if not os.path.exists(path_to_save_result_img):
        os.mkdir(path_to_save_result_img)

    write_result_txt()

    save_result_img()




