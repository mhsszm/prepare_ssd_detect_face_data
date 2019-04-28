# -*- coding: utf-8 -*-

# from mtcnn.base_function import *
import numpy as np
import cv2
from mtcnn.mtcnn import mtcnn
import math
import os
import time
import sys
image_dir = './Images/001'
save_txt_dir = './Labels/001'

#run 2 times
#0 first delete pic with no face
#1 second detect face and write position to txt files
if (len(sys.argv) != 2 ):        
    raise Exception,u"arguments needed"
coporation_selection= int(sys.argv[1])
print (coporation_selection)
if __name__ == "__main__":
    
    m_mtcnn = mtcnn()

    print ("Load model end!!!")
    rgb_files=os.listdir(image_dir) 
    rgb_files.sort()

    # while(1):
    row0 = 1
    row1 = 1
    row2 = 1
    for x in range(len(rgb_files)):
        image_rgb_path = image_dir +'/' + rgb_files[x]        
        
        image_rgb = cv2.imread(image_rgb_path)
        
        image_rgb=cv2.cvtColor(image_rgb,cv2.COLOR_BGR2RGB)
       

        rgb_boxes, rgb_points, status = m_mtcnn.detect_face(image_rgb, intput_type = 2, output_type = 0)
       

        if (not (type(rgb_boxes) is np.ndarray)) or rgb_boxes.shape[0]==0:
            os.remove(image_rgb_path)
            print("remove file : {}.".format(image_rgb_path))
        else :
            if coporation_selection == 0:
                pass
            else:
                str_pos = "{0} {1} {2} {3}".format(int(rgb_boxes[0][0]),int(rgb_boxes[0][1]),int(rgb_boxes[0][2]),int(rgb_boxes[0][3]))
                file_name = rgb_files[x].split('.')[0]
                save_path = save_txt_dir + '/' + file_name + '.txt'
                print(save_path)
                # if (os.path.exists(save_path)):
                #     pass
                # else:
                #     os.makedirs(save_path,0755) 
                with open(save_path,'wb') as f:
                    f.write(str_pos)

            



    # output_workbook.save(output_file)
    # print("all_num is {0} ,negative_one_num is  {1} negative_two_num is{2} positive_one_num is {3}.".format(all_num,negative_one_num,negative_two_num,positive_one_num))

