# -*- coding: utf-8 -*-
import numpy as np
import cv2
import math
import time
import datetime

# function [boundingbox] = bbreg(boundingbox,reg)
def bbreg(boundingbox,reg):
    # calibrate bounding boxes
    if reg.shape[1]==1:
        reg = np.reshape(reg, (reg.shape[2], reg.shape[3]))

    w = boundingbox[:,2]-boundingbox[:,0]+1
    h = boundingbox[:,3]-boundingbox[:,1]+1
    b1 = boundingbox[:,0]+reg[:,0]*w
    b2 = boundingbox[:,1]+reg[:,1]*h
    b3 = boundingbox[:,2]+reg[:,2]*w
    b4 = boundingbox[:,3]+reg[:,3]*h
    boundingbox[:,0:4] = np.transpose(np.vstack([b1, b2, b3, b4 ]))
    return boundingbox
 
def generateBoundingBox(imap, reg, scale, threshold, img_h, img_w, debug =0):
    # use heatmap to generate bounding boxes
    stride=2
    cellsize=12

    imap = np.transpose(imap)
    dx1 = np.transpose(reg[:,:,0])
    dy1 = np.transpose(reg[:,:,1])
    dx2 = np.transpose(reg[:,:,2])
    dy2 = np.transpose(reg[:,:,3])

    y, x = np.where(imap >= threshold)
    if y.shape[0]==1:
        dx1 = np.flipud(dx1)
        dy1 = np.flipud(dy1)
        dx2 = np.flipud(dx2)
        dy2 = np.flipud(dy2)
    score = imap[(y,x)]

    reg = np.transpose(np.vstack([ dx1[(y,x)], dy1[(y,x)], dx2[(y,x)], dy2[(y,x)] ]))
    if reg.size==0:
        reg = np.empty((0,3))
    bb = np.transpose(np.vstack([y,x]))

    # q1  top right corner (absolute)
    # q2  lower right corner (absolute)
    q1 = np.fix((stride*bb+1)/scale)
    q2 = np.fix((stride*bb+cellsize-1+1)/scale)

    q2[np.where((q2[:,0])>img_w),0] = img_w
    q2[np.where((q2[:,1])>img_h),1] = img_h

    # print (q1)
    # print (q2)
    # print (img_w, img_h)
    if debug:
    	exit()
    boundingbox = np.hstack([q1, q2, np.expand_dims(score,1), reg])

    return boundingbox, reg
 
# function finally_boxes = merge_boxes(boxes, nms_threshold, method)
def merge_boxes(boxes, nms_threshold, method):

    if boxes.size==0:
        return np.empty((0,3))
    boxes_x1 = boxes[:,0]
    boxes_y1 = boxes[:,1]
    boxes_x2 = boxes[:,2]
    boxes_y2 = boxes[:,3]
    boxes_score = boxes[:,4]
    boxes_area = (boxes_x2-boxes_x1+1) * (boxes_y2-boxes_y1+1)
    Index = np.arange(boxes.shape[0])
    finally_boxes = np.zeros_like(boxes)

    counter = 0
    while Index.size>0:
        cur_index = Index[-1]
        other_index = Index[0:-1]
        
        overlap_x1 = np.maximum(boxes_x1[cur_index], boxes_x1[other_index])     
        overlap_y1 = np.maximum(boxes_y1[cur_index], boxes_y1[other_index])
        overlap_x2 = np.minimum(boxes_x2[cur_index], boxes_x2[other_index])
        overlap_y2 = np.minimum(boxes_y2[cur_index], boxes_y2[other_index])
        overlap_w  = np.maximum(0.0, overlap_x2-overlap_x1+1)
        overlap_h  = np.maximum(0.0, overlap_y2-overlap_y1+1)
        overlap_area = overlap_w * overlap_h
        if method is 'Min':
            overlap_ratio = overlap_area / np.minimum(boxes_area[cur_index], boxes_area[other_index])
        else:
            overlap_ratio = overlap_area / (boxes_area[cur_index] + boxes_area[other_index] - overlap_area)

        overlap_index = np.hstack([Index[np.where(overlap_ratio >= nms_threshold)] , cur_index])

        if overlap_index.shape[0] == 1:
        	Index = Index[np.where(overlap_ratio < nms_threshold)]
        	finally_boxes[counter] = boxes[cur_index]
        	counter += 1
        	continue
        else:
        	Index = np.hstack([Index[np.where(overlap_ratio < nms_threshold)], cur_index])

        	boxes[cur_index][0] = np.min(boxes_x1[overlap_index])
        	boxes[cur_index][1] = np.min(boxes_y1[overlap_index])
        	boxes[cur_index][2] = np.max(boxes_x2[overlap_index])
        	boxes[cur_index][3] = np.max(boxes_y2[overlap_index])
        	boxes[cur_index][4] = np.mean(boxes_score[overlap_index])

    finally_boxes = finally_boxes[0:counter]
    return finally_boxes

# function pick = nms(boxes,threshold,type)
def nms(boxes, threshold, method):
    if boxes.size==0:
        return np.empty((0,3))
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    s = boxes[:,4]
    area = (x2-x1+1) * (y2-y1+1)
    I = np.argsort(s)
    pick = np.zeros_like(s, dtype=np.int16)
    counter = 0
    while I.size>0:
        i = I[-1]
        pick[counter] = i
        counter += 1
        idx = I[0:-1]
        xx1 = np.maximum(x1[i], x1[idx])
        yy1 = np.maximum(y1[i], y1[idx])
        xx2 = np.minimum(x2[i], x2[idx])
        yy2 = np.minimum(y2[i], y2[idx])
        w = np.maximum(0.0, xx2-xx1+1)
        h = np.maximum(0.0, yy2-yy1+1)
        inter = w * h
        if method is 'Min':
            o = inter / np.minimum(area[i], area[idx])
        else:
            o = inter / (area[i] + area[idx] - inter)
        I = I[np.where(o<=threshold)]
    pick = pick[0:counter]
    return pick


# function finally_boxes = merge_boxes(boxes, nms_threshold, method)

# non_max_suppress

def NMS(boxes, nms_threshold, method):

    if boxes.size==0:
        return np.empty((0,3))
    boxes_x1 = boxes[:,0]
    boxes_y1 = boxes[:,1]
    boxes_x2 = boxes[:,2]
    boxes_y2 = boxes[:,3]
    boxes_score = boxes[:,4]
    boxes_area = (boxes_x2-boxes_x1+1) * (boxes_y2-boxes_y1+1)
    Index = np.argsort(boxes_score)
    finally_boxes = np.zeros_like(boxes)

    counter = 0
    while Index.size>0:
        cur_index = Index[-1]
        other_index = Index[0:-1]
        
        overlap_x1 = np.maximum(boxes_x1[cur_index], boxes_x1[other_index])     
        overlap_y1 = np.maximum(boxes_y1[cur_index], boxes_y1[other_index])
        overlap_x2 = np.minimum(boxes_x2[cur_index], boxes_x2[other_index])
        overlap_y2 = np.minimum(boxes_y2[cur_index], boxes_y2[other_index])
        overlap_w  = np.maximum(0.0, overlap_x2-overlap_x1+1)
        overlap_h  = np.maximum(0.0, overlap_y2-overlap_y1+1)
        overlap_area = overlap_w * overlap_h
        if method is 'Min':
            overlap_ratio = overlap_area / np.minimum(boxes_area[cur_index], boxes_area[other_index])
        else:
            overlap_ratio = overlap_area / (boxes_area[cur_index] + boxes_area[other_index] - overlap_area)
        Index = Index[np.where(overlap_ratio < nms_threshold)]
        finally_boxes[counter] = boxes[cur_index]
        counter += 1
    finally_boxes = finally_boxes[0:counter]
    return finally_boxes


# function [dy edy dx edx y ey x ex tmpw tmph] = pad(total_boxes,w,h)
def pad(total_boxes, w, h):
    # compute the padding coordinates (pad the bounding boxes to square)

    # the width and height of rectangle
    tmpw = (total_boxes[:,2]-total_boxes[:,0]+1).astype(np.int16)
    tmph = (total_boxes[:,3]-total_boxes[:,1]+1).astype(np.int16)

    # the num of rectangle
    numbox = total_boxes.shape[0]

    dx = np.ones((numbox), dtype=np.int16)
    dy = np.ones((numbox), dtype=np.int16)
    edx = tmpw.copy().astype(np.int16)
    edy = tmph.copy().astype(np.int16)

    # (x,y) (ex,ey) the 2 points of rectangle
    x = total_boxes[:,0].copy().astype(np.int16)
    y = total_boxes[:,1].copy().astype(np.int16)
    ex = total_boxes[:,2].copy().astype(np.int16)
    ey = total_boxes[:,3].copy().astype(np.int16)

    # search (x,y) (ex,ey) shich Beyond the boundary
    tmp = np.where(ex>w)
    edx.flat[tmp] = np.expand_dims(-ex[tmp]+w+tmpw[tmp],1)
    ex[tmp] = w
    
    tmp = np.where(ey>h)
    edy.flat[tmp] = np.expand_dims(-ey[tmp]+h+tmph[tmp],1)
    ey[tmp] = h

    tmp = np.where(x<1)
    dx.flat[tmp] = np.expand_dims(2-x[tmp],1)
    x[tmp] = 1

    tmp = np.where(y<1)
    dy.flat[tmp] = np.expand_dims(2-y[tmp],1)
    y[tmp] = 1
    
    return dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph

# function [bboxA] = rerec(bboxA)
def rerec(bboxA):
    # convert bboxA to square
    h = bboxA[:,3]-bboxA[:,1]
    w = bboxA[:,2]-bboxA[:,0]
    l = np.maximum(w, h)
    bboxA[:,0] = bboxA[:,0]+w*0.5-l*0.5
    bboxA[:,1] = bboxA[:,1]+h*0.5-l*0.5
    bboxA[:,2:4] = bboxA[:,0:2] + np.transpose(np.tile(l,(2,1)))
    return bboxA

def imresample(img, sz):
    im_data = cv2.resize(img, (sz[1], sz[0]), interpolation=cv2.INTER_AREA) #@UndefinedVariable
    return im_data

###################################################################################################
#######                                     mtcnn                                           #######

def get_center_result(img_h, img_w, total_boxes, points):
    if len(total_boxes) == 1:
        return total_boxes,points

    center_dis_h = abs((total_boxes[:,0] + total_boxes[:,2] - img_h) / 2)
    center_dis_w = abs((total_boxes[:,1] + total_boxes[:,3] - img_w) / 2)
    center_dis = center_dis_h + center_dis_w
    center_index = np.array([np.argsort(center_dis)[0]])

    total_boxes = total_boxes[center_index,:]
    points = points[:,center_index]  

    return total_boxes,points

def get_max_areas_result(img_h, img_w, total_boxes, points):
    if len(total_boxes) == 1:
        return total_boxes,points

    boxes_h = abs(total_boxes[:,2] - total_boxes[:,0])
    boxes_w = abs(total_boxes[:,3] - total_boxes[:,1])

    boxes_areas = boxes_h * boxes_w
    max_index = np.array([np.argsort(boxes_areas)[-1]])
    total_boxes = total_boxes[max_index,:]
    points = points[:,max_index]  

    return total_boxes,points


def onet_processing(total_boxes, points):
    total_boxes[:,0] = max(total_boxes[:,0], 0)
    total_boxes[:,1] = max(total_boxes[:,1], 0)
    print (total_boxes[:,0])
    return total_boxes, points

def expand_face(total_boxes, img_h, img_w, margin):
    total_boxes[:,0] = np.maximum(0.0, total_boxes[:,0] - margin/ 2)
    total_boxes[:,1] = np.maximum(0.0, total_boxes[:,1] - margin/ 2)
    total_boxes[:,2] = np.minimum(img_w, total_boxes[:,2] + margin/ 2)
    total_boxes[:,3] = np.minimum(img_h, total_boxes[:,3] + margin/ 2)
    return total_boxes



def onet_detect(tempimg, total_boxes, onet, onet_threshold):
    tempimg = (tempimg-127.5)*0.0078125
    tempimg1 = np.transpose(tempimg, (3,1,0,2)).astype(np.float32)
    out = onet(tempimg1)
    out0 = np.transpose(out[0]).astype(np.float32)
    out1 = np.transpose(out[1]).astype(np.float32)
    out2 = np.transpose(out[2]).astype(np.float32)
    score = out2[1,:]
    points = out1
    ipass = np.where(score>onet_threshold)
    points = points[:,ipass[0]]
    total_boxes = np.hstack([total_boxes[ipass[0],0:4].copy(), np.expand_dims(score[ipass].copy(),1)]).astype(np.float32)
    mv = out0[:,ipass[0]]

    w = total_boxes[:,2]-total_boxes[:,0]+1
    h = total_boxes[:,3]-total_boxes[:,1]+1
    points[0:5,:] = np.tile(w,(5, 1))*points[0:5,:] + np.tile(total_boxes[:,0],(5, 1))-1
    points[5:10,:] = np.tile(h,(5, 1))*points[5:10,:] + np.tile(total_boxes[:,1],(5, 1))-1
    if total_boxes.shape[0]>0:
        total_boxes = bbreg(total_boxes.copy(), np.transpose(mv))
        pick = nms(total_boxes.copy(), 0.7, 'Min')
        total_boxes = total_boxes[pick,:]
        points = points[:,pick]    

    return total_boxes, points                
    # return onet_processing(total_boxes, points)


def make_data(img, total_boxes, img_w, img_h, scale_size, save_path = ""):

    numbox = total_boxes.shape[0]


    total_boxes = np.fix(total_boxes).astype(np.int16)

    dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph = pad(total_boxes.copy(), img_w, img_h)
    tempimg = np.zeros((scale_size, scale_size, 3, numbox)).astype(np.float32)
    for k in range(0,numbox):
        ###########################
        if save_path != "":
            # save_fig(img[y[k]-1:ey[k],x[k]-1:ex[k],:], save_path, k)
            cv2.rectangle(img, (int(x[k]-1), int(y[k]-1)), (int(ex[k]), int(ey[k])), (0, 255, 0))
            save_fig(img, save_path, k)
        ###########################
        tmp = np.zeros((int(tmph[k]),int(tmpw[k]),3))
        tmp[dy[k]-1:edy[k],dx[k]-1:edx[k],:] = img[y[k]-1:ey[k],x[k]-1:ex[k],:]
        if (tmp.shape[0]>0 and tmp.shape[1]>0) or (tmp.shape[0]==0 and tmp.shape[1]==0):
            tempimg[:,:,:,k] = imresample(tmp, (scale_size, scale_size))

    return tempimg



def rnet_detect(tempimg, total_boxes, rnet, rnet_threshold):
    tempimg = (tempimg-127.5)*0.0078125
    tempimg1 = np.transpose(tempimg, (3,1,0,2)).astype(np.float32)

    out = rnet(tempimg1)
    out0 = np.transpose(out[0])
    out1 = np.transpose(out[1])
    score = out1[1,:]

    ipass = np.where(score>rnet_threshold)

    total_boxes = np.hstack([total_boxes[ipass[0],0:4].copy(), np.expand_dims(score[ipass].copy(),1)]).astype(np.float32)

    mv = out0[:,ipass[0]]
    if total_boxes.shape[0]>0:
        pick = nms(total_boxes, 0.7, 'Union')
        total_boxes = total_boxes[pick,:]
        total_boxes = bbreg(total_boxes.copy(), np.transpose(mv[:,pick]))
        total_boxes = rerec(total_boxes.copy())

    return total_boxes

def pnet_processing(total_boxes):

    total_boxes = NMS(total_boxes.copy(), 0.7, 'Min')

    regw = total_boxes[:,2]-total_boxes[:,0]
    regh = total_boxes[:,3]-total_boxes[:,1]
    qq1 = total_boxes[:,0]+total_boxes[:,5]*regw
    qq2 = total_boxes[:,1]+total_boxes[:,6]*regh
    qq3 = total_boxes[:,2]+total_boxes[:,7]*regw
    qq4 = total_boxes[:,3]+total_boxes[:,8]*regh
    total_boxes = np.transpose(np.vstack([qq1, qq2, qq3, qq4, total_boxes[:,4]])).astype(np.float32)
    total_boxes = rerec(total_boxes.copy())

    return total_boxes

def pnet_detect(img, img_h, img_w, pnet, scales, pnet_threshold, save_map_path = "", boxes_method = 'Merge',debug =0):
    total_boxes=np.empty((0,9)).astype(np.float32)
    # first stage
    for j in range(len(scales)):
        scale=scales[j]
        hs=int(np.ceil(img_h*scale))
        ws=int(np.ceil(img_w*scale))
        if min(hs,ws) < 12:
            # print (hs, ws)
            continue
        im_data = imresample(img, (hs, ws))
        im_data = (im_data-127.5)*0.0078125
        img_x = np.expand_dims(im_data, 0)
        img_y = np.transpose(img_x, (0,2,1,3))

        out = pnet(img_y)
        # get 4 coordinate
        out0 = np.transpose(out[0], (0,2,1,3)).astype(np.float32)
        # get class scores
        out1 = np.transpose(out[1], (0,2,1,3)).astype(np.float32)

        #save_feature_map
        if save_map_path != "":
            save_fig(out1[0,:,:,1],save_map_path,j)
            continue

        boxes, _ = generateBoundingBox(out1[0,:,:,1].copy(), out0[0,:,:,:].copy(), scale, pnet_threshold, img_h, img_w, debug =debug)
        # inter-scale nms
        if boxes_method is 'Merge':
            pick = merge_boxes(boxes, 0.7, 'Min')
            if boxes.size>0 and pick.size>0:
                total_boxes = np.append(total_boxes, pick, axis=0).astype(np.float32)
        else:
            pick = NMS(boxes.copy(), 0.5, 'Union')
            if boxes.size>0 and pick.size>0:
                total_boxes = np.append(total_boxes, pick, axis=0).astype(np.float32)

    return total_boxes

def save_fig(self, data, save_path, save_name):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not type(save_name) is types.StringType:
        save_name = str(save_name)
    jpg_name = save_name +".jpg"
    cc_save_name = os.path.join(save_path, jpg_name)
    plt.figure(1)
    plt.imshow(data)
    #plt.show()
    plt.savefig(cc_save_name)


def get_scales(img_h, img_w, minsize, maxsize, factor):
    factor_count=0
    if maxsize > minsize:
        minl=np.amin([img_h, img_w, maxsize]) #get min axis
    else:
        minl=np.amin([img_h, img_w]) #get min axis
    m=12.0/minsize
    minl=minl*m

    # creat scale pyramid
    scales=[]
    while minl>=12:
        scales += [m*np.power(factor, factor_count)]
        minl = minl*factor
        factor_count += 1
    return scales

def get_2points_dis(x1, y1, x2, y2):
    dis_x = abs(x1 - x2)
    dis_y = abs(y1 - y2)
    dis   = math.sqrt(math.pow(dis_x, 2) + math.pow(dis_y, 2))
    return dis

# zxzhao 20180315
def get_final_result(rgb_boxes, gray_boxes, rgb_points, nms_thread):
    # This function is a post processing for function detect_rgb_gray_face
    rgb_x1 = rgb_boxes[:,0]
    rgb_y1 = rgb_boxes[:,1]
    rgb_x2 = rgb_boxes[:,2]
    rgb_y2 = rgb_boxes[:,3]
    rgb_area = (rgb_x2-rgb_x1+1) * (rgb_y2-rgb_y1+1)

    gray_x1 = gray_boxes[:,0]
    gray_y1 = gray_boxes[:,1]
    gray_x2 = gray_boxes[:,2]
    gray_y2 = gray_boxes[:,3]
    gray_area = (gray_x2-gray_x1+1) * (gray_y2-gray_y1+1)

    rgb_retain_indexes = []
    rgb_rm_indexes = []
    for rgb_index in range(0, rgb_boxes.shape[0]):
        for gray_index in range(0, gray_boxes.shape[0]):
            overlap_x1 = np.maximum(rgb_x1[rgb_index], gray_x1[gray_index])
            overlap_y1 = np.maximum(rgb_y1[rgb_index], gray_y1[gray_index])
            overlap_x2 = np.minimum(rgb_x2[rgb_index], gray_x2[gray_index])
            overlap_y2 = np.minimum(rgb_y2[rgb_index], gray_y2[gray_index])
            overlap_w = np.maximum(0.0, overlap_x2 - overlap_x1 + 1)
            overlap_h = np.maximum(0.0, overlap_y2 - overlap_y1 + 1)
            overlap_area = overlap_w * overlap_h
            overlap_ratio = overlap_area / (rgb_area[rgb_index] + gray_area[gray_index] - overlap_area)
            if overlap_ratio > nms_thread:
                rgb_retain_indexes.append(rgb_index)
        if not rgb_index in rgb_retain_indexes:
            rgb_rm_indexes.append(rgb_index)
    final_boxes = np.delete(rgb_boxes, rgb_rm_indexes, axis=0)
    final_points = np.delete(rgb_points, rgb_rm_indexes, axis=1)
    return final_boxes, final_points

#######                                     mtcnn                                           #######
###################################################################################################

def is_live_face(face_img, is_RGB = False):
    # This function is used to filter black and white faces printed on A4 paper.
    # The input picture(face_img) must be photographed with a color camera
    # By zxzhao 2018.07.30 18:40
    if not type(face_img) is np.ndarray:
        print ('not type(face_img) is np.ndarray')
        return False
    if len(face_img.shape) != 3:
        print ('len(face_img.shape) != 3')
        return False
    if face_img.shape[2] != 3:
        print ('face_img.shape[2] != 3')
        return False

    resize_len = 10
    img = cv2.resize(face_img,(resize_len,resize_len))
    if is_RGB:
        R = cv2.split(img)[0]
        G = cv2.split(img)[1]
        B = cv2.split(img)[2]
        img = cv2.merge([B,G,R])

    # convert color space from rgb to ycbcr
    imgYcc = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)                 
    # define variables for skin rules
    Wcb = 46.97
    Wcr = 38.76

    WHCb = 14
    WHCr = 10
    WLCb = 23
    WLCr = 20

    Ymin = 16
    Ymax = 235

    Kl = 125
    Kh = 188

    WCb = 0
    WCr = 0

    CbCenter = 0
    CrCenter = 0
    ################################################################################
    skin_count = 0
    gray_count = 0
    for r in range(resize_len):
        for c in range(resize_len):
            ########################################################################
            # color space transformation
            # get values from ycbcr color space     
            Y, Cr, Cb = imgYcc[r,c]

            if Y < Kl:
                WCr = WLCr + (Y - Ymin) * (Wcr - WLCr) / (Kl - Ymin)
                WCb = WLCb + (Y - Ymin) * (Wcb - WLCb) / (Kl - Ymin)

                CrCenter = 154 - (Kl - Y) * (154 - 144) / (Kl - Ymin)
                CbCenter = 108 + (Kl - Y) * (118 - 108) / (Kl - Ymin)            

            elif Y > Kh:
                WCr = WHCr + (Y - Ymax) * (Wcr - WHCr) / (Ymax - Kh)
                WCb = WHCb + (Y - Ymax) * (Wcb - WHCb) / (Ymax - Kh)

                CrCenter = 154 + (Y - Kh) * (154 - 132) / (Ymax - Kh)
                CbCenter = 108 + (Y - Kh) * (118 - 108) / (Ymax - Kh) 

            if Y < Kl or Y > Kh:
                Cr = (Cr - CrCenter) * Wcr / WCr + 154            
                Cb = (Cb - CbCenter) * Wcb / WCb + 108
            ########################################################################
#             if Cb > 77 and Cb < 127 and Cr > 133 and Cr < 173:
            if Cb > 50 and Cb < 180 and Cr > 130 and Cr < 200:
                skin_count += 1
            # print ('Cb:'+ str(int(Cb)),'Cr:'+ str(int(Cr)))
            # with open("./CbCr.txt","a") as f:
            #     f.write(str(int(Cb))+","+str(int(Cr))+"\n")

#             if Cb > 130 and Cb < 140 and Cr > 120 and Cr < 130:
#                 gray_count += 1
    print (skin_count/float(resize_len*resize_len))
#     print (gray_count/float(resize_len*resize_len))
    if (skin_count/float(resize_len*resize_len) >= 0.7):
               # # # save face image
        #img_name = str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')) + '.jpg'
        #if is_RGB:
        #    img_name = 'RGB_' + img_name
        #cv2.imwrite('../task_mtcnn/mtcnn/test_pics/' + img_name, face_img)
        return True
    else:
        return False
        
def get_cropped_face(img_RGB, bounding_boxes, margin=10):
    minsize = 100 # minimum size of face
    face_image=img_RGB
    image_shape = face_image.shape
    crop_faces = []
    dis = -1;
    for face_position in bounding_boxes:
        face_position = face_position.astype(int)

        x0 = max(0, face_position[0])
        y0 = max(0, face_position[1])
        # x0 = face_position[0]
        # y0 = face_position[1]
        x1 = face_position[2]
        y1 = face_position[3]
        x10 = abs(x1-x0)
        y10 = abs(y1-y0)
        if x10 < minsize or y10 < minsize:
            continue
        sub_len = abs(x10 - y10)
        '''
        if x10 > y10:
            y0 = y0 - sub_len/2 if (y0-sub_len/2) > 0 else 0
            y1 = y1 + sub_len/2 if (y1+sub_len/2) < image_shape[0] else image_shape[0]
        else:
            x0 = x0 - sub_len/2 if (x0-sub_len/2) > 0 else 0
            x1 = x1 + sub_len/2 if (x1+sub_len/2) < image_shape[1] else image_shape[1]
        '''

        margin_x0 = x0 - margin//2 if x0 - margin//2 > 0 else 0
        margin_y0 = y0 - margin//2 if y0 - margin//2 > 0 else 0
        margin_x1 = x1 + margin//2 if x1 + margin//2 < image_shape[1] else image_shape[1]
        margin_y1 = y1 + margin//2 if y1 + margin//2 < image_shape[0] else image_shape[0]
        crop = face_image[margin_y0:margin_y1, margin_x0:margin_x1]
        center = ((margin_x1+margin_x0)//2, (margin_y1+margin_y0)//2)
        size = (margin_y1-margin_y0)*(margin_x1-margin_x0)
        tmp = (center[0] - image_shape[0])**2 + (center[1] - image_shape[1])**2
        if tmp > dis:
            dis = tmp
            if crop_faces:
                crop_faces.pop()
            crop_faces.append(crop)
    return crop_faces
