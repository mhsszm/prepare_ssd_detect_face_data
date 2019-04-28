#coding=utf-8  
  
import os
path = './Images/001/'
count = 1
list_path = os.listdir(path)
list_path.sort()
for file in list_path:
    print file
    strcount = str(count).zfill(5)
    os.rename(os.path.join(path,file),os.path.join(path,strcount+".jpg"))
    count+=1
