# 1.运行步骤
- 1)将要处理人脸图片拷贝到Image/001下面
- 2)python rename_jpg.py 给图片重命名，按1至n的序号命名，5位数字，前面补0.
- 3)python detect_test.py 0 删除没有人脸的图片，mtcnn检测不到的图片。
- 4)python rename_jpg.py 再次运行（可不做），使各个图片连续
- 5)python detect_test.py 1 检测人脸，并将结果按序号保存在Labels/001中，具体格式为一个人脸一行,为左上点与右下点的坐标值（整数）
- 6)python create_label.py 将Labels/001的txt文件，转化为Labels/xml_001下的xml文件，这部是必须的，转换成VOC数据格式下以xml方式表示

- 第5步中的检测人脸可以用手工的python main.py来标注，这个标注很耗时，做一个新东西的时候，是无法避免的，生成了一个基本可以模型后，后面的数据可以通过前面的模型来生成导入。

# 2.分test与trainval数据集
- 在原始demo的目录下创建
```
├── Annotations
├── createtest.py
├── ImageSets
├── JPEGImages

```
- 1)将Image/001下的图片拷贝到JPEGImages，将Labels/xml_001的xml文件拷贝到Annotations文件夹下。
- 2)运行python createtest.py 会在ImagesSet/Main下生成test与trainval数据集

# 3.数据转化为lmdb格式
- 1)使用官方给的两个脚本文件进行操作， ./data/VOC0712/create_list.sh 和 ./data/VOC0712/create_data.sh
- 但要改更几个地方，主要就是更为自己数据的路径。

# 4参考文档
- http://www.cnblogs.com/objectDetect/p/5780006.html
- https://github.com/weiliu89/caffe.git

