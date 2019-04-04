"""
NMS 首先将每一类的box 排序，从最大概率的box ，判断前面的box 与其的IOU 是否大于某个阈值
将大于阈值的box 部分扔掉，从剩下的box 中继续选出E 并判断E  与 其他的 IOU 并筛选，找到所有被保留的box
"""

import numpy as np

# coding :utf-8

def py_cpu_nms(dets, thresh):
    """
    :param thresh: 阈值
    :return:  筛选下来的 box
    """
    x1 = dets[:, 0]
    y1= dets[:, 1]
    x2= dets[:, 2]
    y2= dets[:, 3]
    scores =dets[:, 4]
    areas = (x2-x1+1)*(y2-y1+1)  # caluate the all area
    order =scores.argsort()[::-1] # uprise depend the score
    keep = []
    while order.size>0:
        i = order[0]
        keep.append(i)
        xx1= np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2= np.minimum(x2[i],x2[order[1:]])
        yy2= np.minimum(y2[i], y2[order[1:]])

        # caluate the IOU of the  box
        w = np.maximum(0.0, xx2- xx1+1)
        h = np.maximum(0.0, yy2-yy1+1)
        inter = w*h
        over = inter/(areas[i]+areas[order[1:]]-inter)

        # keep the inds that have a lower IOU than the thresh

        inds = np.where(over<= thresh)[0]
        indss = inds +1
        order = order[inds+1]  # include the first number so  included

    return keep
boxes = np.array([[100, 100, 150, 168, 0.63],[166, 70, 312, 190, 0.55],[221, 250, 389, 500, 0.79],[12, 190, 300, 399, 0.9],[28, 130, 134, 302, 0.3]])
thresh = 0.1
keep = py_cpu_nms(boxes, thresh)
print(keep)
# scores = boxes[:, 4]
# #print(scores)
# scores = scores.argsort()[::-1]
# print('scores:', scores[1:])
# x1 = boxes[:, 0]
# y1 = boxes[:, 1]
# x2 = boxes[:, 2]
# y2 = boxes[:, 3]
#
# print(x1)
# print(x1[scores[1:]])
# xx1 = np.maximum(x1[scores[0]], x1[scores[1:]])
# yy1 = np.maximum(y1[scores[0]], y1[scores[1:]])
# xx2 = np.maximum(x2[scores[0]], x2[scores[1:]])
# yy2 = np.maximum(y2[scores[0]], y2[scores[1:]])
# print(xx1)
# print(yy1)
# inter = (yy1-xx1+1)*(yy2-xx2+1)
# print(inter)
# areas =(x2-x1+1)*(y2-y1+1)
# iou = inter/(areas[scores[0]]+areas[scores[1:]]-inter)
# inds = np.where(iou <= 0.1)[0]
# print(inds)
# print(iou)
# order = scores[inds+1]  #
# print(order)








































