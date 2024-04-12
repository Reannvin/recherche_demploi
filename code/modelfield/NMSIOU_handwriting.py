import numpy as np

def nms(boxes, scores, iou_thresh):
    # 按照分数从高到低对boxes索引进行排序
    order = scores.argsort()[::-1]
    keep = [] # 用于存储保留的boxes索引
    
    # 循环处理每个box
    while order.size > 0:
        i = order[0] # 获取当前分数最高的box索引
        keep.append(i) # 将当前box索引添加到保留列表中
        
        # 计算当前box与其他剩余boxes的交集坐标
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        
        # 计算交集区域的宽度和高度
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        
        # 计算交集区域的面积
        intersection = w * h
        
        # 计算两个box的并集区域面积
        union = (boxes[i, 2] - boxes[i, 0] + 1) * (boxes[i, 3] - boxes[i, 1] + 1) + \
                (boxes[order[1:], 2] - boxes[order[1:], 0] + 1) * (boxes[order[1:], 3] - boxes[order[1:], 1] + 1) - \
                intersection
        
        # 计算IoU (Intersection over Union)
        iou = intersection / union
        
        # 获取IoU小于阈值的剩余boxes索引
        inds = np.where(iou <= iou_thresh)[0]
        
        # 更新order,移除IoU大于阈值的boxes索引
        order = order[inds + 1]
    
    # 返回保留的boxes索引
    return keep