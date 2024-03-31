import numpy as np

class Tracker():
    def __init__(self, iou_treshold=0.3, update_last_x=-5, update_last_y=5):
        self.last_bboxs = np.empty((0, 5))
        self.counter = 0
        self.iou_treshold = iou_treshold
        self.update_last_x = update_last_x
        self.update_last_y = update_last_y

    def update(self, bboxs):
        for i in range(len(bboxs)):
            for j in range(len(self.last_bboxs)):
                self.last_bboxs[j][0] += self.update_last_x
                self.last_bboxs[j][1] += self.update_last_y
                self.last_bboxs[j][2] += self.update_last_x
                self.last_bboxs[j][3] += self.update_last_y
                iou = self.iou_score(bboxs[i], self.last_bboxs[j])
                if iou > self.iou_treshold:
                    bboxs[i][4] = self.last_bboxs[j][4]
                    break
            if  bboxs[i][4] == -1:
                bboxs[i][4] = self.counter
                self.counter += 1
        temp = self.last_bboxs
        self.last_bboxs = bboxs
        return bboxs[:, 4], temp
        

    def iou_score(self, bbox1, bbox2):
        x11, y11, x12, y12, _ = bbox1
        x21, y21, x22, y22, _ = bbox2
        x1 = max(x11, x21)
        y1 = max(y11, y21)
        x2 = min(x12, x22)
        y2 = min(y12, y22)
        if x2 < x1 or y2 < y1:
            return 0
        inter_area = (x2 - x1) * (y2 - y1)
        bbox1_area = (x12 - x11) * (y12 - y11)
        bbox2_area = (x22 - x21) * (y22 - y21)
        union_area = bbox1_area + bbox2_area - inter_area
        return inter_area / union_area