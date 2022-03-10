import numpy as np
import cv2

class BoundingBoxes:
    def __init__(self, boxesList):
        assert(len(boxesList) % 5 == 0)
        self.boxes = []
        for i in range(int(len(boxesList) / 5)):
            self.boxes.append(BoundingBox(boxesList[i*5:(i+1)*5]))
        self.num = len(self.boxes)

    def crop(self, img):
        cropped_imgs = []
        for box in self.boxes:
            cropped_imgs.append(img[box.y1:box.y2, box.x1:box.x2])
        return cropped_imgs

    def draw_boundingboxes(self, img):
        for box in self.boxes:
            cv2.rectangle(img, (box.x1, box.y1), (box.x2, box.y2), (255,0,0), 2)
            cv2.putText(img, f'rc_car: {box.score}%', (box.x1, box.y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        return img
    
    def bytetrack_input(self):
        if self.num == 0:
            return [], [], []
        detects = []
        scores = []
        classes = []
        for i, box in enumerate(self.boxes):
            detects.append([box.x1, box.y1, box.x2, box.y2])  
            scores.append([float(box.score)])
            classes.append([0])

        return detects, scores, classes
    
    def __str__(self):
        s = f'{self.num} bounding box{"es" if self.num != 1 else ""}\n'
        for box in self.boxes:
            s += f'[({box.x1}, {box.y1}), ({box.x2}, {box.y2})] score: {box.score} '
        return s

class BoundingBox:
    def __init__(self, box):
        assert(len(box) == 5)
        self.x1 = box[0]
        self.y1 = box[1]
        self.x2 = box[2]
        self.y2 = box[3]
        self.score = box[4]
