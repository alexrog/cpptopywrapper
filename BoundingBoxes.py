import numpy as np

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
    
    def bytetrack_input(self):
        if self.num == 0:
            return np.array([]), np.array([]), np.array([])
        detects = np.zeros(shape=(self.num,4))
        scores = np.zeros(shape=(self.num,1))
        for i, box in enumerate(self.boxes):
            detects[i] = np.array([box.x1, box.y1, box.x2, box.y2])
            scores[i] = np.array([box.score])
        
        classes = np.zeros(shape=(self.num,1))

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
