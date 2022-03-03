class BoundingBoxes:
    def __init__(self, boxesList):
        assert(len(boxesList) % 4 == 0)
        self.boxes = []
        for i in range(int(len(boxesList) / 4)):
            self.boxes.append(BoundingBox(boxesList[i*4:(i+1)*4]))
        self.num = len(self.boxes)

    def crop(self, img):
        cropped_imgs = []
        for box in self.boxes:
            cropped_imgs.append(img[box.y1:box.y2, box.x1:box.x2])
        return cropped_imgs

    def __str__(self):
        s = f'{self.num} bounding box{"es" if self.num > 1 else ""}\n'
        for box in self.boxes:
            s += f'[({box.x1}, {box.y1}), ({box.x2}, {box.y2})] '
        return s

class BoundingBox:
    def __init__(self, box):
        assert(len(box) == 4)
        self.x1 = box[0]
        self.y1 = box[1]
        self.x2 = box[2]
        self.y2 = box[3]
