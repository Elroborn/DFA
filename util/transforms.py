import numpy as np
from PIL import Image
class Readjust_image(object):
    def __init__(self,block):
        """block is equal n^2"""
        self.block = block


    def __call__(self, image):
        width,height = image.size
        assert width == height
        b = int(np.sqrt(self.block))

        new_width = int(np.floor(width/b) *b)
        new_height = int(np.floor(height/b) *b)

        item_width = int(new_width / b)
        item_heigh = int(new_height / b)


        image = image.resize((new_width,new_height)) # 为了取整

        box_list = []
        for i in range(b):
            for j in range(b):
                # (left, upper, right, lower)
                box = (j * item_width, i * item_heigh, (j + 1) * item_width, (i + 1) * item_heigh)
                box_list.append(box)

        target = Image.new('RGB', (new_width, new_height))  # 最终拼接的图像的大小

        image_list = [image.crop(box) for box in box_list]
        np.random.shuffle(image_list) # 打乱顺序

        for i in range(b):
            for j in range(b):
                box = (j * item_width, i * item_heigh, (j + 1) * item_width, (i + 1) * item_heigh)
                target.paste(image_list[i*b + j], box)
        return target.resize((width,height)) # 为了取整