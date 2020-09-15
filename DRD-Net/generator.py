from pathlib import Path
import random
import numpy as np
import cv2
from keras.utils import Sequence


class NoisyImageGenerator(Sequence):
    def __init__(self, image_dir_zao, image_dir_yuan, if_n=False, batch_size=32, image_size=64):
        self.image_zao_paths = list(Path(image_dir_zao).glob("*.*"))
        self.image_yuan_paths = list(Path(image_dir_yuan).glob("*.*"))
        self.image_num = len(self.image_zao_paths)
        self.if_n = if_n
        self.batch_size = batch_size
        self.image_size = image_size

    def __len__(self):
        return self.image_num // self.batch_size

    def __getitem__(self, idx):
        batch_size = self.batch_size
        image_size = self.image_size
        x = np.zeros((batch_size, image_size, image_size, 3), dtype=np.float)
        # rain = np.zeros((batch_size, image_size, image_size, 3), dtype=np.float)
        y = np.zeros((batch_size, image_size, image_size, 3), dtype=np.float)
        sample_id = 0

        while True:
            aa = random.randint(0, self.image_num - 1)
            image_zao_path = self.image_zao_paths[aa]
            image_yuan_path = self.image_yuan_paths[aa]
            image_zao = cv2.imread(str(image_zao_path))
            image_yuan = cv2.imread(str(image_yuan_path))
            if self.if_n:
                image_zao = image_zao / 255.0
                image_yuan = image_yuan / 255.0

            h, w, _ = image_zao.shape

            if h >= image_size and w >= image_size:
                h, w, _ = image_zao.shape
                i = np.random.randint(h - image_size + 1)
                j = np.random.randint(w - image_size + 1)
                zao_patch = image_zao[i:i + image_size, j:j + image_size]
                yuan_patch = image_yuan[i:i + image_size, j:j + image_size]
                # rain_patch = zao_patch - yuan_patch
                x[sample_id] = zao_patch.astype(np.float)
                y[sample_id] = yuan_patch.astype(np.float)
                # rain[sample_id] = rain_patch.astype(np.uint8)

                sample_id += 1

                if sample_id == batch_size:
                    return [{'Rain_image': x}, {'subtract_1': y, 'add_36': y}]


class ValGenerator(Sequence):
    def __init__(self, image_dir_zao, image_dir_yuan, if_n=False):
        image_zao_paths = list(Path(image_dir_zao).glob("*.*"))
        image_yuan_paths = list(Path(image_dir_yuan).glob("*.*"))
        self.image_num = len(image_zao_paths)
        self.if_n = if_n
        self.data = []

        for i_image in range(self.image_num):
            x = cv2.imread(str(image_zao_paths[i_image]))
            y = cv2.imread(str(image_yuan_paths[i_image]))
            if self.if_n:
                x = x / 255.0
                y = y / 255.0

            self.data.append([{'Rain_image': np.expand_dims(x, axis=0)}, {'subtract_1': np.expand_dims(y, axis=0),
                                                                          'add_36': np.expand_dims(y, axis=0)}])

    def __len__(self):
        return self.image_num

    def __getitem__(self, idx):
        return self.data[idx]
