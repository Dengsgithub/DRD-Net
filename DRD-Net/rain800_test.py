# -*- coding: utf-8 -*-
import argparse
import numpy as np
from pathlib import Path
import cv2
import os
from model import get_model
from skimage.measure import compare_psnr



def get_args():
    parser = argparse.ArgumentParser(description="Test trained model",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--image_dir_rain", type=str, default='dataset/rain800/test/norain',
                        help="rain image dir")
    parser.add_argument("--image_dir_gt", type=str, default='dataset/rain800/test/rain',
                        help="gt image dir")
    parser.add_argument("--model", type=str, default="the_end",
                        help="model architecture ")
    parser.add_argument("--weight_file", type=str, default='./weights/rain800/weights.107-201.542-26.45626.hdf5',
                        help="trained weight file")
    parser.add_argument("--If_n", type=bool, default=False,
                        help="If normalizing the image")
    parser.add_argument("--output_dir", type=str, default='./result',
                        help="if set, save resulting images otherwise show result using imshow")
    args = parser.parse_args()
    return args


def get_image(image):
    image = np.clip(image, 0, 255)
    return np.uint8(image)


def main():
    args = get_args()
    image_dir_rain = args.image_dir_rain
    image_dir_gt = args.image_dir_gt
    weight_file = args.weight_file
    if_n = args.If_n
    model = get_model(args.model)
    model.load_weights(weight_file)
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = list(Path(image_dir_rain).glob("*.*"))
    psnr_all = []
    for image_path_rain in image_paths:
        name = str(image_path_rain).split("\\")[-1]
        image_rain = cv2.imread(str(image_path_rain))
        image_gt = cv2.imread(str(os.path.join(image_dir_gt, name)))
        if if_n:
            image_rain = image_rain/255.0
        noise_image = image_rain
        pred = model.predict(np.expand_dims(noise_image, 0))
        if if_n:
            denoised_image = get_image(pred[1][0]*[255])
        else:
            denoised_image = get_image(pred[1][0])
        psnr_all.append(compare_psnr(denoised_image, image_gt))


        if args.output_dir:
            cv2.imwrite(str(output_dir.joinpath(name)), denoised_image)

    print(np.mean(psnr_all))

if __name__ == '__main__':
    main()
