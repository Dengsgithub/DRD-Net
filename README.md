# DRD-Net
This website shares the codes of the "Detail-recovery Image Deraining via Context Aggregation Networks",CVPR 2020.

Prerequisites:
tensorflow == 1.10.0
keras == 2.2.4
python == 3.6
CUDA ==10.0

For train the Rain200H, please run:
python train.py --image_dir_noise you rain data --image_dir_original you gt data --test_dir_noise you test rain data --test_dir_original you test gt data --If_n True

For train the Rain200L, please run:
python train.py --image_dir_noise you rain data --image_dir_original you gt data --test_dir_noise you test rain data --test_dir_original you test gt data --If_n False

For train the Rain800, please run:
python train.py --image_dir_noise you rain data --image_dir_original you gt data --test_dir_noise you test rain data --test_dir_original you test gt data --If_n False

The dataset "Rain200H" and "Rain200L" you can download here:
https://www.icst.pku.edu.cn/struct/Projects/joint_rain_removal.html
the dataset "Rain800" you can download here:
https://drive.google.com/drive/folders/0Bw2e6Q0nQQvGbi1xV1Yxd09rY2s
