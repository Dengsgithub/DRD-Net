import argparse
import numpy as np
from pathlib import Path
from keras.callbacks import LearningRateScheduler, ModelCheckpoint,TensorBoard, ReduceLROnPlateau
from keras.optimizers import Adam
from model import get_model, PSNR, L0Loss, UpdateAnnealingParameter
from generator import NoisyImageGenerator, ValGenerator


class Schedule:
    def __init__(self, nb_epochs, initial_lr):
        self.epochs = nb_epochs
        self.initial_lr = initial_lr

    def __call__(self, epoch_idx):
        if epoch_idx < 6:
            return 0.01
        elif epoch_idx < 15:
            return 0.005
        elif epoch_idx < 30:
            return 0.0025
        elif epoch_idx < 45:
            return 0.00125
        elif epoch_idx < 60:
            return 0.001
        elif epoch_idx < 75:
            return 0.0005
        elif epoch_idx < 90:
            return 0.000125
        return 0.00001


def get_args():
    parser = argparse.ArgumentParser(description="train derain model",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--image_dir_noise", type=str, default='./dataset/Rain_H/train/x_in',
                        help="train image dir")
    parser.add_argument("--image_dir_original", type=str, default='./dataset/Rain_H/train/y_out',
                        help="train image dir")
    parser.add_argument("--test_dir_noise", type=str, default='./dataset/Rain100H/rain',
                        help="test image dir")
    parser.add_argument("--test_dir_original", type=str, default='./dataset/Rain100H/norain',
                        help="test image dir")
    parser.add_argument("--If_n", type=bool, default=True,
                        help="If normalizing the image")
    parser.add_argument("--image_size", type=int, default=128,
                        help="training patch size")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="batch size")
    parser.add_argument("--nb_epochs", type=int, default=120,
                        help="number of epochs")
    parser.add_argument("--lr", type=float, default=0.1,
                        help="learning rate")
    parser.add_argument("--steps", type=int, default=1000,
                        help="steps per epoch")
    parser.add_argument("--loss", type=str, default="mse",
                        help="loss; mse', 'mae', or 'l0' is expected")
    parser.add_argument("--output_path", type=str, default="impulse_clean",
                        help="checkpoint dir")
    parser.add_argument("--model", type=str, default="the_end",
                        help="model architecture ('Similarity')")
    args = parser.parse_args()

    return args


def main():
    args = get_args()
    image_dir_noise = args.image_dir_noise
    image_dir_original = args.image_dir_original
    test_dir_noise = args.test_dir_noise
    test_dir_original = args.test_dir_original
    image_size = args.image_size
    batch_size = args.batch_size
    nb_epochs = args.nb_epochs
    if_n = args.If_n
    lr = args.lr
    steps = args.steps
    loss_type = args.loss
    output_path = Path(__file__).resolve().parent.joinpath(args.output_path)
    model = get_model(args.model)
    opt = Adam(lr=lr)
    callbacks = []

    if loss_type == "l0":
        l0 = L0Loss()
        callbacks.append(UpdateAnnealingParameter(l0.gamma, nb_epochs, verbose=1))
        loss_type = l0()

    # model.compile(optimizer=opt, loss=loss_type, metrics=[PSNR])
    model.compile(optimizer=opt, loss={"subtract_1": "mse", "add_36": "mse"},
                  loss_weights={'subtract_1': 0.1, 'add_36': 1}, metrics=[PSNR])
    model.summary()
    generator = NoisyImageGenerator(image_dir_noise, image_dir_original, if_n=if_n, batch_size=batch_size,
                                    image_size=image_size)
    val_generator = ValGenerator(test_dir_noise, test_dir_original, if_n=if_n)
    output_path.mkdir(parents=True, exist_ok=True)
    # callbacks.append(ReduceLROnPlateau(monitor='val_add_35_loss', factor=0.5, patience=5, verbose=1, mode='min',
    #                                    cooldown=0, min_lr=0.000000001))
    callbacks.append(LearningRateScheduler(schedule=Schedule(nb_epochs, lr)))
    callbacks.append(TensorBoard(log_dir='./log', histogram_freq=0, batch_size=batch_size, write_images=True))
    callbacks.append(ModelCheckpoint(str(output_path) + "/weights.{epoch:03d}-{val_add_36_loss:.3f}-{val_add_36_PSNR:.5f}.hdf5",
                                     monitor="val_add_36_PSNR",
                                     verbose=1,
                                     mode="max",
                                     save_best_only=True))

    hist = model.fit_generator(generator=generator,
                               steps_per_epoch=steps,
                               epochs=nb_epochs,
                               validation_data=val_generator,
                               verbose=1,
                               callbacks=callbacks)

    np.savez(str(output_path.joinpath("history.npz")), history=hist.history)


if __name__ == '__main__':
    main()
