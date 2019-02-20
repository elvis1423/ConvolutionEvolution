import numpy as np
import cv2


class Data:
    def __init__(self):
        print('This is the __init__ func of class Data')

    def _scale_image(self, image, height, width):
        (_, _, C) = image.shape
        scaled = np.zeros((height, width, C))
        return scaled

    def _crop_image_patches_randomly(self, image, count, height, width):
        (_, _, C) = image.shape
        cropped = np.zeros((count, height, width, C))
        return cropped

    def _mirror_images(self, images):
        (count, H, W, C) = images[0].shape
        mirrored = np.zeros((count*2, H, W, C))
        return mirrored

    def data_augmentation(self, image):
        (H, W, _) = image.shape
        scaled = self._scale_image(image, H, W)
        cropped_imgs = self._crop_image_patches_randomly(scaled, 1024, 256, 256)
        augmented_imgs = self._mirror_images(cropped_imgs)
        return augmented_imgs

    def get_test_img_and_label(self):
        img = cv2.imread('../../resource/test.bmp')
        (h, w, c) = img.shape
        label = np.zeros(1000)
        label[534] = 1  # dummy a label
        if h > w:
            factor = 256/w
        else:
            factor = 256/h
        dst = cv2.resize(img, (0, 0), fx=factor, fy=factor)  # make shortest side 256 pixel
        start_h = (dst.shape[0]-244) // 2
        start_w = (dst.shape[0]-244) // 2
        crop = dst[start_h:start_h+244, start_w:start_w+244].copy()
        return np.reshape(crop, (-1, crop.shape[0], crop.shape[1], crop.shape[2])), np.reshape(label, (-1, 1000))
