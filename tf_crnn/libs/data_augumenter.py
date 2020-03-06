import random

import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa
from tqdm import tqdm
import cv2

from libs.utils import prob


class DataAugumenter:
    def __init__(self, cfg=None):
        self.cfg = cfg

        ia.seed(random.randint(0, 1000))

        self.seq = iaa.Sequential(
            [
                iaa.GaussianBlur(sigma=(0.3, 0.8)),
                iaa.PiecewiseAffine(0.005, 2, 2),
                iaa.Sometimes(0.1, iaa.AdditiveGaussianNoise(scale=0.02 * 255)),
                iaa.Sometimes(0.3, iaa.Scale({'height': (0.8, 1), 'width': (0.95, 1)})),
            ],
            random_order=False
        )

        self.blur = iaa.GaussianBlur(sigma=(0, 2.))

        self.scale = iaa.Scale({"height": (0.85, 1)})

        self.affine = iaa.OneOf([
            iaa.PerspectiveTransform((0.02, 0.03), False),
            iaa.Affine(shear=(-20, 20), cval=255)
        ])

        self.pwa = iaa.PiecewiseAffine((0.008, 0.015), 2, 2, cval=255)
        self.rotate = iaa.Affine(rotate=(-3, 3), cval=255)

        self.pad_x = iaa.Pad(
            percent=(0, 0.06, 0, 0.06),
            pad_mode=["constant"],
            pad_cval=(255),
            keep_size=False
        )

        self.pad_y = iaa.Pad(
            percent=(0.2, 0, 0.2, 0),
            pad_mode=["constant"],
            pad_cval=(255),
            keep_size=False
        )

        self.crop = iaa.OneOf(
            [
                iaa.Crop(
                    percent=((0.05, 0.1), 0, 0, 0),
                    keep_size=False
                ),
                iaa.Crop(
                    percent=(0, (0.05, 0.1), 0, 0),
                    keep_size=False
                ),
                iaa.Crop(
                    percent=(0, 0, (0.05, 0.1), 0),
                    keep_size=False
                ),
                iaa.Crop(
                    percent=(0, 0, 0, (0.05, 0.1)),
                    keep_size=False
                ),
            ]
        )
        self.noise = iaa.AdditiveGaussianNoise(scale=(0.07 * 255))

        self.change_bright = iaa.Multiply((0.5, 1.5), per_channel=0.5)

        self.change_bright_px = iaa.Multiply((0.5, 1.5), per_channel=0.5)

        self.dropout_random = iaa.CoarseDropout((0.01, 0.02), size_percent=1)

        self.contrast_norm = iaa.ContrastNormalization((0.1, 1.5))

        self.elastic = iaa.ElasticTransformation(alpha=(0.8, 1.7), sigma=(0.4, 0.7), cval=255)

    def _resize(self, img, min_scale=2, max_scale=3):
        """
        先缩小再放大图片
        """

        flag = random.choice([
            cv2.INTER_LINEAR,
            cv2.INTER_CUBIC,
            cv2.INTER_AREA,
            cv2.INTER_NEAREST
        ])

        scale = random.uniform(min_scale, max_scale)

        resized_img = cv2.resize(img, None, fx=1 / scale, fy=1 / scale, interpolation=flag)
        resized_img = cv2.resize(resized_img, None, fx=scale, fy=scale, interpolation=flag)
        return resized_img

    def run(self, img):
        """
        Gray opencv image
        :param img:
        :return:
        """
        if not self.cfg.da.enable:
            return img

        if not prob(self.cfg.da.rate):
            return img

        blured = False
        if prob(0.1):
            img = self.blur.augment_images([img])[0]
            blured = True

        if not blured and self.cfg.da.resize.enable:
            if prob(self.cfg.da.resize.rate):
                # img = self.scale.augment_images([img])[0]
                img = self._resize(img, self.cfg.da.resize.min, self.cfg.da.resize.max)

        if prob(0.4):
            if prob(0.5):
                img = self.pad_x.augment_images([img])[0]
                img = self.affine.augment_images([img])[0]
            else:
                img = self.pad_y.augment_images([img])[0]
                img = self.rotate.augment_images([img])[0]

        if self.cfg.da.crop.enable and prob(self.cfg.da.crop.rate):
            img = self.crop.augment_images([img])[0]

        if prob(0.2):
            img = self.pwa.augment_images([img])[0]

        if prob(0.3):
            img = self.noise.augment_images([img])[0]

        c5 = random.random()
        if c5 < 0.2:
            mask = 255 - np.zeros(img.shape, dtype=np.uint8)
            mask = self.dropout_random.augment_images([mask])[0]
            img = img.astype(np.uint16) + (255 - mask)
            img = np.minimum(img, 255).astype(np.uint8)
        elif c5 < 0.3:
            img = self.elastic.augment_images([img])[0]

        if prob(0.1):
            img = self.blur.augment_images([img])[0]

        if prob(0.2):
            img = self.contrast_norm.augment_images([img])[0]

        return img

    def da_img(self, img):
        return self.seq.augment_images([img])[0]


if __name__ == '__main__':
    import glob
    import cv2
    import os

    da = DataAugumenter()
    output_dir = '/Users/cwq/data/da_test/da'
    img_path = '/Users/cwq/code/text_renderer/output/default/00000094.jpg'
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)


    def da_test(func, img, name):
        for i in tqdm(range(100)):
            da_img = func.augment_images([img])[0]
            outdir = os.path.join(output_dir, name)
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            da_path = os.path.join(outdir, '{}.jpg'.format(i))
            cv2.imwrite(da_path, da_img)


    da_test(da.blur, img, 'blur')
    da_test(da.scale, img, 'scale')
    da_test(da.pad_x, img, 'pad_x')
    da_test(da.pad_y, img, 'pad_y')
    da_test(da.affine, img, 'affine')
    da_test(da.rotate, img, 'rotate')
    da_test(da.crop, img, 'crop')
    da_test(da.pwa, img, 'pwa')
    da_test(da.noise, img, 'noise')
    da_test(da.elastic, img, 'elastic')
    da_test(da.contrast_norm, img, 'contrast_norm')
    da_test(da.dropout_random, img, 'dropout')

    for i in tqdm(range(100)):
        da_img = da._resize(img)
        outdir = os.path.join(output_dir, 'resize')
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        da_path = os.path.join(outdir, '{}.jpg'.format(i))
        cv2.imwrite(da_path, da_img)

    for i in tqdm(range(100)):
        da_img = da.run(img)
        outdir = os.path.join(output_dir, 'da_idcard')
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        da_path = os.path.join(outdir, '{}.jpg'.format(i))
        cv2.imwrite(da_path, da_img)
