import glob
import os

import numpy
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold, train_test_split, StratifiedKFold
from torch.utils import data
from PIL import Image
from src.config import DIRECTORY, N_FOLDS, DEBUG, IMAGE_PADDED, IMAGE_TOTAL_SIZE
from src.utils import load_image
from imgaug import augmenters as iaa
import cv2
from albumentations import (
    HorizontalFlip,
    VerticalFlip,
    Resize,
    CenterCrop,
    ToGray,
    RandomCrop,
    OpticalDistortion,
    RandomRotate90,
    RandomSizedCrop,
    Transpose,
    GridDistortion,
    Blur,
    InvertImg,
    GaussNoise,
    OneOf,
    ElasticTransform,
    MedianBlur,
    ShiftScaleRotate,
    Rotate,
    Normalize,
    Crop,
    CLAHE,
    Flip,
    LongestMaxSize,
    RandomScale,
    PadIfNeeded,
    FilterBboxes,
    Compose,
    RandomBrightness,
    RandomContrast,
    convert_bboxes_to_albumentations,
    filter_bboxes_by_visibility,
    denormalize_bbox,
    RandomGamma)


def load_img(path, mask=False):
    img = cv2.imread(str(path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # return img
    if mask:
        return img[:, :, 0:1]
    else:
        return img


class TGSSaltDatasetAug(data.Dataset):
    def __init__(self, root_path, file_list, aug=False, is_test=False):
        self.is_test = is_test
        self.root_path = root_path
        self.file_list = file_list
        if aug:
            self._aug = Compose([
                HorizontalFlip(),
                # OneOf([
                #     ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=15,
                #                      interpolation=cv2.INTER_NEAREST),
                    # RandomSizedCrop(min_max_height=(70, 90),
                    #                 height=IMAGE_PADDED, width=IMAGE_PADDED,
                    #                 interpolation=cv2.INTER_NEAREST)]
                # ], p=1),
                # ElasticTransform(p=0.5, border_mode=cv2.BORDER_REPLICATE,
                # alpha=1, sigma=5, alpha_affine=5, interpolation=cv2.INTER_NEAREST),
                # OneOf([GridDistortion(distort_limit=0.1, border_mode=cv2.BORDER_REPLICATE,
                #                       interpolation=cv2.INTER_NEAREST),
                #        OpticalDistortion(border_mode=cv2.BORDER_REPLICATE, interpolation=cv2.INTER_NEAREST, p=0)], p=1),
                # InvertImg(p=0.5),
                RandomBrightness(p=0.5, limit=0.5),
                RandomContrast(p=0.5, limit=0.5),
                RandomGamma(p=0.5, gamma_limit=(70, 130)),
                CLAHE(p=0.5),
                # OneOf([
                #     Blur(p=1, blur_limit=4),
                #     MedianBlur(p=1, blur_limit=4),
                #     GaussNoise(p=1)
                # ], p=1),
                Normalize(),
                # Resize(IMAGE_PADDED, IMAGE_PADDED),
                PadIfNeeded(min_height=IMAGE_TOTAL_SIZE, min_width=IMAGE_TOTAL_SIZE, border_mode=cv2.BORDER_REPLICATE)
            ])
        else:
            self._aug = Compose([
                Normalize(),
                # Resize(IMAGE_PADDED, IMAGE_PADDED),
                PadIfNeeded(min_height=IMAGE_TOTAL_SIZE, min_width=IMAGE_TOTAL_SIZE, border_mode=cv2.BORDER_REPLICATE)
            ])

    def __len__(self):
        if DEBUG:
            return 2
        else:
            return len(self.file_list)

    def __getitem__(self, index):
        if index not in range(0, len(self.file_list)):
            return self.__getitem__(np.random.randint(0, self.__len__()))

        file_id = self.file_list[index]
        image_folder = os.path.join(self.root_path, "images")
        image_path = os.path.join(image_folder, file_id + ".png")

        mask_folder = os.path.join(self.root_path, "masks")
        mask_path = os.path.join(mask_folder, file_id + ".png")

        image = load_img(image_path)

        if self.is_test:
            data = {"image": image}
            augmented = self._aug(**data)
            image = augmented["image"]
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            image = torch.from_numpy(image).float().unsqueeze(2).permute([2, 0, 1]).repeat([3, 1, 1])
            image = add_depth_channels(image)
            return (image,)
        else:
            mask = load_img(mask_path, mask=True)
            data = {"image": image, "mask": mask}
            augmented = self._aug(**data)
            image, mask = augmented["image"], augmented["mask"]
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            image = torch.from_numpy(image).float().unsqueeze(2).permute([2, 0, 1]).repeat([3, 1, 1])
            image = add_depth_channels(image)
            mask = torch.from_numpy(mask).float().unsqueeze(2).permute([2, 0, 1]) // 255
            return image, mask

def add_depth_channels(image_tensor):
    _, h, w = image_tensor.size()
    for row, const in enumerate(np.linspace(0, 1, h)):
        image_tensor[1, row, :] = const
    image_tensor[2] = image_tensor[0] * image_tensor[1]
    return image_tensor

def get_train_files(directory):
    n_bins = 5
    depths_df = pd.read_csv(os.path.join(directory, 'train.csv')) \
        .assign(masks=lambda x: [np.sum(cv2.imread("data/train/masks/{}.png".format(xx), 0) // 255) for xx in x.id]) \
        .assign(label=lambda x: np.digitize(x.masks, np.linspace(0, x.masks.max(), n_bins)))
    return depths_df


def get_test_dataset(directory):
    test_path = os.path.join(directory, 'test')
    test_file_list = glob.glob(os.path.join(test_path, 'images', '*.png'))
    test_file_list = [f.split('/')[-1].split('.')[0] for f in test_file_list]
    test_dataset = TGSSaltDatasetAug(test_path, test_file_list, is_test=True)
    return test_dataset, test_file_list


def split_train_val():
    train_df = get_train_files(DIRECTORY)
    split = train_test_split(train_df.id.values,
                             train_size=0.92,
                             test_size=0.08,
                             random_state=17,
                             stratify=train_df.label.values)
    return [(split[0], split[1])]


def kfold_split():
    file_list = get_train_files(DIRECTORY)
    kf = StratifiedKFold(n_splits=N_FOLDS, random_state=17)
    folds = []
    for train, test in kf.split(file_list.id.values, file_list.label.values):
        folds.append((file_list.iloc[train].id.values, file_list.iloc[test].id.values))
    return folds
