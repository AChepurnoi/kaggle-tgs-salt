import cv2
import numpy
import torch
import numpy as np
from imgaug import augmenters as iaa
from pycocotools import mask as cocomask
import pandas as pd
from src.config import *
import matplotlib.pyplot as plt


def load_image(path, mask=False, aug=None):
    """
    Load image from a given path and pad it on the sides, so that eash side is divisible by 32 (newtwork requirement)

    if pad = True:
        returns image as numpy.array, tuple with padding in pixels as(x_min_pad, y_min_pad, x_max_pad, y_max_pad)
    else:
        returns image as numpy.array
    """

    img = cv2.imread(str(path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if aug:
        img = aug.augment_images([img])[0]

    height, width, _ = img.shape

    # Padding in needed for UNet models because they need image size to be divisible by 32
    if height % 32 == 0:
        y_min_pad = 0
        y_max_pad = 0
    else:
        y_pad = 32 - height % 32
        y_min_pad = int(y_pad / 2)
        y_max_pad = y_pad - y_min_pad

    if width % 32 == 0:
        x_min_pad = 0
        x_max_pad = 0
    else:
        x_pad = 32 - width % 32
        x_min_pad = int(x_pad / 2)
        x_max_pad = x_pad - x_min_pad
    # plt.subplot(121)
    # plt.imshow(img)
    img = cv2.copyMakeBorder(img, y_min_pad, y_max_pad, x_min_pad, x_max_pad, cv2.BORDER_REFLECT_101)
    # plt.subplot(122)
    # plt.imshow(img)
    # plt.show()
    if mask:
        # Convert mask to 0 and 1 format
        img = img[:, :, 0:1] // 255
        return torch.from_numpy(img).float().permute([2, 0, 1])
    else:
        img = img / 255.0
        return torch.from_numpy(img).float().permute([2, 0, 1])


def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b > prev + 1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths


def save_checkpoint(model, extra, checkpoint, optimizer=None):
    state = {'state_dict': model.state_dict(),
             'extra': extra}
    if optimizer:
        state['optimizer'] = optimizer.state_dict()

    torch.save(state, CHECKPOINT_DIR + checkpoint)

    print('model saved to %s' % (CHECKPOINT_DIR + checkpoint))


def load_checkpoint(model, checkpoint, optimizer=None):
    state = torch.load(CHECKPOINT_DIR + checkpoint)
    # del state['state_dict']['final.weight']
    # del state['state_dict']['final.bias']
    model.load_state_dict(state['state_dict'], strict=False)
    optimizer_state = state.get('optimizer')
    if optimizer and optimizer_state:
        optimizer.load_state_dict(optimizer_state)

    print("Checkpoint loaded: %s " % state['extra'])
    return state['extra']


def get_paddings():
    height, width = IMAGE_PADDED, IMAGE_PADDED
    if height % 32 == 0:
        y_min_pad = 0
        y_max_pad = 0
    else:
        y_pad = 32 - height % 32
        y_min_pad = int(y_pad / 2)
        y_max_pad = y_pad - y_min_pad
    if width % 32 == 0:
        x_min_pad = 0
        x_max_pad = 0
    else:
        x_pad = 32 - width % 32
        x_min_pad = int(x_pad / 2)
        x_max_pad = x_pad - x_min_pad
    return x_max_pad, x_min_pad, y_max_pad, y_min_pad


def build_submission(binary_prediction, test_file_list):
    all_masks = []
    for p_mask in list(binary_prediction):
        p_mask = rle_encoding(p_mask)
        all_masks.append(' '.join(map(str, p_mask)))
    submit = pd.DataFrame([test_file_list, all_masks]).T
    submit.columns = ['id', 'rle_mask']
    return submit


def crop_to_original_size(predictions):
    x_max_pad, x_min_pad, y_max_pad, y_min_pad = get_paddings()
    stacked_predictions = np.vstack(predictions)[:, 0, :, :]
    stacked_predictions = stacked_predictions[:, y_min_pad:IMAGE_TOTAL_SIZE - y_max_pad,
                          x_min_pad:IMAGE_TOTAL_SIZE - x_max_pad]
    return stacked_predictions
