import os
import torch

DIRECTORY = 'data'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 8
N_FOLDS = 1
TAKE_FOLDS = 1


# CHECKPOINT_DIR = 'schk/'
CHECKPOINT_DIR = 'v3/'

# CHECKPOINT_DIR = 'tchk/'
# CHECKPOINT_DIR = 'hchk/'

# Last big model saved here
# CHECKPOINT_DIR = 'fchk/'


if not os.path.exists(CHECKPOINT_DIR):
    os.mkdir(CHECKPOINT_DIR)


LOAD_CHECKPOINT = False
DEBUG = False


CYCLES = 40
TRAINING_EPOCH = 300
# SGD
# LEARNING_RATE = 0.000005
LEARNING_RATE = 1e-2

# Adam
# LEARNING_RATE = 0.00005
DROPOUT_RATE = 0.5

L2_REG = 0.0001
LOSS_FUNC = 'bce'
# If Loss func is combined
BCE_C = 0.2
DICE_C = 0.8

LOVASZ_C = 0.9

N_WORKERS = 2

# Deep supervision
DSV_C = 0.1
CLF_C = 0.05

DECREASE_LR_EPOCH = 30

# Submission
BIN_THRESHOLD = 0.5
MIN_MASK_SIZE = 20

IMAGE_TOTAL_SIZE = 128
IMAGE_PADDED = 101

