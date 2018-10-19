import torch
from tensorboardX import SummaryWriter
from torch.utils import data

from torch.nn import functional as F
import src.lovasz as L
from src.config import *
from src.data import get_test_dataset, kfold_split, split_train_val, TGSSaltDatasetAug
from src.metrics import *
from src.utils import build_submission, save_checkpoint, crop_to_original_size
# from src.model_dsv_exp import load_model as exp_load_model
from src.model_dsv_exp_v3 import load_model as exp_load_model


class DiceLoss(torch.nn.Module):
    def __init__(self, smooth=0, eps=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.eps = eps

    def forward(self, output, target):
        return 1 - (2 * (output * target).sum() + self.smooth) / (
                output.sum() + target.sum() + self.smooth + self.eps)


class FocalLoss2d(torch.nn.Module):
    def __init__(self, gamma=2, eps=1e-8):
        super(FocalLoss2d, self).__init__()
        self.gamma = gamma
        self.eps = eps

    def forward(self, input, target):
        # print(input.size(), target.size())
        assert input.size() == target.size()

        # input = input.transpose(1,2).transpose(2,3).contiguous().view(-1,1)
        # target = target.transpose(1,2).transpose(2,3).contiguous().view(-1,1)

        p = input.sigmoid()
        p = p.clamp(min=self.eps, max=1. - self.eps)

        pt = p * target + (1. - p) * (1 - target)
        # from Heng, don't apply focal weight to predictions with prob < 0.1
        # pt[pt < 0.1] = 0.

        w = (1. - pt).pow(self.gamma)

        loss = F.binary_cross_entropy_with_logits(input, target, w)

        return loss


def clear_small_masks(binary_prediction):
    mask_sizes = np.sum(binary_prediction, axis=(1, 2))
    binary_prediction[np.where(mask_sizes < MIN_MASK_SIZE)] = np.zeros((101, 101))
    return binary_prediction


def main():
    TRAIN_PATH = os.path.join(DIRECTORY, 'train')
    if N_FOLDS == 1:
        folds = split_train_val()
    else:
        folds = kfold_split()

    board = SummaryWriter()
    loaded_models = [exp_load_model("fold-{}.pth".format(f)) for f in range(min(N_FOLDS, TAKE_FOLDS))]
    accuracy = [loaded_model[3]['lb_acc'] for loaded_model in loaded_models]
    for e in range(TRAINING_EPOCH):
        accuracy = []
        for i, (train_list, val_list) in enumerate(folds[:TAKE_FOLDS]):
            model, optimizer, scheduler, state = loaded_models[i]
            start_epoch, best_lb_acc, rlr_iter = state['epoch'], state['lb_acc'], state.get('iter', 0)

            dataset = TGSSaltDatasetAug(TRAIN_PATH, train_list, aug=True)
            dataset_val = TGSSaltDatasetAug(TRAIN_PATH, val_list)

            train_loss = train_iter(dataset, model, optimizer)
            with torch.no_grad():
                val_loss, lb_acc, _ = eval(dataset_val, model)
            scheduler.step(lb_acc, epoch=e)
            # scheduler.step()
            # scheduler.batch_step()
            accuracy.append(lb_acc)

            # Chekpoints
            if lb_acc > best_lb_acc:
                save_checkpoint(model, optimizer=optimizer, extra={
                    'epoch': start_epoch + e,
                    'lb_acc': lb_acc
                }, checkpoint='fold-%d.pth' % i)
                state['lb_acc'] = lb_acc

            # if e % CYCLES == 0:
            #     save_checkpoint(model, optimizer=optimizer, extra={
            #         'epoch': start_epoch + e,
            #         'lb_acc': lb_acc
            #     }, checkpoint='cycle-%d-%.3f.pth' % (e % CYCLES, lb_acc))
            # el
            if e % 30 == 0 or e == TRAINING_EPOCH - 1:
                save_checkpoint(model, optimizer=optimizer, extra={
                    'epoch': start_epoch + e,
                    'lb_acc': lb_acc
                }, checkpoint='ep%s-%.3f.pth' % (start_epoch + e, lb_acc))
            else:
                save_checkpoint(model, optimizer=optimizer, extra={
                    'epoch': start_epoch + e,
                    'lb_acc': lb_acc
                }, checkpoint='last.pth')

            # Tensorboard
            board.add_scalars('seresnet/losses',
                              {'train_loss': train_loss,
                               'val_loss': val_loss}, e)

            board.add_scalar('seresnet/lb_acc', lb_acc, e)

            log = "Epoch: %d, Fold %d, Train: %.3f, Val: %.3f, LB: %.3f (Best: %.3f)" % (
                start_epoch + e, i, train_loss, val_loss, lb_acc, best_lb_acc)

            print(log)

        print("Mean accuracy %.3f , Variance %.3f" % (np.mean(accuracy), np.var(accuracy)))

    test_dataset, test_file_list = get_test_dataset(DIRECTORY)
    test_predictions = []
    for (model, _, _, _) in loaded_models:
        model.eval()
        with torch.no_grad():
            all_predictions_stacked = test_tta(model, test_dataset)
        test_predictions.append(all_predictions_stacked)

    fold_mean_prediciton = np.mean(test_predictions, axis=0)
    binary_prediction = (fold_mean_prediciton > BIN_THRESHOLD).astype(int)
    # predictions = binary_prediction
    predictions = clear_small_masks(binary_prediction)
    submit = build_submission(predictions, test_file_list)
    submit.to_csv('submitM%.3fV%.3f.csv' % (np.mean(accuracy), np.std(accuracy)), index=False)
    board.close()


def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()


def train_iter(dataset, model, optimizer, scheduler=None):
    model.train()
    model.apply(set_bn_eval)

    train_loss = []
    bce = torch.nn.BCEWithLogitsLoss()
    for image, mask in tqdm.tqdm(data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=N_WORKERS,
                                                 pin_memory=True)):
        image = image.type(torch.float).to(DEVICE)
        y_pred, dsvs, is_empty_pred = model(image)
        y_pred_attn = y_pred * is_empty_pred.view(image.size(0), 1, 1, 1)
        mask_bool = (mask.sum([2, 3]) > 0).float().to(DEVICE)

        loss = compute_loss(mask, y_pred)

        if is_empty_pred is not None:
            loss += CLF_C * bce(is_empty_pred, mask_bool)
            loss += CLF_C * compute_loss(mask, y_pred_attn)

        non_zero_masks = mask.byte().any(2).any(2).flatten()
        if dsvs is not None:
            for dsv in dsvs:
                loss = loss + DSV_C * compute_loss(mask[non_zero_masks, :, :, :], dsv[non_zero_masks, :, :, :])

        optimizer.zero_grad()
        loss.backward()
        # if scheduler is not None:
        #     scheduler.batch_step()

        optimizer.step()
        train_loss.append(loss.item())

    return np.mean(train_loss)


def compute_loss(mask, y_pred):
    bce = torch.nn.BCELoss()
    dice = DiceLoss()
    focal = FocalLoss2d()
    if LOSS_FUNC is 'bce':
        loss = bce(F.sigmoid(y_pred), mask.to(DEVICE))
    elif LOSS_FUNC is 'lovasz':
        loss = L.lovasz_hinge(y_pred, mask.to(DEVICE), per_image=False)
    elif LOSS_FUNC is 'combined':
        loss = BCE_C * bce(F.sigmoid(y_pred), mask.to(DEVICE)) \
               + LOVASZ_C * L.lovasz_hinge(y_pred, mask.to(DEVICE), per_image=False)
    elif LOSS_FUNC is "dice":
        loss = dice(y_pred.sigmoid(), mask.to(DEVICE))
    elif LOSS_FUNC is "other":
        loss = BCE_C * bce(F.sigmoid(y_pred), mask.to(DEVICE))
        loss += DICE_C * dice(y_pred.sigmoid(), mask.to(DEVICE))
        # loss = focal(y_pred, mask.to(DEVICE))
        # elif LOSS_FUNC is "dice":
        #     loss = dice(F.sigmoid(y_pred), mask.to(DEVICE))
    else:
        raise RuntimeError("Unknown loss")
    return loss


def eval(dataset_val, model):
    model.eval()
    val_loss = []
    val_predictions = []
    val_masks = []
    bce = torch.nn.BCEWithLogitsLoss()
    for image, mask in data.DataLoader(dataset_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=N_WORKERS,
                                       pin_memory=True):
        image = image.to(DEVICE)
        y_pred, dsvs, is_empty_pred = model(image)
        y_pred_sig = F.sigmoid(y_pred)

        y_pred_attn = y_pred * is_empty_pred.view(image.size(0), 1, 1, 1)
        mask_bool = (mask.sum([2, 3]) > 0).float().to(DEVICE)

        loss = compute_loss(mask, y_pred)

        if is_empty_pred is not None:
            loss += CLF_C * bce(is_empty_pred, mask_bool)
            loss += CLF_C * compute_loss(mask, y_pred_attn)

        non_zero_masks = mask.byte().any(2).any(2).flatten()

        if dsvs is not None:
            for dsv in dsvs:
                loss = loss + DSV_C * compute_loss(mask[non_zero_masks, :, :, :], dsv[non_zero_masks, :, :, :])

        val_loss.append(loss.item())
        val_predictions.append(y_pred_sig.cpu().detach().numpy())
        val_masks.append(mask)

    val_masks_stacked = crop_to_original_size(val_masks)
    val_predictions_stacked = crop_to_original_size(val_predictions)
    mean_loss = np.mean(val_loss)

    bin_val_predictions_stacked = (val_predictions_stacked > BIN_THRESHOLD).astype(int)
    lb_acc = get_iou_vector(bin_val_predictions_stacked, val_masks_stacked)

    return mean_loss, lb_acc, BIN_THRESHOLD


def test(model, test_dataset):
    all_predictions = []
    for image in tqdm.tqdm(data.DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=N_WORKERS,
                                           pin_memory=True)):
        image = image[0].type(torch.float).to(DEVICE)
        y_pred, _, _ = F.sigmoid(model(image)).cpu().detach().numpy()
        all_predictions.append(y_pred)
    all_predictions_stacked = crop_to_original_size(all_predictions)
    return all_predictions_stacked


def test_tta(model, test_dataset):
    all_predictions = []
    for image in tqdm.tqdm(data.DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=N_WORKERS,
                                           pin_memory=True)):
        # TTA horizontal flip for each image
        image_batch_src = image[0].type(torch.float).to(DEVICE)
        image_batch_aug = torch.flip(image[0], [3]).type(torch.float).to(DEVICE)

        y_pred_src, _, _ = model(image_batch_src)
        y_pred_aug = torch.flip(model(image_batch_aug)[0], [3])

        y_mean = torch.stack([y_pred_src, y_pred_aug]).mean(dim=0)

        y_pred = F.sigmoid(y_mean).cpu().detach().numpy()
        all_predictions.append(y_pred)
    all_predictions_stacked = crop_to_original_size(all_predictions)
    return all_predictions_stacked


if __name__ == '__main__':
    main()
