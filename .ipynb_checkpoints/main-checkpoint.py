import sys
import time
import numpy as np
import yaml
import random
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import wandb

import torch
from torch import optim
from torchmetrics import Dice, JaccardIndex
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

import monai
from monai.networks.utils import one_hot
from monai.losses import DiceCELoss, DiceFocalLoss

from ReadData.create_ID_list_selected_dataset import create_list_ID_training
from utils.data import Data

import pandas as pd
import segmentation_models_pytorch as smp

import models.regionvit_unet.regionvit_unet as regionvit_unet
from models.regionvit_fpn.fpn.model import FPN
from models.swin_unet.config_swin_transformer import get_config
from models.swin_unet.vision_transformer import SwinUnet as ViT_seg
from monai.networks.nets import BasicUNet

########################################################################################################################
# read yaml file for training and evaluation/test configuration

def fParseConfig(sFile):
    # get config file
    with open(sFile, 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)

    return cfg


########################################################################################################################
# pixel accuracy and mean Intersection over Union used as evaluation metrics

def pixel_accuracy(output, mask):
    with torch.no_grad():
        output = torch.argmax(output, dim=1)
        mask = torch.argmax(mask, dim=1)
        correct = torch.eq(output, mask).int()
        accuracy = float(correct.sum()) / float(correct.numel())
    return accuracy


def mIoU(pred_mask, mask, smooth=1e-10, n_classes=2):
    with torch.no_grad():
        pred_mask = torch.argmax(pred_mask, dim=1)
        pred_mask = pred_mask.contiguous().view(-1)
        mask = torch.argmax(mask, dim=1)
        mask = mask.contiguous().view(-1)

        iou_per_class = []
        for clas in range(0, n_classes):  # loop per pixel class
            true_class = pred_mask == clas
            true_label = mask == clas

            if true_label.long().sum().item() == 0:  # no exist label in this loop
                iou_per_class.append(np.nan)
            else:
                intersect = torch.logical_and(true_class, true_label).sum().float().item()
                union = torch.logical_or(true_class, true_label).sum().float().item()

                iou = (intersect + smooth) / (union + smooth)
                iou_per_class.append(iou)
        return np.nanmean(iou_per_class)


########################################################################################################################
# get current learning rate for learning rate scheduler

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


########################################################################################################################
# perform prediction tests for evaluation and test to measure the model performance

def predict(model, test_loader, criterion):
    test_loss = 0.0
    test_mIoU_macro = 0.0
    test_hd95 = 0.0
    test_dice_macro = 0.0
    test_acc = 0.0
    test_sensitivity = 0.0
    test_specificity = 0.0
    test_precision = 0.0
    test_negative_predictive_value = 0.0
    test_loss_list = []
    test_mIoU_macro_list = []
    test_hd95_list = []
    test_dice_macro_list = []
    test_acc_list = []
    test_sensitivity_list = []
    test_specificity_list = []
    test_precision_list = []
    test_negative_predictive_value_list = []

    test_hausdorff = monai.metrics.HausdorffDistanceMetric(include_background=True, reduction="mean", percentile=95.0,
                                                           get_not_nans=False)
    test_dice = Dice(num_classes=2, average='macro').to(device)

    model.eval()  # Optional when not using Model Specific layer
    for images, labels in tqdm(test_loader):
        with torch.no_grad():
            images, labels = images.to(device), labels.to(device)

            output_one_hot = model(images)

            labels_binarized_one_hot = one_hot(labels, num_classes=2, dim=1)

            loss = criterion(output_one_hot, labels)
            test_loss += loss.item()
            test_loss_list.append(loss)

            output_one_hot = torch.nn.functional.softmax(output_one_hot)
            output_binarized_one_hot = (output_one_hot > 0.5)
            test_mIoU_macro += mIoU(output_one_hot, labels_binarized_one_hot)
            test_mIoU_macro_list.append(mIoU(output_one_hot, labels_binarized_one_hot))
            test_hausdorff(y_pred=output_binarized_one_hot, y=labels_binarized_one_hot)
            test_hd95 += test_hausdorff.aggregate().item()
            test_hd95_list.append(test_hausdorff.aggregate().item())
            test_dice_macro += test_dice(preds=output_one_hot, target=labels.type(torch.int8))
            test_dice_macro_list.append(test_dice(preds=output_one_hot, target=labels.type(torch.int8)))
            test_acc += pixel_accuracy(output_one_hot, labels_binarized_one_hot)
            test_acc_list.append(pixel_accuracy(output_one_hot, labels_binarized_one_hot))

            output = torch.argmax(output_one_hot, dim=1)
            output = torch.unsqueeze(output, dim=1)

            tp, fp, fn, tn = smp.metrics.get_stats(output, labels.long(), mode='binary', threshold=0.5)
            test_sensitivity += smp.metrics.functional.sensitivity(tp, fp, fn, tn, reduction='micro-imagewise')
            test_sensitivity_list.append(smp.metrics.functional.sensitivity(tp, fp, fn, tn, reduction='micro-imagewise'))
            test_specificity += smp.metrics.functional.specificity(tp, fp, fn, tn, reduction='micro-imagewise')
            test_specificity_list.append(smp.metrics.functional.specificity(tp, fp, fn, tn, reduction='micro-imagewise'))
            test_precision += smp.metrics.functional.precision(tp, fp, fn, tn, reduction='micro-imagewise')
            test_precision_list.append(smp.metrics.functional.precision(tp, fp, fn, tn, reduction='micro-imagewise'))
            test_negative_predictive_value += smp.metrics.functional.negative_predictive_value(tp, fp, fn, tn, reduction='micro-imagewise')
            test_negative_predictive_value_list.append(smp.metrics.functional.negative_predictive_value(tp, fp, fn, tn, reduction='micro-imagewise'))

    raw_data = {
        'loss': [loss.cpu().numpy() / len(test_loader)],
        'mIoU_macro': [test_mIoU_macro / len(test_loader)],
        'dice_macro': [test_dice_macro.cpu().detach().numpy() / len(test_loader)],
        'accuracy':  [test_acc / len(test_loader)],
        'hd95': [test_hd95 / len(test_loader)],
        'sensitivity': [test_sensitivity.cpu().detach().numpy() / len(test_loader)],
        'specificity': [test_specificity.cpu().detach().numpy() / len(test_loader)],
        'precision': [test_precision.cpu().detach().numpy() / len(test_loader)],
        'negative_predictive_value': [test_negative_predictive_value.cpu().detach().numpy() / len(test_loader)],
      }

    test_loss_list = torch.tensor(test_loss_list, device='cpu')
    test_mIoU_macro_list = torch.tensor(test_mIoU_macro_list, device='cpu')
    test_dice_macro_list = torch.tensor(test_dice_macro_list, device='cpu')
    test_acc_list = torch.tensor(test_acc_list, device='cpu')
    test_hd95_list = torch.tensor(test_hd95_list, device='cpu')
    test_sensitivity_list = torch.tensor(test_sensitivity_list, device='cpu')
    test_specificity_list = torch.tensor(test_specificity_list, device='cpu')
    test_precision_list = torch.tensor(test_precision_list, device='cpu')
    test_negative_predictive_value_list = torch.tensor(test_negative_predictive_value_list, device='cpu')
    raw_data_slice = {
        'loss': test_loss_list.numpy(),
        'mIoU_macro': test_mIoU_macro_list.numpy(),
        'dice_macro': test_dice_macro_list.numpy(),
        'accuracy':  test_acc_list.numpy(),
        'hd95': test_hd95_list.numpy(),
        'sensitivity': test_sensitivity_list.numpy(),
        'specificity': test_specificity_list.numpy(),
        'precision': test_precision_list.numpy(),
        'negative_predictive_value': test_negative_predictive_value_list.numpy(),
      }

    df = pd.DataFrame(raw_data, columns=['loss', 'mIoU_macro', 'dice_macro', 'accuracy', 'hd95', 'sensitivity',
                                         'specificity', 'precision', 'negative_predictive_value'])

    df_slice = pd.DataFrame(raw_data_slice, columns=['loss', 'mIoU_macro', 'dice_macro', 'accuracy', 'hd95', 'sensitivity',
                                         'specificity', 'precision', 'negative_predictive_value'])

    df.to_csv(cfg['CSV'] + '_average.csv', index=True)
    df_slice.to_csv(cfg['CSV'] + '_slicewise.csv', index=True)

    print("Test Loss: {:.3f}..".format(test_loss / len(test_loader)),
          "Test mIoU macro: {:.3f}..".format(test_mIoU_macro / len(test_loader)),
          "Test Dice macro: {:.3f}..".format(test_dice_macro / len(test_loader)),
          "Test hd95: {:.3f}..".format(test_hd95 / len(test_loader)),
          "Test Acc:{:.3f}..".format(test_acc / len(test_loader)),
          "Test Sensitivity:{:.3f}..".format(test_sensitivity / len(test_loader)),
          "Test Specificity:{:.3f}..".format(test_specificity / len(test_loader)),
          "Test Precision:{:.3f}..".format(test_precision / len(test_loader)),
          "Test Negative Predictive Value:{:.3f}..".format(test_negative_predictive_value / len(test_loader)))


########################################################################################################################
# model training

def fit(epochs, model, train_loader, val_loader, criterion, optimizer, scheduler):
    torch.cuda.empty_cache()

    val_hausdorff = monai.metrics.HausdorffDistanceMetric(include_background=True, reduction="mean", percentile=95.0, get_not_nans=False)
    val_dice = Dice(num_classes=2, average='macro').to(device)
    val_jc = JaccardIndex(num_classes=2, average='micro').to(device)

    min_loss = np.inf
    decrease = 1
    not_improve = 0
    lrs = []

    columns = ["image", "prediction", "ground truth"]
    wandb_table = wandb.Table(columns=columns)

    t0 = time.time()
    for e in range(epochs):
        train_loss = 0.0
        train_mIoU_macro = 0.0
        train_acc = 0.0
        model.train()  # Optional when not using Model Specific layer
        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            output_one_hot = model(images)

            labels_binarized_one_hot = one_hot(labels, num_classes=2, dim=1)

            loss = criterion(output_one_hot, labels)
            loss.backward()

            clip = cfg['Clip']
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

            optimizer.step()
            train_loss += loss.item()

            output_one_hot = torch.nn.functional.softmax(output_one_hot)
            train_mIoU_macro += mIoU(output_one_hot, labels_binarized_one_hot).item()
            train_acc += pixel_accuracy(output_one_hot, labels_binarized_one_hot)

            lrs.append(get_lr(optimizer))
            if scheduler is not None:
                scheduler.step()

        # validation during training
        idx = 0
        val_loss = 0.0
        val_mIoU_macro = 0.0
        val_mIoU_micro = 0.0
        val_hd95 = 0.0
        val_dice_macro = 0.0
        val_acc = 0.0
        model.eval()
        for images, labels in tqdm(val_loader):
            with torch.no_grad():
                images, labels = images.to(device), labels.to(device)

                output_one_hot = model(images)

                labels_binarized_one_hot = one_hot(labels, num_classes=2, dim=1)

                loss = criterion(output_one_hot, labels)
                val_loss += loss.item()

                output_one_hot = torch.nn.functional.softmax(output_one_hot)
                output_binarized_one_hot = (output_one_hot > 0.5)
                val_mIoU_macro += mIoU(output_one_hot, labels_binarized_one_hot).item()
                val_mIoU_micro += val_jc(preds=output_one_hot, target=labels.type(torch.int8)).item()
                val_hausdorff(y_pred=output_binarized_one_hot, y=labels_binarized_one_hot)
                val_hd95 += val_hausdorff.aggregate().item()
                val_dice_macro += val_dice(preds=output_one_hot, target=labels.type(torch.int8)).item()
                val_acc += pixel_accuracy(output_one_hot, labels_binarized_one_hot)

                # plot some pictures in wandb to see how the model performance for validation set
                if idx % 10 == 0:
                    image_wb = images[1,:,:,:].cpu().detach().numpy().transpose(1,2,0)
                    pred_wb = torch.argmax(output_one_hot[1,:,:,:], dim=0)
                    pred_wb = pred_wb.cpu().detach().numpy()
                    label_wb = labels[1,:,:,:].type(torch.FloatTensor)
                    label_wb = label_wb.cpu().detach().numpy().transpose(1,2,0)
                    wandb_table.add_data(wandb.Image(image_wb), wandb.Image(pred_wb), wandb.Image(label_wb))
                    new_table = wandb.Table(columns=wandb_table.columns, data=wandb_table.data)
                    wandb.log({"validation samples": new_table}, commit=False)

                idx += 1

        # save best performing models
        if min_loss > (val_loss / len(val_loader)):
            print('Loss Decreasing.. {:.3f} >> {:.3f} '.format(min_loss, (val_loss / len(val_loader))))
            min_loss = (val_loss / len(val_loader))
            decrease += 1
            if decrease % 2 == 0:
                print('saving model...')
                torch.save(model, cfg['SaveModel'] + 'mIoU_macro-{:.3f}.pt'.format(val_mIoU_macro / len(val_loader)))
                torch.save(model, cfg['SaveModel'] + 'pixel_acc-{:.3f}.pt'.format(val_acc / len(val_loader)))

        if (val_loss / len(val_loader)) > min_loss:
            not_improve += 1
            min_loss = (val_loss / len(val_loader))
            print(f'Loss Not Decrease for {not_improve} time')
            if not_improve == 100:
                print('Loss not decrease for 50 times, Stop Training')
                break

        print("Epoch:{}/{}..".format(e + 1, epochs),
              "Train Loss: {:.3f}..".format(train_loss / len(train_loader)),
              "Val Loss: {:.3f}..".format(val_loss / len(val_loader)),
              "Val mIoU: {:.3f}..".format(val_mIoU_macro / len(val_loader)),
              "Val Acc:{:.3f}..".format(val_acc / len(val_loader)),
              "Time: {:.2f}sec".format(time.time() - t0))

        # log data for wandb for better visualization and comparison between runs and models
        wandb.log({'train loss': train_loss / len(train_loader), 'train accuracy': train_acc / len(train_loader),
                   'val loss': val_loss / len(val_loader), 'train mIoU macro': train_mIoU_macro / len(train_loader),
                   'val mIoU macro': val_mIoU_macro / len(val_loader), 'val accuracy (dice micro)': val_acc / len(val_loader),
                   'val mIoU micro': val_mIoU_micro / len(val_loader), 'val dice macro': val_dice_macro / len(val_loader),
                   'val hd95': val_hd95 / len(val_loader), 'learning rate': get_lr(optimizer)})

    print('Training finished!')


if __name__ == '__main__':
    cfg = fParseConfig('/home/students/studborst1/MotionDetection/MotionDetection/config.yml')

    device = torch.device(cfg['GPU'])
    random.seed(cfg['SEED'])
    np.random.seed(cfg['SEED'])
    torch.manual_seed(cfg['SEED'])
    torch.cuda.manual_seed(cfg['SEED'])

    data = Data(cfg)
    partition = create_list_ID_training(cfg)

    if cfg['Mode'] == 'training':
        # load training and validation data
        images_patches_list, mask_patches_list = data.process_image_data(partition['train'])
        train_data = [{"image": img, "label": label} for img, label in zip(images_patches_list, mask_patches_list)]

        train_data, val_data = train_test_split(train_data, test_size=cfg['Train_test_ratio'], random_state=cfg['SEED'])

        train_dataset_local = data.create_image_space_train_dataset(train_data)
        validation_dataset_local = data.create_image_space_test_dataset(val_data)

        train_loader = DataLoader(train_dataset_local, batch_size=cfg['BatchSize'], shuffle=True,
                                  num_workers=2, pin_memory=True)
        val_loader = DataLoader(validation_dataset_local, batch_size=cfg['BatchSize'], shuffle=False,
                                num_workers=4, pin_memory=True)

        # select model
        if cfg['Model'] == 'UNet':
            model = BasicUNet(spatial_dims=2, in_channels=1, out_channels=2, features=(32, 64, 128, 256, 512, 64),
                              upsample='deconv')
        elif cfg['Model'] == 'Swin_UNet':
            config = get_config()
            model = ViT_seg(config, img_size=224, num_classes=cfg['Classes']).to(device)
        elif cfg['Model'] == 'RegionViT_FPN':
            model = FPN(encoder_name='regionvit', encoder_depth=5, encoder_weights='None', decoder_pyramid_channels=512,
                        decoder_segmentation_channels=128, in_channels=512, classes=2, activation=None)
        elif cfg['Model'] == 'RegionViT_UNet':
            model = regionvit_unet.RegionViT_UNET()
        else:
            print('No model found')

        model = model.to(device)

        # select optimizer
        if cfg['Optimizer'] == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=cfg['LearningRate'], momentum=0.9, weight_decay=1e-4,
                                  nesterov=True)
            scheduler = OneCycleLR(optimizer, max_lr=cfg['LearningRate'], epochs=cfg['Epochs'],
                                   steps_per_epoch=len(train_loader), div_factor=cfg['DivFactor'])
        elif cfg['Optimizer'] == 'AdamW':
            optimizer = optim.AdamW(model.parameters(), lr=cfg['LearningRate'], betas=(0.9, 0.999), eps=1e-08,
                                    weight_decay=1e-4)
            scheduler = OneCycleLR(optimizer, max_lr=cfg['LearningRate'], epochs=cfg['Epochs'],
                                   steps_per_epoch=len(train_loader), div_factor=cfg['DivFactor'])

        # select loss function
        if cfg['LossFunction'] == 'DiceCELoss':
            criterion = DiceCELoss(to_onehot_y=True, softmax=True)
        elif cfg['LossFunction'] == 'DiceFocalLoss':
            criterion = DiceFocalLoss(to_onehot_y=True, focal_weight=[10, 0.1])

        # wandb
        config = {"Epochs": cfg['Epochs'], "BatchSize": cfg['BatchSize'], "ImageSize": cfg['PatchSize'],
                  "Optimizer": cfg['Optimizer'], "MaxLearningRate": cfg['LearningRate'], "DivFactor": cfg['DivFactor'],
                  "PatchOverlap_NAKO": cfg['PatchOverlap_NAKO'], "PatchOverlap_MRP": cfg['PatchOverlap_MRP'],
                  "HoldOutPatient": cfg['SelectedPatient'], "Scheduler": cfg['Scheduler'],
                  "Normalized": cfg['Normalized'], "Clip": cfg['Clip']}

        wandb.init(settings=wandb.Settings(start_method="fork"), config=config, project=cfg['Wandb'])
        wandb.watch_called = False

        # start training
        fit(cfg['Epochs'], model, train_loader, val_loader, criterion, optimizer, scheduler)

    if cfg['Mode'] == 'prediction':
        # select model and get weights from checkpoint
        if cfg['Model'] == 'UNet':
            model = BasicUNet(spatial_dims=2, in_channels=1, out_channels=2, features=(32, 64, 128, 256, 512, 64),
                              upsample='deconv')
        elif cfg['Model'] == 'Swin_UNet':
            config = get_config()
            model = ViT_seg(config, img_size=224, num_classes=cfg['Classes']).to(device)
        elif cfg['Model'] == 'RegionViT_FPN':
            model = FPN(encoder_name='regionvit', encoder_depth=5, encoder_weights='None', decoder_pyramid_channels=512,
                        decoder_segmentation_channels=128, in_channels=512, classes=2, activation=None)
        elif cfg['Model'] == 'RegionViT_UNet':
            model = regionvit_unet.RegionViT_UNET()
        else:
            print('No model found')

        checkpoint = torch.load(cfg['Checkpoint'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)

        # get test data
        if cfg['Database'][0] != 'NAKO194':
            images_patches_list_test, mask_patches_list_test, images_test, labels_test = data.process_image_data(partition['test'])

            test_data = [{"image": img, "label": label} for img, label in zip(images_patches_list_test, mask_patches_list_test)]
            test_dataset_local = data.create_image_space_test_dataset(test_data)

            test_loader = DataLoader(test_dataset_local, batch_size=cfg['BatchSize'], shuffle=False,
                                      num_workers=8, pin_memory=True)

            # select loss function
            if cfg['LossFunction'] == 'DiceCELoss':
                criterion = DiceCELoss(to_onehot_y=True, softmax=True)
            elif cfg['LossFunction'] == 'DiceFocalLoss':
                criterion = DiceFocalLoss(to_onehot_y=True, focal_weight=[10, 0.1])

            # test model
            predict(model, test_loader, criterion)

            # plot segmentation masks if necessary
            if cfg['Plotting'] == True:
                data.plot_patient(images_test, labels_test, partition, model, device)

        else:
            images_test = []
            labels_test = []
            data.plot_patient(images_test, labels_test, partition, model, device)
