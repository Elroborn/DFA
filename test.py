from dataset import AlignedDataset
from torch.utils.data import DataLoader
from torch import nn
from model import FaceModel
from options import opt
import torchvision.utils as vutils
import os
import torch
import numpy as np
from statistics import PADMeter
def eval_model(data_loader,model):
    model.eval()
    pad_meter = PADMeter()
    for data in data_loader:
        model.set_input(data)
        model.forward()
        class_output = nn.functional.softmax(model.output, dim=1)
        pad_meter.update(model.label.cpu().data.numpy(),
                            class_output.cpu().data.numpy())
    return pad_meter
if __name__ == '__main__':
    dev_file_list = opt.dev_file_list
    test_file_list = opt.test_file_list
    dev_data_loader = DataLoader(AlignedDataset(dev_file_list,isTrain= False), batch_size=20,
                                   shuffle=False, num_workers=8)
    test_data_loader = DataLoader(AlignedDataset(test_file_list,isTrain= False), batch_size=20,
                                   shuffle=False, num_workers=8)
    model = FaceModel(opt,isTrain = False)
    model.load_networks("best")
    model.eval()

    pad_dev_mater = eval_model(dev_data_loader,model) 
    pad_meter = eval_model(test_data_loader,model)

    pad_meter.get_eer_and_thr()
    pad_dev_mater.get_eer_and_thr()

    pad_meter.get_hter_apcer_etal_at_thr(pad_dev_mater.threshold) # pad_meter.threshold
    pad_meter.get_accuracy(pad_dev_mater.threshold)

    print('HTER {pad_meter.hter:.4f} EER {pad_meter.eer:.4f} ACC {pad_meter.accuracy:.4f}'.format(
        pad_meter=pad_meter))
    print('APCER {pad_meter.apcer:.4f} BPCER {pad_meter.bpcer:.4f} ACER {pad_meter.acer:.4f} AUC {pad_meter.auc:.4f}'.format(
        pad_meter=pad_meter))

