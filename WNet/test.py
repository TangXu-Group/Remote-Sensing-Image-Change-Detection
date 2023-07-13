import torch
import cv2
import os
import glob
import numpy as np
from option.test_options import TestOptions
from data.data_loader import CreateDataLoader
from model.cd_model import create_model
from util.visualizer import Visualizer
from util.metric_tool import ConfuseMatrixMeter


if __name__ == '__main__':
    opt = TestOptions().parse(save=False)

    test_loader = CreateDataLoader(opt)
    test_data = test_loader.load_data()
    visualizer = Visualizer(opt)

    cd_model = create_model(opt)
    if opt.checkpoint_name is not None:
        checkpoint = torch.load(opt.checkpoint_name)
        cd_model.load_state_dict(checkpoint['network'], False)
    else:
        assert opt.load_pretrain == True

    running_metric = ConfuseMatrixMeter(n_class=2)
    running_metric.clear()
    opt.phase = 'test'
    cd_model.eval()
    with torch.no_grad():
        for i, data in enumerate(test_data):
            try:
                test_pred = cd_model.module.inference(data['t1_img'].cuda(), data['t2_img'].cuda())
            except:
                test_pred = cd_model.inference(data['t1_img'], data['t2_img'])
            # update metric
            test_target = data['label'].detach()
            test_pred = torch.argmax(test_pred.detach(), dim=1)
            pred = test_pred.cpu().numpy()
            pred = np.squeeze(pred)
            test_acc = running_metric.update_cm(pr=test_pred.cpu().numpy(), gt=test_target.cpu().numpy())
        
        test_scores = running_metric.get_scores()
        epoch = str(opt.which_epoch) if opt.load_pretrain else 'unknown'
        visualizer.print_scores(opt.phase, epoch, test_scores)
