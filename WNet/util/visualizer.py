"""
Copied and modified from
https://github.com/NVIDIA/pix2pixHD/tree/master/util
"""
import os
import torch
import numpy as np
import time
from . import util
import matplotlib.pyplot as plt
from skimage import io


class Visualizer():
    def __init__(self, opt):
        self.name = opt.name
        self.log_name = os.path.join(opt.checkpoint_dir, opt.name, 'loss_log.txt')
        self.visualization_dir = os.path.join(opt.checkpoint_dir, opt.name, 'visualization')
        self.result_dir = os.path.join(opt.result_dir)

        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)


    def print_current_errors_acc(self, phase, epoch, i, errors, acc, t):
        message = '(phase: %s, epoch: %d, iters: %d, running_mf1: %.5f, time: %.5f) ' % (phase, epoch, i, acc, t)
        for k, v in errors.items():
            if v != 0:
                message += '%s: %.3f ' % (k, v)

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)


    def print_scores(self, phase, epoch, scores):
        message = '(phase: %s, epoch: %s) ' % (phase, epoch)
        for k, v in scores.items():
            message += '%s: %.3f ' % (k, v*100)

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)


    def visualize_current_results(self, phase, epoch, t1_img, t2_img, gt, pred):
        file_name = 'phase_' + str(phase) + '_epoch_' + str(epoch) + '.jpg'
        t1_img_ = util.make_numpy_grid(util.de_norm(t1_img))
        t2_img_ = util.make_numpy_grid(util.de_norm(t2_img))
        gt_ = util.make_numpy_grid(gt)
        pred_ = torch.argmax(pred, dim=1, keepdim=True)
        pred_ = util.make_numpy_grid(pred_ * 255)

        vis = np.concatenate([t1_img_, t2_img_, gt_, pred_], axis=0)
        vis = np.clip(vis, a_min=0.0, a_max=1.0)
        save_path = os.path.join(self.visualization_dir, file_name)
        plt.imsave(save_path, vis)


    def save_pred(self, t1_img, t2_img, name, model):
        model.eval()
        try:
            preds = model.module.inference(t1_img.cuda(), t2_img.cuda())
        except:
            preds = model.inference(t1_img, t2_img)
        pred = torch.argmax(preds, dim=1, keepdim=True)
        pred = pred * 255
        save_path = os.path.join(self.result_dir, str(name) + '.png')
        pred = pred[0].cpu().numpy()
        io.imsave(save_path, np.array(np.squeeze(pred), dtype=np.uint8))

