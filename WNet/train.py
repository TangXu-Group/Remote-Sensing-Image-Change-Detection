import os
import time
import torch
import numpy as np
from data.data_loader import CreateDataLoader
from model.cd_model import *
from option.train_options import TrainOptions
from util.visualizer import Visualizer
from util.metric_tool import ConfuseMatrixMeter
import math


def lcm(a,b): return abs(a * b)/math.gcd(a,b) if a and b else 0

if __name__ == '__main__':
    opt = TrainOptions().parse()
    iter_path = os.path.join(opt.checkpoint_dir, opt.name, 'iter.txt')
    if opt.load_pretrain:
        try:
            start_epoch, epoch_iter = np.loadtxt(iter_path, delimiter=',', dtype=int)
        except:
            start_epoch, epoch_iter = 1, 0
        print('Resuming from epoch %d at iteration %d' % (start_epoch, epoch_iter))
    else:
        start_epoch, epoch_iter = 1, 0

    train_loader = CreateDataLoader(opt)
    train_data = train_loader.load_data()
    train_size = len(train_loader)
    print('#training images = %d' % train_size)
    opt.phase = 'val'
    val_loader = CreateDataLoader(opt)
    val_data = val_loader.load_data()
    val_size = len(val_loader)
    print('#validation images = %d' % val_size)
    opt.phase = 'train'

    cd_model = create_model(opt)
    try:
        optimizer = cd_model.module.optimizer
    except:
        optimizer = cd_model.optimizer
    visualizer = Visualizer(opt)

    opt.print_freq = lcm(opt.print_freq, opt.batch_size)
    total_steps = (start_epoch-1) * train_size + epoch_iter
    display_delta = total_steps % opt.display_freq
    print_delta = total_steps % opt.print_freq
    save_delta = total_steps % opt.save_latest_freq

    tmp = 1
    running_metric = ConfuseMatrixMeter(n_class=2)
    TRAIN_ACC = np.array([], np.float32)
    VAL_ACC = np.array([], np.float32)
    best_val_acc = 0.0
    best_epoch = 0

    for epoch in range(start_epoch, opt.num_epochs + opt.num_decay_epochs + 1):
        epoch_start_time = time.time()
        if epoch != start_epoch:
            epoch_iter = epoch_iter % train_size
        running_metric.clear()
        opt.phase = 'train'
        cd_model.train()
        for i, data in enumerate(train_data, start=epoch_iter):
            if total_steps % opt.print_freq == print_delta:
                iter_start_time = time.time()
            total_steps += opt.batch_size
            epoch_iter += opt.batch_size

            ############## Forward Pass ######################
            cd_loss, cd_pred = cd_model(data['t1_img'].cuda(), data['t2_img'].cuda(), data['label'].cuda())
            # sum per device losses
            cd_loss = [torch.mean(x) if not isinstance(x, int) else x for x in cd_loss]
            try:
                loss_dict = dict(zip(cd_model.module.loss_names, cd_loss))
            except:
                loss_dict = dict(zip(cd_model.loss_names, cd_loss))

            # calculate final loss scalar
            if opt.use_ce_loss:
                loss = loss_dict['CE']
            elif opt.use_hybrid_loss:
                loss = loss_dict['Focal'] + loss_dict['Dice']
            else:
                raise NotImplementedError

            ############### Backward Pass ####################
            # update generator weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        running_metric.clear()
        opt.phase = 'val'
        cd_model.eval()
        with torch.no_grad():
            for i, data in enumerate(val_data):
                try:
                    val_pred = cd_model.module.inference(data['t1_img'].cuda(), data['t2_img'].cuda())
                except:
                    val_pred = cd_model.inference(data['t1_img'], data['t2_img'])
                # update metric
                val_target = data['label'].detach()
                val_pred = torch.argmax(val_pred.detach(), dim=1)
                val_acc = running_metric.update_cm(pr=val_pred.cpu().numpy(), gt=val_target.cpu().numpy())
                
            val_scores = running_metric.get_scores()
            visualizer.print_scores(opt.phase, epoch, val_scores)
            val_epoch_acc = val_scores['mf1']
            VAL_ACC = np.append(VAL_ACC, [val_epoch_acc])
            np.save(os.path.join(opt.checkpoint_dir, opt.name,  'val_acc.npy'), VAL_ACC)
            if val_epoch_acc > best_val_acc:
                best_val_acc = val_epoch_acc
                best_epoch = epoch

        # end of epoch
        iter_end_time = time.time()
        print('End of epoch %d / %d \t Time Taken: %d sec \t best acc: %.5f (at epoch: %d) ' %
            (epoch, opt.num_epochs + opt.num_decay_epochs, time.time() - epoch_start_time, best_val_acc, best_epoch))

        if epoch == best_epoch:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
            try:
                cd_model.module.save(epoch)
            except:
                cd_model.save(epoch)
            np.savetxt(iter_path, (epoch + 1, 0), delimiter=',', fmt='%d')

        ### linearly decay learning rate after certain iterations
        if epoch > opt.num_epochs:
            try:
                cd_model.module.update_learning_rate()
            except:
                cd_model.update_learning_rate()

