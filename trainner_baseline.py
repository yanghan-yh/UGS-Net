from model.Models import *
from dataprocess import *
import loss.losses as losses
from metrics import *
import torch.optim as optim
import time
import numpy as np
import os
import torch
from config import Config
import shutil
from tqdm import tqdm
import imageio
import math
from bisect import bisect_right

config = Config()

torch.cuda.set_device(config.gpu)  

model_name = config.arch
if not os.path.isdir('result'):
    os.mkdir('result')
if config.resume is False:
    with open('result/' + str(os.path.basename(__file__).split('.')[0]) + '_' + model_name + '.txt', 'a+') as f:
        f.seek(0)
        f.truncate()
model = U_Net()
model.cuda()
best_dice = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, betas=(0.5, 0.999))    
dataloader, dataloader_val = get_dataloader(config, batchsize=config.batch_size, mode='row')   # 64
criterion = losses.init_loss('BCE_logit').cuda()

if config.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    if config.evaluate:
        checkpoint = torch.load('./checkpoint/' + str(model_name) + '_best.pth.tar')
    else:
        checkpoint = torch.load('./checkpoint/' + str(model_name) + '.pth.tar')
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    best_dice = checkpoint['dice']
    start_epoch = config.epochs

def adjust_lr(optimizer, epoch, eta_max=0.0001, eta_min=0.):
    cur_lr = 0.
    if config.lr_type == 'SGDR':
        i = int(math.log2(epoch / config.sgdr_t + 1))
        T_cur = epoch - config.sgdr_t * (2 ** (i) - 1)
        T_i = (config.sgdr_t * 2 ** i)

        cur_lr = eta_min + 0.5 * (eta_max - eta_min) * (1 + np.cos(np.pi * T_cur / T_i))

    elif config.lr_type == 'multistep':
        cur_lr = config.learning_rate * 0.1 ** bisect_right(config.milestones, epoch)

    for param_group in optimizer.param_groups:
        param_group['lr'] = cur_lr
    return cur_lr

def train(epoch):
    model.train()
    train_loss = 0

    start_time = time.time()
    for batch_idx, (inputs, lungs, medias, targets_u, targets_i, targets_s) in enumerate(dataloader):
        iter_start_time = time.time()
        inputs = inputs.cuda()
        lungs = lungs.cuda()
        medias = medias.cuda()
        targets_i = targets_i.cuda()
        targets_u = targets_u.cuda()
        targets_s = targets_s.cuda()

        outputs = model(medias)

        outputs_sig = torch.sigmoid(outputs)

        loss_seg = criterion(outputs_sig, targets_u)

        loss_all = loss_seg

        optimizer.zero_grad()
        loss_all.backward()
        optimizer.step()

        train_loss += loss_all.item()

        print('Epoch:{}\t batch_idx:{}/All_batch:{}\t duration:{:.3f}\t loss_all:{:.3f}'
          .format(epoch, batch_idx, len(dataloader), time.time()-iter_start_time, loss_all.item()))
        iter_start_time = time.time()
    print('Epoch:{0}\t duration:{1:.3f}\ttrain_loss:{2:.6f}'.format(epoch, time.time()-start_time, train_loss/len(dataloader)))
    
    with open('result/' + str(os.path.basename(__file__).split('.')[0]) + '_' + model_name + '.txt', 'a+') as f:
        f.write('Epoch:{0}\t duration:{1:.3f}\t learning_rate:{2:.6f}\t train_loss:{3:.4f}'
          .format(epoch, time.time()-start_time, config.learning_rate, train_loss/len(dataloader)))

def test(epoch):
    global best_dice
    model.eval()
    dices_all = []
    ious_all = []
    nsds_all = []
    with torch.no_grad():
        for batch_idx, (inputs, lungs, medias, targets_u, targets_i, targets_s) in enumerate(dataloader_val):
            inputs = inputs.cuda()
            medias = medias.cuda()
            targets_i = targets_i.cuda()
            targets_u = targets_u.cuda()
            targets_s = targets_s.cuda()

            outputs = model(medias)

            outputs_final_sig = torch.sigmoid(outputs)

            dices_all = meandice(outputs_final_sig, targets_u, dices_all)
            ious_all = meandIoU(outputs_final_sig, targets_u, ious_all)
            nsds_all = meanNSD(outputs_final_sig, targets_u, nsds_all)

            print('Epoch:{}\tbatch_idx:{}/All_batch:{}\tdice:{:.4f}\tiou:{:.4f}\tnsd:{:.4f}'
            .format(epoch, batch_idx, len(dataloader_val), np.mean(np.array(dices_all)), np.mean(np.array(ious_all)), np.mean(np.array(nsds_all))))
        with open('result/' + str(os.path.basename(__file__).split('.')[0]) + '_' + model_name + '.txt', 'a+') as f:
            f.write('\tdice:{:.4f}\tiou:{:.4f}\tnsd:{:.4f}'.format(np.mean(np.array(dices_all)), np.mean(np.array(ious_all)), np.mean(np.array(nsds_all)))+'\n')

    # Save checkpoint.
    if config.resume is False:
        dice = np.mean(np.array(dices_all))
        print('Test accuracy: ', dice)
        state = {
            'model': model.state_dict(),
            'dice': dice,
            'epoch': epoch,
            'optimizer': optimizer.state_dict()
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/'+str(model_name)+'.pth.tar')

        is_best = False
        if best_dice < dice:
            best_dice = dice
            is_best = True

        if is_best:
            shutil.copyfile('./checkpoint/' + str(model_name) + '.pth.tar',
                            './checkpoint/' + str(model_name) + '_best.pth.tar')
        print('Save Successfully')
        print('------------------------------------------------------------------------')

if __name__ == '__main__':

    if config.resume:
        test(start_epoch)
    else:
        for epoch in tqdm(range(start_epoch, config.epochs)):
            train(epoch)
            test(epoch)
