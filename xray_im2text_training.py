import os
from tensorboard_logger import configure, log_value
import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.utils.data as torchdata
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tqdm
import utils
import torch.optim as optim
import pdb
from data_loader import OpenI
from PIL import Image
from xray_utils import transform_xray, get_model
import cv2
import numpy as np
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

import argparse
parser = argparse.ArgumentParser(description='Radiograph_Pretraining')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--wd', type=float, default=0.0, help='weight decay')
parser.add_argument('--cv_dir', default='cv/', help='checkpoint directory (models and logs are saved here)')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--epoch_step', type=int, default=10000, help='epochs after which lr is decayed')
parser.add_argument('--max_epochs', type=int, default=10000, help='total epochs to run')
parser.add_argument('--parallel', action ='store_true', default=False, help='use multiple GPUs for training')
args = parser.parse_args()

if not os.path.exists(args.cv_dir):
    os.system('mkdir ' + args.cv_dir)
utils.save_args(__file__, args)

def train(epoch, counter):

    rnet.train()
    losses = []
    for batch_idx, (inputs, targets) in tqdm.tqdm(enumerate(trainloader), total=len(trainloader)):
        inputs, targets = Variable(inputs), Variable(targets).cuda()
        if not args.parallel:
            inputs = inputs.cuda()

        v_inputs = Variable(inputs.data, volatile=True)

        preds = rnet.forward(v_inputs)

        loss = criterion(preds, targets, torch.ones((inputs.size(0))).cuda())
        if batch_idx % 50 == 0:
            log_value('train_loss_iteration', loss, counter)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.cpu())
        counter += 1

    loss = torch.stack(losses).mean()
    log_str = 'E: %d | L: %.3f'%(epoch, loss)
    print(log_str)

    log_value('train_loss_epoch', loss, epoch)

def test(epoch):

    rnet.eval()
    losses = []
    for batch_idx, (inputs, targets) in tqdm.tqdm(enumerate(devloader), total=len(devloader)):

        inputs, targets = Variable(inputs, volatile=True), Variable(targets).cuda()
        if not args.parallel:
            inputs = inputs.cuda()

        v_inputs = Variable(inputs.data, volatile=True)

        preds = rnet.forward(v_inputs)

        loss = criterion(preds, targets, torch.ones((inputs.size(0))).cuda())

        losses.append(loss.cpu())

    loss = torch.stack(losses).mean()
    log_str = 'TS: %d | L: %.3f'%(epoch, loss)
    print(log_str)

    log_value('test_loss', loss, epoch)

    # save the model parameters
    rnet_state_dict = rnet.module.state_dict() if args.parallel else rnet.state_dict()

    state = {
      'state_dict': rnet_state_dict,
      'epoch': epoch,
    }
    torch.save(state, args.cv_dir+'/ckpt_E_%d'%(epoch))
#--------------------------------------------------------------------------------------------------------#
train_set = OpenI('train', 'data', transform_xray())
dev_set = OpenI('dev', 'data', transform_xray(), doc2vec_file="report2vec.model")
test_set = OpenI('test', 'data', transform_xray(), doc2vec_file="report2vec.model")
trainloader = torchdata.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=16)
devloader = torchdata.DataLoader(dev_set, batch_size=args.batch_size, shuffle=False, num_workers=4)
testloader = torchdata.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4)

rnet = get_model(50)
if args.parallel:
    rnet = nn.DataParallel(rnet)
rnet.cuda()

start_epoch = 0
counter = 0
criterion = nn.CosineEmbeddingLoss()
optimizer = optim.Adam(rnet.parameters(), lr=args.lr)
configure(args.cv_dir+'/log', flush_secs=5)
for epoch in range(start_epoch, start_epoch+args.max_epochs+1):
    train(epoch, counter)
    if epoch % 1 == 0:
        test(epoch)
    lr_scheduler.step()
