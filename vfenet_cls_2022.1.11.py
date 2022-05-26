import random, shutil, json
import os
from os.path import join, exists, isfile, realpath, dirname
from torch.autograd import Variable
from os import makedirs, remove, chdir, environ
from tools.code_helpers_public_pr_curve import getPAt100R, getPR, getPRCurve, getPRCurveWrapper, getRAt99P, get_recall_helper
from tools.correspondence_event_camera_frame_camera import traverse_to_name, name_to_consumervideo, video_beginning

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import torchvision
import numpy as np
import argparse
from datetime import datetime
from tensorboardX import SummaryWriter
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
import scipy
from scipy.spatial import distance

import model.VFENet_tran as VFENet

from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as tf

parser = argparse.ArgumentParser(description='VFENet_cls')
parser.add_argument('--mode', type=str, default='train', help='Mode', choices=['train', 'test', 'draw'])
parser.add_argument('--data', type=str, default='ve', help='data mode', choices=['ve','ov','oe'])
parser.add_argument('--resume', type=str, default='', help='Path to load checkpoint from, for resuming training or testing.')
parser.add_argument('--ckpt', type=str, default='best', help='Resume from latest or best checkpoint.', choices=['latest', 'best'])
parser.add_argument('--runsPath', type=str, default='/home/hz/Workspace/EventVPR/vidFe/', help='Path to save runs to.')
parser.add_argument('--savePath', type=str, default='checkpoints', help='Path to save checkpoints to in logdir. Default=checkpoints/')

parser.add_argument('--cuda', action='store_true', help='use cuda')
parser.add_argument('--oc', type=int, default=0, help='cuda number')
parser.add_argument('--nGPU', type=int, default=1, help='number of GPU to use.')
parser.add_argument('--seed', type=int, default=123, help='Random seed to use.')
parser.add_argument('--patience', type=int, default=15, help='Patience for early stopping. 0 is off.')
parser.add_argument('--evalEvery', type=int, default=1, help='Do a validation set run, and save, every N epochs.')
parser.add_argument('--printFrequence', type=int, default=10, help='the frequence of printing data to the screen')
parser.add_argument('--nEpochs', type=int, default=300, help='number of epochs to train for')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')

parser.add_argument('--batch_size', type=int, default=128,  help='rgb+event.')
parser.add_argument('--threads', type=int, default=8, help='Number of threads for each data loader to use')
parser.add_argument('--dataset_name', type=str, default='caltech101', help='choose dataset.', choices=['caltech101', 'cifar100', 'cifar10'])

parser.add_argument('--K', type=int, default=4, help='multi head for attention layer')
parser.add_argument('--arch', type=str, default='VFENet', help='model type')
parser.add_argument('--pooling', type=str, default='netvlad', help='type of pooling to use', choices=['netvlad', 'wpca', 'avg', 'cls', 'branch'])

parser.add_argument('-lr', '--learning_rate', default=0.05, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--lr_decay_ratio', default=0.4, type=float, help='learning rate decay factor')
parser.add_argument('-epochs', '--no_epochs', default=300, type=int, metavar='epochs', help='no. epochs')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--nesterov', default=False, type=bool, help='yesterov?')
parser.add_argument('--weight_decay', '--wd', default=0.0005, type=float, metavar='W', help='weight decay')
parser.add_argument('--epoch_step', default='[40,70,100,130,160,190]', type=str, help='json list with epochs to drop lr on')

def get_error(output, target, topk=(1,)):
    """Computes the error@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    
    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
        res.append(100.0 - correct_k.mul_(100.0 / batch_size))
    return res
    
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    model_out_path = join(opt.savePath, filename)
    torch.save(state, model_out_path)
    if is_best:
        shutil.copyfile(model_out_path, join(opt.savePath, 'model_best.pth.tar'))

def train(epoch):
    epoch_loss, aerr1, aerr5 = 0, 0, 0
    index = 0
    lenth = len(train_loader)
    model.train()
    for i_batch, data in enumerate(train_loader):
    #for i_batch, (rgb, label) in enumerate(train_loader):
        rgb, event, label = data['inp1'].to(device), data['inp2'].to(device), data['label'].to(device)
        #rgb, event, label = rgb.to(device), rgb.to(device), label.to(device)
        optimizer.zero_grad()
        loss = 0
        rgb_encoded = model.encode_layer(rgb)
        event_encoded = model.encode_layer(event)
        frame_encoded, event_encoded = rgb_encoded.reshape(rgb_encoded.shape[0], rgb_encoded.shape[1], rgb_encoded.shape[2]* rgb_encoded.shape[3]), event_encoded.reshape(event_encoded.shape[0], event_encoded.shape[1], event_encoded.shape[2]* event_encoded.shape[3])
        if opt.data == 've': 
            frame_encoded, event_encoded = model.samp_layer(frame_encoded), model.samp_layer(event_encoded)
            x_vis, x_event = model.fusion_layer(frame_encoded, event_encoded, useCrossAtt=True, useSelfAtt=True)
            fusion_vlad = model.ouput_layer(x_vis, x_event, out_layer = 'fc6_pred')
        elif opt.data == 'ov':
            frame_encoded, event_encoded = model.samp_layer(frame_encoded), model.samp_layer(frame_encoded)
            x_vis, x_event = model.fusion_layer(frame_encoded, event_encoded, useCrossAtt=True, useSelfAtt=True)
            fusion_vlad = model.ouput_layer(x_vis, x_event, out_layer = 'fc6_pred')
        elif opt.data == 'oe':
            frame_encoded, event_encoded = model.samp_layer(event_encoded), model.samp_layer(event_encoded)
            x_vis, x_event = model.fusion_layer(frame_encoded, event_encoded, useCrossAtt=True, useSelfAtt=True)
            fusion_vlad = model.ouput_layer(x_vis, x_event, out_layer = 'fc6_pred')
        loss = criterion(fusion_vlad, label.long())
        epoch_loss += loss.item()
        loss = loss.to(device)
        loss.backward()    
        optimizer.step()

        if(index % opt.printFrequence == 0):
            print('Epoch[{}][{}/{}]:current loss is {}'.format(epoch, index, lenth, loss.item()))
            writer.add_scalar('Train/Loss', loss.item(), ((epoch-1)*lenth + index))
        err1, err5 = get_error(fusion_vlad.detach(), label.long(), topk=(1, 5))
        aerr1+= err1.item()
        aerr5+= err5.item()
        index +=1
    print('Avg Err1 = {}, Avg Err5 = {}'.format(aerr1/ index, aerr5 / index))      
    return
    
def test(epoch, write_tboard=False):
    model.eval()
    aerr1, aerr5 = 0, 0
    index = 0
    with torch.no_grad():
        for i_batch, data in enumerate(test_loader):
        #for i_batch, (rgb, label) in enumerate(test_loader):
            rgb, event, label = data['inp1'].to(device), data['inp2'].to(device), data['label'].to(device)
            #rgb, event, label = rgb.to(device), rgb.to(device), label.to(device)
            rgb_encoded = model.encode_layer(rgb)
            event_encoded = model.encode_layer(event)
            frame_encoded, event_encoded = rgb_encoded.reshape(rgb_encoded.shape[0], rgb_encoded.shape[1], rgb_encoded.shape[2]* rgb_encoded.shape[3]), event_encoded.reshape(event_encoded.shape[0], event_encoded.shape[1], event_encoded.shape[2]* event_encoded.shape[3])
            if opt.data == 've': 
                frame_encoded, event_encoded = model.samp_layer(frame_encoded), model.samp_layer(event_encoded)
                x_vis, x_event = model.fusion_layer(frame_encoded, event_encoded, useCrossAtt=True, useSelfAtt=True)
                fusion_vlad = model.ouput_layer(x_vis, x_event, out_layer = 'fc6_pred')
            elif opt.data == 'ov':
                frame_encoded, event_encoded = model.samp_layer(frame_encoded), model.samp_layer(frame_encoded)
                x_vis, x_event = model.fusion_layer(frame_encoded, event_encoded, useCrossAtt=True, useSelfAtt=True)
                fusion_vlad = model.ouput_layer(x_vis, x_event, out_layer = 'fc6_pred')
            elif opt.data == 'oe':
                frame_encoded, event_encoded = model.samp_layer(event_encoded), model.samp_layer(event_encoded)
                x_vis, x_event = model.fusion_layer(frame_encoded, event_encoded, useCrossAtt=True, useSelfAtt=True)
                fusion_vlad = model.ouput_layer(x_vis, x_event, out_layer = 'fc6_pred')
            loss = criterion(fusion_vlad, label.long())

            err1, err5 = get_error(fusion_vlad.detach(), label.long(), topk=(1, 5))
            aerr1+= err1.item()
            aerr5+= err5.item()
            index +=1
        print('Avg Err1 = {}, Avg Err5 = {}'.format(aerr1/ index, aerr5/ index))      
    return aerr1/ index         
    

if __name__ == '__main__':
    opt = parser.parse_args()
    cuda = opt.cuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run with --nocuda")
    device = torch.device("cuda" if cuda else "cpu")
    if cuda:
        torch.cuda.set_device(opt.oc)
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)
        torch.cuda.set_device(opt.oc)
    
    isParallel = False
    if opt.nGPU > 1 and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        isParallel = True
 
    print('===> Building model')
    model = VFENet.VFENet(num_class = 100, pooling=opt.pooling)
    print('===> Loading dataset')
    if opt.dataset_name == 'caltech101':
        import dataloading_caltech101 as dataset
        whole_dataset, test_set = dataset.get_dataset()
        train_loader = DataLoader(dataset=whole_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.threads, drop_last=False, pin_memory=cuda)
        test_loader = DataLoader(dataset=test_set, batch_size=opt.batch_size, shuffle=False, num_workers=opt.threads, drop_last=False, pin_memory=cuda)
    elif opt.dataset_name == 'cifar100':
        MEAN =  (0.5071, 0.4867, 0.4408)
        STD =   (0.2675, 0.2565, 0.2761)
        NO_CLASSES = 100
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(256),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(MEAN,STD),
        ])

        transform_val = transforms.Compose([
            transforms.Resize([256,256]),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
        trainset = torchvision.datasets.CIFAR100(root='/mnt/data/', train=True, download=True, transform=transform_train)
        valset = torchvision.datasets.CIFAR100(root='/mnt/data/', train=False, download=True, transform=transform_val)
        train_loader = DataLoader(trainset, batch_size=opt.batch_size, num_workers=opt.threads, pin_memory=cuda, shuffle=True)
        test_loader = DataLoader(valset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.threads, pin_memory=cuda)
        
    if not opt.resume:
        model = model.to(device)
         
    if opt.mode.lower() == 'train':
        optimizer = torch.optim.SGD(model.parameters(),
                                lr=opt.learning_rate, momentum=opt.momentum, weight_decay=opt.weight_decay,
                                nesterov=opt.nesterov)
        epoch_step = json.loads(opt.epoch_step)
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=epoch_step, gamma=opt.lr_decay_ratio)
        
    criterion = nn.CrossEntropyLoss()
    if opt.resume:
        if opt.ckpt.lower() == 'latest':
            resume_ckpt = join(opt.resume, 'checkpoints', 'checkpoint.pth.tar')
        elif opt.ckpt.lower() == 'best':
            resume_ckpt = join(opt.resume, 'checkpoints', 'model_best.pth.tar')

        if isfile(resume_ckpt):
            print("=> loading checkpoint '{}'".format(resume_ckpt))
            checkpoint = torch.load(resume_ckpt, map_location=lambda storage, loc: storage)
            #opt.start_epoch = checkpoint['epoch']
            best_metric = checkpoint['best_score']
            model.load_state_dict(checkpoint['state_dict'], strict = False)
            model = model.to(device)
            if opt.mode == 'train':
                #optimizer.load_state_dict(checkpoint['optimizer'])
                print('train')
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(resume_ckpt, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(resume_ckpt))
            
    if opt.mode.lower() == 'test':
        print('===> Running evaluation step')
        epoch = 1
        recalls = test(epoch, write_tboard=False)
        
    elif opt.mode.lower() == 'train':
        print('===> Training model')
        writer = SummaryWriter(log_dir=join(opt.runsPath, opt.dataset_name+'_'+datetime.now().strftime('%b%d_%H-%M-%S')+'_'+opt.arch+'_'+opt.data+'_K'+str(opt.K)+'_bs'+str(opt.batch_size)))
        
        # write checkpoints in logdir
        logdir = writer.file_writer.get_logdir()
        opt.savePath = join(logdir, opt.savePath)
        #if not opt.resume:
        makedirs(opt.savePath)

        with open(join(opt.savePath, 'flags.json'), 'w') as f:
            f.write(json.dumps(
                {k:v for k,v in vars(opt).items()}
                ))
        print('===> Saving state to:', logdir)

        not_improved = 0
        best_score = 100    
        
        for epoch in range(opt.start_epoch+1, opt.nEpochs + 1):
            train(epoch)
            scheduler.step(epoch)
            if (epoch % opt.evalEvery) == 0:
                el = test(epoch, write_tboard=False)
                is_best = el < best_score 
                if is_best:
                    not_improved = 0
                    best_score = el
                else: 
                    not_improved += 0

                save_checkpoint({
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'best_score': best_score,
                        'optimizer' : optimizer.state_dict(),
                        'parallel' : isParallel,
                }, is_best)

                if opt.patience > 0 and not_improved > (opt.patience / opt.evalEvery):
                    print('Performance did not improve for', opt.patience, 'epochs. Stopping.')
                    break

        print("=> Best Err@1: {:.4f}".format(best_score), flush=True)
        writer.close()
            

