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
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.data.dataset import Subset

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import numpy as np
import argparse
from datetime import datetime
from tensorboardX import SummaryWriter
from tqdm import tqdm
import scipy
from scipy.spatial import distance
from math import log10, ceil
import cv2

import h5py
import faiss

import model.VFENet_tran as VFENet
import model.seqNet as seqNet

from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as tf
np.set_printoptions(threshold=np.inf)

parser = argparse.ArgumentParser(description='VFENet')
#basic
parser.add_argument('--mode', type=str, default='train', help='Mode', choices=['train', 'test', 'draw', 'cluster'])
parser.add_argument('--data', type=str, default='ve', help='data mode', choices=['ve','ov','oe'])
parser.add_argument('--resume', type=str, default='', help='Path to load checkpoint from, for resuming training or testing.')
parser.add_argument('--ckpt', type=str, default='best', help='Resume from latest or best checkpoint.', choices=['latest', 'best'])
parser.add_argument('--runsPath', type=str, default='/home/hz/Workspace/EventVPR/vidFe/', help='Path to save runs to.')
parser.add_argument('--savePath', type=str, default='checkpoints', help='Path to save checkpoints to in logdir. Default=checkpoints/')
#train
parser.add_argument('--nGPU', type=int, default=1, help='number of GPU to use.')
parser.add_argument('--oc', type=int, default=0, help='cuda number')
parser.add_argument('--cuda', action='store_true', help='use cuda')
parser.add_argument('--seed', type=int, default=123, help='Random seed to use.')
parser.add_argument('--patience', type=int, default=15, help='Patience for early stopping. 0 is off.')
parser.add_argument('--evalEvery', type=int, default=1, help='Do a validation set run, and save, every N epochs.')
parser.add_argument('--printFrequence', type=int, default=10, help='the frequence of printing data to the screen')
parser.add_argument('--nEpochs', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
#model
parser.add_argument('--K', type=int, default=4, help='multi head for attention layer')
parser.add_argument('--arch', type=str, default='VFENet', help='model type')
parser.add_argument('--pooling', type=str, default='netvlad', help='type of pooling to use', choices=['netvlad', 'wpca', 'avg', 'cls', 'branch'])
#data
parser.add_argument('--batch_size', type=int, default=16,  help='Number of triplets (query, pos, negs)*2. Each triplet consists of 6 images.')
parser.add_argument('--threads', type=int, default=8, help='Number of threads for each data loader to use')
#for dataset
parser.add_argument('--dataset_name', type=str, default='Brisbane', help='choose dataset.', choices=['Brisbane', 'MVSEC'])
parser.add_argument('--nNeg', type=int, default=5, help='Number of negative samples')
parser.add_argument('--dataset_folder', type=str, default='/mnt/data/Event-VPR/', help='Path for dataset.')
parser.add_argument('--query_traverse', type=str, default='dvs_vpr_2020-04-21-17-03-03', help='Name for Brisbane query set.')
parser.add_argument('--reference_traverse', type=list,default=['dvs_vpr_2020-04-29-06-20-23', 'dvs_vpr_2020-04-28-09-14-11', 'dvs_vpr_2020-04-24-15-12-03'], help='name for Brisbane reference set' )
parser.add_argument('--test_query_traverse', type=str, default='dvs_vpr_2020-04-22-17-24-21', help='Name for Brisbane test query set')
parser.add_argument('--test_reference_traverse', type=list, default=['dvs_vpr_2020-04-28-09-14-11', 'dvs_vpr_2020-04-21-17-03-03', 'dvs_vpr_2020-04-29-06-20-23',  'dvs_vpr_2020-04-24-15-12-03'], help='Name for Brisbane test query set')
parser.add_argument('--ignore',type=str, default='reconstruction', help='ignored type', choices=['reconstruction', 'reconstruction/events', 'frames'])

parser.add_argument('--subset_train', type=str, default='whole_relative_train.txt', help='Name for mvsec train set.')
parser.add_argument('--subset_test', type=str, default='GT_files_day2_part.txt', help='Name for mvsec test set.')
#loss function
parser.add_argument('--margin', type=float, default=0.1, help='Margin for triplet loss. Default=0.1')
#optimizer set
parser.add_argument('--optim', type=str, default='SGD', help='optimizer to use', choices=['SGD', 'ADAM'])
parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate.')
parser.add_argument('--lrStep', type=float, default=10, help='Decay LR ever N steps.')
parser.add_argument('--lrGamma', type=float, default=0.5, help='Multiply LR by Gamma for decaying.')
parser.add_argument('--weightDecay', type=float, default=0.001, help='Weight decay for SGD.')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD.')
#netvlad
parser.add_argument('--num_clusters', type=int, default=32, help='Number of NetVlad clusters. Default=64')
parser.add_argument('--cacheBatchSize', type=int, default=24, help='Batch size for caching and testing')
parser.add_argument('--cacheRefreshRate', type=int, default=1000, help='How often to refresh cache, in number of queries. 0 for off')
parser.add_argument('--dataPath', type=str, default='/mnt/data/Event-VPR/centroids/', help='Path for centroid data.')

def show_double_img(query, positive):
    imghstack = np.hstack((np.array(query), np.array(positive)))
    cv2.namedWindow('idt')
    cv2.imshow('idt', imghstack)
    cv2.waitKey(0)
    return

class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, input):
        return F.normalize(input, p=2, dim=self.dim)
        
def get_clusters(cluster_set):
    nDescriptors = 2000
    nPerImage = 10
    encoder_dim = 512
    nIm = ceil(nDescriptors/nPerImage)

    sampler = SubsetRandomSampler(np.random.choice(len(cluster_set), nIm, replace=False))
    data_loader = DataLoader(dataset=cluster_set, 
                num_workers=opt.threads, batch_size=opt.cacheBatchSize, shuffle=False, 
                pin_memory=cuda,
                sampler=sampler)

    if not exists(join(opt.dataPath, 'centroids')):
        makedirs(join(opt.dataPath, 'centroids'))

    initcache = join(opt.dataPath, 'centroids', opt.arch + '_' + opt.dataset_name + '_' + str(opt.num_clusters) + '_desc_cen.hdf5')
    with h5py.File(initcache, mode='w') as h5: 
        with torch.no_grad():
            model.eval()
            print('====> Extracting Descriptors')
            dbFeat = h5.create_dataset("descriptors", 
                        [nDescriptors, encoder_dim], 
                        dtype=np.float32)

            for iteration, data in enumerate(data_loader, 1):
                frame, event = data['1q'].to(device), data['2q'].to(device)
                if opt.data == 've':
                    frame_encoded = model.encode_layer(frame)
                    event_encoded = model.encode_layer(event)
                    frame_encoded = frame_encoded.reshape(frame_encoded.shape[0], frame_encoded.shape[1], frame_encoded.shape[2]* frame_encoded.shape[3])
                    event_encoded = event_encoded.reshape(event_encoded.shape[0], event_encoded.shape[1], event_encoded.shape[2]* event_encoded.shape[3])     
                    frame_encoded, event_encoded = model.samp_layer(frame_encoded), model.samp_layer(event_encoded)
                    x_vis, x_event = model.fusion_layer(frame_encoded, event_encoded, useCrossAtt=True, useSelfAtt=True)               
                    image_descriptors = model.ouput_layer(x_vis, x_event).view(frame_encoded.size(0), encoder_dim, -1).permute(0, 2, 1)

                batchix = (iteration-1)*opt.cacheBatchSize*nPerImage
                for ix in range(image_descriptors.size(0)):
                    # sample different location for each image in batch
                    sample = np.random.choice(image_descriptors.size(1), nPerImage, replace=False)
                    startix = batchix + ix*nPerImage
                    dbFeat[startix:startix+nPerImage, :] = image_descriptors[ix, sample, :].detach().cpu().numpy()

                if iteration % 50 == 0 or len(data_loader) <= 10:
                    print("==> Batch ({}/{})".format(iteration, 
                        ceil(nIm/opt.cacheBatchSize)), flush=True)
                del image_descriptors
        
        print('====> Clustering..')
        niter = 100
        kmeans = faiss.Kmeans(encoder_dim, opt.num_clusters, niter=niter, verbose=False)
        kmeans.train(dbFeat[...])

        print('====> Storing centroids', kmeans.centroids.shape)
        h5.create_dataset('centroids', data=kmeans.centroids)
        print('====> Done!')

def train(epoch):
    epoch_loss = 0
    index = 0
    lenth = len(train_loader)
    model.train()
    for data in train_loader:
        frame_q , frame_p, frame_n = data['1q'].to(device), data['1p'].to(device), data['1n'].to(device)
        event_q , event_p, event_n = data['2q'].to(device), data['2p'].to(device), data['2n'].to(device)
        nNeg = opt.nNeg
        B, C, H, W = frame_q.shape[0], frame_q.shape[1], frame_q.shape[2], frame_q.shape[3]
        
        optimizer.zero_grad()
        loss = 0
        frame_encoded = model.encode_layer(torch.cat([frame_q, frame_p, frame_n.reshape(B*nNeg, C, H, W)], axis=0))
        event_encoded = model.encode_layer(torch.cat([event_q, event_p, event_n.reshape(B*nNeg, C, H, W)], axis=0))
        
        #frame_out, event_out = model.samp_layer_s(frame_encoded, event_encoded)
        #frame_out, event_out = frame_out.squeeze(3).squeeze(2), event_out.squeeze(3).squeeze(2)
        
        frame_encoded = frame_encoded.reshape(frame_encoded.shape[0], frame_encoded.shape[1], frame_encoded.shape[2]* frame_encoded.shape[3])
        event_encoded = event_encoded.reshape(event_encoded.shape[0], event_encoded.shape[1], event_encoded.shape[2]* event_encoded.shape[3])
        if opt.data == 've': 
            frame_encoded, event_encoded = model.samp_layer(frame_encoded), model.samp_layer(event_encoded)
            x_vis, x_event = model.fusion_layer(frame_encoded, event_encoded, useCrossAtt=False, useSelfAtt=False)
            fusion_vlad = model.ouput_layer(x_vis, x_event, out_layer = 'fc6_wpca')
        elif opt.data == 'ov':
            frame_encoded, event_encoded = model.samp_layer(frame_encoded), model.samp_layer(frame_encoded)
            x_vis, x_event = model.fusion_layer(frame_encoded, event_encoded, useCrossAtt=True, useSelfAtt=True)
            fusion_vlad = model.ouput_layer(x_vis, x_event, out_layer = 'fc6_wpca')
        elif opt.data == 'oe':
            frame_encoded, event_encoded = model.samp_layer(event_encoded), model.samp_layer(event_encoded)
            x_vis, x_event = model.fusion_layer(frame_encoded, event_encoded, useCrossAtt=True, useSelfAtt=True)
            fusion_vlad = model.ouput_layer(x_vis, x_event, out_layer = 'fc6_wpca')
        for idb in range(B):
            for idn in range(nNeg):
                loss += criterion(fusion_vlad[idb].unsqueeze(0), fusion_vlad[idb + B].unsqueeze(0), fusion_vlad[2*B + idb*nNeg + idn].unsqueeze(0))
                #loss += 0.5* criterion(frame_out[idb].unsqueeze(0), frame_out[idb + B].unsqueeze(0), frame_out[2*B + idb*nNeg + idn].unsqueeze(0))
                #loss += 0.5* criterion(event_out[idb].unsqueeze(0), event_out[idb + B].unsqueeze(0), event_out[2*B + idb*nNeg + idn].unsqueeze(0))        
        loss = loss/ float(nNeg)
        loss = loss.to(device)
        loss.backward()    
        optimizer.step()
        
        if(index % opt.printFrequence == 0):
            print('Epoch[{}][{}/{}]:current loss is {}'.format(epoch, index, lenth, loss.item()))
            writer.add_scalar('Train/Loss', loss.item(), ((epoch-1)*lenth + index))
        index +=1
        
def test(epoch=0, write_tboard=False, plot=True):
    def input_transform():
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
        ])
    def get_fusion_vlad(images_all_combined_set_q):
        model.eval()
        features_all_combined_q = list()
        trans = input_transform()
        with torch.no_grad():
            print('====> Evaluating')
            for idx in tqdm(range(len(images_all_combined_set_q['frames']))):
                #data_f = torch.from_numpy(images_all_combined_set_q['frames'][idx]).float().unsqueeze(0).permute(0,3,1,2).to(device)
                #data_e = torch.from_numpy(images_all_combined_set_q['reconstruction/events'][idx]).float().unsqueeze(0).permute(0,3,1,2).to(device)
                data_f = trans(Image.fromarray(images_all_combined_set_q['frames'][idx])).unsqueeze(0).to(device)
                data_e = trans(Image.fromarray(images_all_combined_set_q['reconstruction/events'][idx])).unsqueeze(0).to(device)
                data_f_encoded = model.encode_layer(data_f)
                data_e_encoded = model.encode_layer(data_e)
                frame_encoded = data_f_encoded.reshape(data_f_encoded.shape[0], data_f_encoded.shape[1], data_f_encoded.shape[2]*data_f_encoded.shape[3])
                event_encoded = data_e_encoded.reshape(data_e_encoded.shape[0], data_e_encoded.shape[1], data_e_encoded.shape[2]*data_e_encoded.shape[3])
                if opt.data == 've':
                    frame_encoded, event_encoded = model.samp_layer(frame_encoded), model.samp_layer(event_encoded)
                    x_vis, x_event = model.fusion_layer(frame_encoded, event_encoded, useCrossAtt=False, useSelfAtt=False)
                    fusion_vlad = model.ouput_layer(x_vis, x_event, out_layer = 'fc6_wpca')
                elif opt.data == 'ov':
                    frame_encoded, event_encoded = model.samp_layer(frame_encoded), model.samp_layer(frame_encoded)
                    x_vis, x_event = model.fusion_layer(frame_encoded, event_encoded, useCrossAtt=True, useSelfAtt=True)
                    fusion_vlad = model.ouput_layer(x_vis, x_event, out_layer = 'fc6_wpca')
                elif opt.data == 'oe':
                    frame_encoded, event_encoded = model.samp_layer(event_encoded), model.samp_layer(event_encoded)
                    x_vis, x_event = model.fusion_layer(frame_encoded, event_encoded, useCrossAtt=True, useSelfAtt=True)
                    fusion_vlad = model.ouput_layer(x_vis, x_event, out_layer = 'fc6_wpca')                             
                features_all_combined_q.append(fusion_vlad.squeeze(0).cpu().numpy())
        return features_all_combined_q
    
    if opt.dataset_name == 'Brisbane':
        n_values_all = {'1':0, '5':0, '10':0, '20':0}
        for i in range(len(opt.test_reference_traverse)):
            trt = opt.test_reference_traverse[i]  
            print('====> Evaluating {} vs {}'.format(opt.test_query_traverse, trt))                   
            images_all_combined_set_q, images_all_combined_set_r = dataset.get_test_dataset(opt.dataset_folder, opt.test_query_traverse, trt)
            features_all_combined_q = get_fusion_vlad(images_all_combined_set_q)
            features_all_combined_r = get_fusion_vlad(images_all_combined_set_r)
            lenf = len(features_all_combined_q)
            print('====> Building dist matrix')
            dist_matrix_all_combined = scipy.spatial.distance.cdist(features_all_combined_q, features_all_combined_r, 'cosine').T  
            recalls_combined, tps_combined, best_matches_combined = get_recall_helper(np.array(dist_matrix_all_combined))

            print('====> Calculating recall @ N')
            match_index = np.argsort(dist_matrix_all_combined)
            name = trt + '_vfenet.txt'
            np.savetxt(name, match_index, fmt="%d")
            n_values = {'1':0, '5':0, '10':0, '20':0}
            for idx in range(lenf):
                for idr in range(20):
                    if(match_index[idx][idr] == idx):
                        #print(idx)                           
                        #print(idr)
                        #show_double_img(images_all_combined_set_q['frames'][idx], images_all_combined_set_r['frames'][match_index[idx][0]])
                        #show_double_img(images_all_combined_set_q['reconstruction/events'][idx], images_all_combined_set_r['reconstruction/events'][match_index[idx][0]])
                        if idr < 1:
                            n_values['1']+=1
                            n_values['5']+=1
                            n_values['10']+=1
                            n_values['20']+=1
                            continue
                        elif idr < 5:
                            n_values['5']+=1
                            n_values['10']+=1
                            n_values['20']+=1
                            continue
                        elif idr < 10:
                            n_values['10']+=1
                            n_values['20']+=1
                            continue
                        elif idr < 20:
                            n_values['20']+=1
                            continue
            n_values['1'] = n_values['1'] / lenf
            n_values['5'] = n_values['5'] / lenf
            n_values['10'] = n_values['10'] / lenf
            n_values['20'] = n_values['20'] / lenf
            n_values_all['1'] += n_values['1']
            n_values_all['5'] += n_values['5']
            n_values_all['10'] += n_values['10']
            n_values_all['20'] += n_values['20']
            print(n_values)
        print('====> Calculating avg recall @ N')
        n_values_all['1'] = n_values_all['1'] / len(opt.test_reference_traverse)
        n_values_all['5'] = n_values_all['5'] / len(opt.test_reference_traverse)
        n_values_all['10'] = n_values_all['10'] / len(opt.test_reference_traverse)
        n_values_all['20'] = n_values_all['20'] / len(opt.test_reference_traverse)
        print(n_values_all)        
            
    
    elif opt.dataset_name == 'MVSEC':  
        images_all_combined_set_q, images_all_combined_set_r = dataset.Generate_MVSEC_VPR_test(opt.subset_test)
        features_all_combined_q = get_fusion_vlad(images_all_combined_set_q)
        features_all_combined_r = get_fusion_vlad(images_all_combined_set_r)
        lenf = len(features_all_combined_q)
        print('====> Building dist matrix')
        dist_matrix_all_combined = scipy.spatial.distance.cdist(features_all_combined_q, features_all_combined_r, 'cosine').T  
        recalls_combined, tps_combined, best_matches_combined = get_recall_helper(np.array(dist_matrix_all_combined))

        print('====> Calculating recall @ N')
        match_index = np.argsort(dist_matrix_all_combined)
        n_values = {'1':0, '5':0, '10':0, '20':0}
        for idx in range(lenf):
            for idr in range(20):
                if(match_index[idx][idr] == idx):
                    if idr < 1:
                        n_values['1']+=1
                        n_values['5']+=1
                        n_values['10']+=1
                        n_values['20']+=1
                        continue
                    elif idr < 5:
                        n_values['5']+=1
                        n_values['10']+=1
                        n_values['20']+=1
                        continue
                    elif idr < 10:
                        n_values['10']+=1
                        n_values['20']+=1
                        continue
                    elif idr < 20:
                        n_values['20']+=1
                        continue
            n_values['1'] = n_values['1'] / lenf
            n_values['5'] = n_values['5'] / lenf
            n_values['10'] = n_values['10'] / lenf
            n_values['20'] = n_values['20'] / lenf        
        n_values_all = n_values      
        print(n_values_all)                       
    return n_values_all
    
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    model_out_path = join(opt.savePath, filename)
    torch.save(state, model_out_path)
    if is_best:
        shutil.copyfile(model_out_path, join(opt.savePath, 'model_best.pth.tar'))

if __name__ == "__main__":
    opt = parser.parse_args()
    cuda = opt.cuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run with --nocuda")

    device = torch.device("cuda" if cuda else "cpu")
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)             
        torch.cuda.set_device(opt.oc)
    print('===> Building model')
    
    model = VFENet.VFENet(num_class = 100, pooling = opt.pooling, mode = opt.mode)

    if opt.mode.lower() != 'cluster':
        if opt.pooling.lower() == 'netvlad':
            if not opt.resume or 1: 
                if opt.mode.lower() == 'train':
                    initcache = join(opt.dataPath, 'centroids', opt.arch + '_' + opt.dataset_name + '_' + str(opt.num_clusters) +'_desc_cen.hdf5')
                else:
                    initcache = join(opt.dataPath, 'centroids', opt.arch + '_' + opt.dataset_name + '_' + str(opt.num_clusters) +'_desc_cen.hdf5')

                if not exists(initcache):
                    raise FileNotFoundError('Could not find clusters, please run with --mode=cluster before proceeding')
                print('===> Init parameter for netvlad')
                with h5py.File(initcache, mode='r') as h5: 
                    clsts = h5.get("centroids")[...]
                    traindescs = h5.get("descriptors")[...]
                    model.net_vlad.init_params(clsts, traindescs) 
                    del clsts, traindescs
            
    isParallel = False
    if opt.nGPU > 1 and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        isParallel = True 
        
    print('===> Loading dataset')
    if opt.dataset_name == 'Brisbane':
        import dataloading_brisbane_v2 as dataset
        whole_dataset = dataset.Generate_from_BrisbanVPR(opt.dataset_folder, opt.query_traverse, opt.reference_traverse, opt.ignore, nNeg = opt.nNeg)
        train_loader = DataLoader(dataset=whole_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.threads, drop_last=False, pin_memory=cuda)
    elif opt.dataset_name == 'MVSEC':
        import dataloading_mvsec as dataset
        whole_dataset = dataset.Generate_MVSEC_VPR_train(opt.subset_train)
        train_loader = DataLoader(dataset=whole_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.threads, drop_last=False, pin_memory=cuda)
    
    if not opt.resume:
        model = model.to(device) 
    if opt.mode.lower() == 'train':
        if opt.optim.upper() == 'ADAM':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, 
                model.parameters()), lr=opt.lr)#, betas=(0,0.9))
        elif opt.optim.upper() == 'SGD':
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, 
                model.parameters()), lr=opt.lr,
                momentum=opt.momentum,
                weight_decay=opt.weightDecay)

            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=opt.lrStep, gamma=opt.lrGamma)
        else:
            raise ValueError('Unknown optimizer: ' + opt.optim)

        # original paper/code doesn't sqrt() the distances, we do, so sqrt() the margin, I think :D
        criterion = nn.TripletMarginLoss(margin=opt.margin**0.5, p=2, reduction='sum').to(device)   
        
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
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            model = model.to(device)
            if opt.mode == 'train':
                #optimizer.load_state_dict(checkpoint['optimizer'])
                print('training!')
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(resume_ckpt, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(resume_ckpt))

    if opt.mode.lower() == 'test':
        print('===> Running evaluation step')
        epoch = 1
        recalls = test(epoch, write_tboard=False, plot=True)
    elif opt.mode.lower() == 'cluster':
        print('===> Calculating descriptors and clusters')
        get_clusters(whole_dataset)
    elif opt.mode.lower() == 'train':
        print('===> Training model')
        writer = SummaryWriter(log_dir=join(opt.runsPath, opt.dataset_name+'_'+datetime.now().strftime('%b%d_%H-%M-%S')+'_'+opt.arch+'_'+opt.data+'_K'+str(opt.K)+'_bs'+str(opt.batch_size)))

        # write checkpoints in logdir
        logdir = writer.file_writer.get_logdir()
        opt.savePath = join(logdir, opt.savePath)
        makedirs(opt.savePath)

        with open(join(opt.savePath, 'flags.json'), 'w') as f:
            f.write(json.dumps(
                {k:v for k,v in vars(opt).items()}
                ))
        print('===> Saving state to:', logdir)

        not_improved = 0
        best_score = 0
        for epoch in range(opt.start_epoch+1, opt.nEpochs + 1):
            train(epoch)
            if opt.optim.upper() == 'SGD':
                scheduler.step(epoch)
            if (epoch % opt.evalEvery) == 0:
                recalls = test(epoch, write_tboard=False, plot=False)
                is_best = recalls['5'] > best_score 
                if is_best:
                    not_improved = 0
                    best_score = recalls['5']
                else: 
                    not_improved += 1

                save_checkpoint({
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'recalls': recalls,
                        'best_score': best_score,
                        'optimizer' : optimizer.state_dict(),
                        'parallel' : isParallel,
                }, is_best)

                if opt.patience > 0 and not_improved > (opt.patience / opt.evalEvery):
                    print('Performance did not improve for', opt.patience, 'epochs. Stopping.')
                    break

        print("=> Best Recall@5: {:.4f}".format(best_score), flush=True)
        writer.close()
            
