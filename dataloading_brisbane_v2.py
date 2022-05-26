import numpy as np
import cv2
import os
import random
from os import listdir, symlink
from os.path import join, isfile, basename, abspath
from pathlib import Path
from tqdm import tqdm

from tools.correspondence_event_camera_frame_camera import traverse_to_name, name_to_consumervideo, video_beginning
from tools.read_gps import get_gps 
from initial_setup import get_correspondence_info, get_image_paths
from tools.code_helpers_public import compare_images, get_timestamps, get_timestamp_matches, get_image_sets_on_demand, get_vlad_features 

import torch
from torch.utils.data import DataLoader

from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as tf

def input_transform():
    return transforms.Compose([
        #transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
    ])             

class Fusion_VPR_Dataset(object):
    def __init__(self, object, count, ignore, combine, nNeg, use='v', input_transform = input_transform()):
        self.whole_dataset = object
        self.mapping = list()
        self.lenth = len(self.whole_dataset)
        self.count = count
        self.margin = 10
        for i in range(self.lenth):
            for j in range(i+1, self.lenth):
                self.mapping.append([i,j])
        self.ignore = ignore
        self.combine = combine
        self.use = use
        self.transform = input_transform
        self.nNeg = nNeg
                
    def select_negative(self, idx, idy):
        neg_idx_list, neg_idy_list = list(), list()
        for i in range(self.nNeg):
            choose = True
            while(choose):
                neg_idx = random.randint(0,self.count-1)
                if(abs(neg_idx - idx) >= self.margin and neg_idx not in neg_idx_list):
                    choose = False
                    neg_idx_list.append(neg_idx)
            neg_idy = random.randint(0, self.lenth-1)
            neg_idy_list.append(neg_idy)    
        return neg_idx_list, neg_idy_list
        
    def get_id(self, index):
        idy = index % int((self.lenth)*(self.lenth-1)/2)
        index = (index - idy) // int((self.lenth)*(self.lenth-1)/2)
        if(index == 0):
            neg_idx, neg_idy = self.select_negative(0, idy)
            return 0, idy, neg_idx, neg_idy
        else:
            idx = index % self.count
            neg_idx, neg_idy = self.select_negative(idx, idy)
            return idx, idy, neg_idx, neg_idy                      

    def get_triplet_slice(self, idx, idy, neg_idx, neg_idy, idt):
        return_list = list()
        neg_list = list()
        for child in self.whole_dataset:
            return_list.append(self.whole_dataset[child][idt][idx])
            #neg_list.append(self.whole_dataset[child][idt][neg_idx])
            for i in range(len(neg_idx)):
                neg_list.append(self.whole_dataset[child][idt][neg_idx[i]])
        for j in range(len(neg_idy)):
            if j == 0:
                neg_out = self.transform(Image.fromarray(neg_list[j* (self.lenth-1) + neg_idy[j]])).unsqueeze(0)  
            else:
                neg_out = torch.cat((neg_out, self.transform(Image.fromarray(neg_list[j* (self.lenth-1) + neg_idy[j]])).unsqueeze(0)), axis=0)       
        return self.transform(Image.fromarray(return_list[self.mapping[idy][0]])), self.transform(Image.fromarray(return_list[self.mapping[idy][1]])), neg_out
        
    def show_triplet_frame(self, query, positive, negative, idt):
        imghstack = np.hstack((np.array(query), np.array(positive), np.array(negative)))
        cv2.namedWindow(idt)
        cv2.imshow(idt, imghstack)
        cv2.waitKey(0)
        return
        
    def __len__(self):
        return self.count*int((self.lenth)*(self.lenth-1)/2)
    
    def __getitem__(self, index):
        idx, idy, neg_idx, neg_idy = self.get_id(index)
        query_f, positive_f, negative_f = self.get_triplet_slice(idx, idy, neg_idx, neg_idy, 'frames')
        query_e, positive_e, negative_e = self.get_triplet_slice(idx, idy, neg_idx, neg_idy, 'reconstruction/events')
        query_r, positive_r, negative_r = self.get_triplet_slice(idx, idy, neg_idx, neg_idy, 'reconstruction')

        if self.combine:
            if self.ignore == 'frames':
                sample = {'1q':query_r, '1p':positive_r, '1n':negative_r, '2q':query_e, '2p':positive_e, '2n':negative_e, 'nNeg': self.nNeg}
            elif self.ignore == 'reconstruction/events':
                sample = {'1q':query_f, '1p':positive_f, '1n':negative_f, '2q':query_r, '2p':positive_r, '2n':negative_r, 'nNeg': self.nNeg}
            elif self.ignore == 'reconstruction':
                sample = {'1q':query_f, '1p':positive_f, '1n':negative_f, '2q':query_e, '2p':positive_e, '2n':negative_e, 'nNeg': self.nNeg}
        return sample
            
            
def get_empty_list(reference_traverse):
    empty_list = {}
    for name in reference_traverse:
        empty_list[traverse_to_name[name]] = {}
    return empty_list
        
def Generate_from_BrisbanVPR(dataset_folder, query_traverse, reference_traverse, ignore, combine=True, use='v', nNeg = 1):
    vid_path_1 = join(dataset_folder, name_to_consumervideo[traverse_to_name[query_traverse]])
    vid_path_2 = list()
    for idx in range(len(reference_traverse)):
        vid_path_2.append(join(dataset_folder, name_to_consumervideo[traverse_to_name[reference_traverse[idx]]]))

    frames_subfolder = 'frames'    
    event_subfolder = 'reconstruction/events'
    rec_subfolder = 'reconstruction'
    all_subfolders = [frames_subfolder] + [rec_subfolder] + [event_subfolder]

    image_paths_combined1 = {}
    image_paths_combined2 = get_empty_list(reference_traverse)

    for subfolder in all_subfolders:
        image_paths_combined1[subfolder] = get_image_paths(join(dataset_folder, query_traverse, subfolder))
    
    for name in reference_traverse:
        for subfolder in all_subfolders:
            image_paths_combined2[traverse_to_name[name]][subfolder] = get_image_paths(join(dataset_folder, name, subfolder))
        
    timestamps_combined1 = {}
    timestamps_combined2 = get_empty_list(reference_traverse)

    for subfolder in all_subfolders:
        if subfolder == 'frames':
            timestamps_combined1[subfolder] = np.array([float(os.path.splitext(basename(f))[0]) for f in image_paths_combined1[subfolder]])
        else:
            timestamps_combined1[subfolder]= get_timestamps(join(dataset_folder, query_traverse, 'reconstruction'))
        
    for name in reference_traverse:
        for subfolder in all_subfolders:
            if subfolder == 'frames':
                timestamps_combined2[traverse_to_name[name]][subfolder] = np.array([float(os.path.splitext(basename(f))[0]) for f in image_paths_combined2[traverse_to_name[name]][subfolder]])
            else:
                timestamps_combined2[traverse_to_name[name]][subfolder]= get_timestamps(join(dataset_folder, name, 'reconstruction'))
            
    x1 = get_gps(vid_path_1 + '_concat.nmea')
    x2 = vid_path_2
    for idx in range(len(vid_path_2)):
        x2[idx] = get_gps(vid_path_2[idx] + '_concat.nmea')

    match_x1_to_x2 = get_empty_list(reference_traverse)
    timestamps_gps1 = {}
    timestamps_gps2 = get_empty_list(reference_traverse)
    for idx in range(len(x2)):
        match, tsg1, tsg2 = get_correspondence_info(x1, x2[idx], query_traverse, reference_traverse[idx])
        match_x1_to_x2[traverse_to_name[reference_traverse[idx]]] = match
        timestamps_gps1 = tsg1
        timestamps_gps2[traverse_to_name[reference_traverse[idx]]] = tsg2
    
    images_all_combined_set1 = {}
    images_all_combined_set2 = get_empty_list(reference_traverse)
    matches_fixedlength_combined1 = {}
    matches_fixedlength_combined2 = get_empty_list(reference_traverse)
    
    for subfolder in all_subfolders:
        matches_fixedlength_combined1[subfolder] = get_timestamp_matches(timestamps_combined1[subfolder], timestamps_gps1)
        images_all_combined_set1[subfolder] = get_image_sets_on_demand(image_paths_combined1[subfolder], matches_fixedlength_combined1[subfolder])
    for name in reference_traverse:
        for subfolder in all_subfolders:
            matches_fixedlength_combined2[traverse_to_name[name]][subfolder] = get_timestamp_matches(timestamps_combined2[traverse_to_name[name]][subfolder], timestamps_gps2[traverse_to_name[name]])
            images_all_combined_set2[traverse_to_name[name]][subfolder] = get_image_sets_on_demand(image_paths_combined2[traverse_to_name[name]][subfolder], matches_fixedlength_combined2[traverse_to_name[name]][subfolder])
        
    images_all_combined_set2[traverse_to_name[query_traverse]] = images_all_combined_set1 #The lenth of all sets are 557
    count = len(images_all_combined_set2[traverse_to_name[query_traverse]]['frames'])

    whole_dataset = Fusion_VPR_Dataset(images_all_combined_set2, count, ignore, combine, nNeg, use)
    return whole_dataset
    
def get_test_dataset(dataset_folder, query_traverse, reference_traverse):
    vid_path_1 = join(dataset_folder, name_to_consumervideo[traverse_to_name[query_traverse]])
    vid_path_2 = join(dataset_folder, name_to_consumervideo[traverse_to_name[reference_traverse]])

    frames_subfolder = 'frames'    
    event_subfolder = 'reconstruction/events'
    all_subfolders = [frames_subfolder] + [event_subfolder]

    image_paths_combined1 = {}
    image_paths_combined2 = {}

    for subfolder in all_subfolders:
        image_paths_combined1[subfolder] = get_image_paths(join(dataset_folder, query_traverse, subfolder))
        image_paths_combined2[subfolder] = get_image_paths(join(dataset_folder, reference_traverse, subfolder))
        
    timestamps_combined1 = {}
    timestamps_combined2 = {}

    for subfolder in all_subfolders:
        if subfolder == 'frames':
            timestamps_combined1[subfolder] = np.array([float(os.path.splitext(basename(f))[0]) for f in image_paths_combined1[subfolder]])
            timestamps_combined2[subfolder] = np.array([float(os.path.splitext(basename(f))[0]) for f in image_paths_combined2[subfolder]])
        else:
            timestamps_combined1[subfolder]= get_timestamps(join(dataset_folder, query_traverse, 'reconstruction'))
            timestamps_combined2[subfolder]= get_timestamps(join(dataset_folder, reference_traverse, 'reconstruction'))
           
    x1 = get_gps(vid_path_1 + '_concat.nmea')
    x2 = get_gps(vid_path_2 + '_concat.nmea')

    match_x1_to_x2 = {}
    timestamps_gps1 = {}
    timestamps_gps2 = {}
    
    match, tsg1, tsg2 = get_correspondence_info(x1, x2, query_traverse, reference_traverse)
    match_x1_to_x2[traverse_to_name[reference_traverse]] = match
    timestamps_gps1 = tsg1
    timestamps_gps2 = tsg2
    
    images_all_combined_set1 = {}
    images_all_combined_set2 = {}
    matches_fixedlength_combined1 = {}
    matches_fixedlength_combined2 = {}
    
    for subfolder in all_subfolders:
        matches_fixedlength_combined1[subfolder] = get_timestamp_matches(timestamps_combined1[subfolder], timestamps_gps1)
        matches_fixedlength_combined2[subfolder] = get_timestamp_matches(timestamps_combined2[subfolder], timestamps_gps2)
        images_all_combined_set1[subfolder] = get_image_sets_on_demand(image_paths_combined1[subfolder], matches_fixedlength_combined1[subfolder])
        images_all_combined_set2[subfolder] = get_image_sets_on_demand(image_paths_combined2[subfolder], matches_fixedlength_combined2[subfolder])
    return images_all_combined_set1, images_all_combined_set2
    

if __name__ == "__main__":
    dataset_folder = '/mnt/data/Event-VPR/'
    test_query_traverse = 'dvs_vpr_2020-04-22-17-24-21'
    test_reference_traverse = ['dvs_vpr_2020-04-21-17-03-03', 'dvs_vpr_2020-04-29-06-20-23', 'dvs_vpr_2020-04-28-09-14-11', 'dvs_vpr_2020-04-24-15-12-03']
    for i in range(len(test_reference_traverse)):
        trt = test_reference_traverse[i]
        dirname = test_query_traverse + '_vs_' + trt
        dirname_q = test_query_traverse + '_vs_' + trt + '/query/'
        dirname_r = test_query_traverse + '_vs_' + trt + '/reference/'
        whole_path = dataset_folder + dirname
        query_path = dataset_folder + dirname_q
        reference_path = dataset_folder + dirname_r
        if(os.path.exists(whole_path) == False):
            os.makedirs(whole_path)
            os.makedirs(query_path)
            os.makedirs(reference_path)
        images_all_combined_set_q, images_all_combined_set_r = get_test_dataset(dataset_folder, test_query_traverse, trt)
        imgname = os.path.join(query_path, '{:0>5d}.png')
        for idx in range(len(images_all_combined_set_q['frames'])):
            img = images_all_combined_set_q['frames'][idx]
            cv2.imwrite(imgname.format(idx), img)
        imgname = os.path.join(reference_path, '{:0>5d}.png')
        for idx in range(len(images_all_combined_set_r['frames'])):
            img = images_all_combined_set_r['frames'][idx]
            cv2.imwrite(imgname.format(idx), img)
    
    