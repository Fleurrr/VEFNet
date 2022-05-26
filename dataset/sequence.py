import numpy as np
import torch
import cv2
import os
import glob
from pathlib import Path
from torch.utils.data import Dataset


traverse_to_name = {
    'dvs_vpr_2020-04-21-17-03-03': 'sunset1',
    'dvs_vpr_2020-04-22-17-24-21': 'sunset2',
    'dvs_vpr_2020-04-24-15-12-03': 'daytime',
    'dvs_vpr_2020-04-28-09-14-11': 'morning',
    'dvs_vpr_2020-04-29-06-20-23': 'sunrise',
    'dvs_vpr_2020-04-27-18-13-29': 'night'
}

name_to_consumervideo = {
    'sunset1': '20200421_170039-sunset1',
    'sunset2': '20200422_172431-sunset2',
    'daytime': '20200424_151015-daytime',
    'night': '20200427_181204-night',
    'morning': '20200428_091154-morning',
    'sunrise': '20200429_061912-sunrise'
}

video_beginning = {
    'sunset1': 1587452582.35,
    'sunset2': 1587540271.65,
    'daytime': 1587705130.80,
    'morning': 1588029265.73,
    'sunrise': 1588105232.91,
    'night': 1587975221.10
}


class Sequence(Dataset):
    def __init__(self, seq_path: Path):
        event_frame_dir = seq_path / 'reconstruction' / 'events'
        rec_frame_dir = seq_path / 'reconstruction'
        rgb_frame_dir = seq_path / 'frames'
        rec_timestamp = rec_frame_dir / 'timestamps.txt'
    
        self.timestamps = np.loadtxt(rec_timestamp, dtype='str')
        
        self.events = glob.glob(os.path.join(event_frame_dir, '*.png'))
        self.events.sort()
        
        self.recs = glob.glob(os.path.join(rec_frame_dir, '*.png'))
        self.recs.sort()
        
        self.rgbs = glob.glob(os.path.join(rgb_frame_dir, '*.png'))
        self.rgbs.sort()
        
        self.gps_data = str(seq_path)[:-27] + name_to_consumervideo[traverse_to_name[str(seq_path)[-27:]]] + '_concat.nmea'  
        
    def get_frame(self, filepath):
        frame = cv2.imread(str(filepath)) #(260,346,3)
        return frame
        
    def get_gps_data(self):
        return self.gps_data
        
    def get_whole_frames(self):
        return self.recs,self.events,self.rgbs
        
    def __len__(self):
        return len(self.timestamps)
        
    def __getitem__(self, index):
        return self.recs[index], self.events[index], self.get_frame(self.recs[index]), self.get_frame(self.events[index])     
    
    
        
        