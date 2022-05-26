import torch
import os
import numpy as np
from pathlib import Path
from sequence import Sequence

from torch.utils.data import DataLoader
from tqdm import tqdm
from read_gps import get_gps

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

consumervideo_to_name = {
    '20200421_170039-sunset1':'sunset1', 
    '20200422_172431-sunset2':'sunset2',
    '20200424_151015-daytime':'daytime', 
    '20200427_181204-night':'night',
    '20200428_091154-morning':'morning',
    '20200429_061912-sunrise':'sunrise'
}

video_beginning = {
    'sunset1': 1587452582.35,
    'sunset2': 1587540271.65,
    'daytime': 1587705130.80,
    'morning': 1588029265.73,
    'sunrise': 1588105232.91,
    'night': 1587975221.10
}


class DatasetProvider:
    def __init__(self, dataset_path: Path):
        sequences_name = list()
        sequences_data = list()
        for child in dataset_path.iterdir():
            if os.path.isdir(child):
                sequences_name.append(child)
                sequences_data.append(Sequence(child))  
        self.dataset_path = str(dataset_path)        
        self.db_dir = sequences_name
        self.db_data = sequences_data
    
    def get_correspondence_info(self, x1, x2, query_traverse, reference_traverse):
        match_x1_to_x2 = []
        for idx1, (latlon, t) in enumerate(zip(x1[:, 0:2], x1[:, 2])):
            if len(match_x1_to_x2) < 6:
                min_idx2 = 0
                max_idx2 = int(0.25 * len(x2))
            elif idx1 > 0.5 * len(x1):
                min_idx2 = match_x1_to_x2[-5]
                max_idx2 = len(x2)
            else:
                min_idx2 = match_x1_to_x2[-5]
                max_idx2 = int(0.75 * len(x2))
            best_match = (np.linalg.norm(x2[min_idx2:max_idx2, 0:2] - latlon, axis=1)).argmin() + min_idx2
            match_x1_to_x2.append(best_match)
        match_x1_to_x2 = np.array(match_x1_to_x2)
    
        t_raw1 = x1[:, 2]
        t_raw2 = x2[match_x1_to_x2, 2]
        timestamps_gps1 = np.array([t + video_beginning[consumervideo_to_name[query_traverse]] for t in t_raw1])
        timestamps_gps2 = np.array([t + video_beginning[consumervideo_to_name[reference_traverse]] for t in t_raw2])
        return match_x1_to_x2, timestamps_gps1, timestamps_gps2
    
    def get_whole_db_dirname(self):
        return self.db_dir
    
    def get_whole_db_data(self):
        seqlist =  list()
        match_list = list()
        timestamp_list = list()
        event_list = list()
        rec_list = list()
        for idx in range(len(self.db_data)):
            if self.db_data[idx].get_gps_data()[-35: -12] == '20200421_170039-sunset1':
                seq_base = self.db_data[idx].get_gps_data()
                rec_base = self.db_data[idx].get_whole_frames()[0]
                event_base = self.db_data[idx].get_whole_frames()[1]
            else:
                seqlist.append(self.db_data[idx].get_gps_data())  
                data_frames = self.db_data[idx].get_whole_frames()
                rec_list.append(data_frames[0])
                event_list.append(data_frames[1])
                              
        for idx in range(len(seqlist)):
            match_x1_to_x2, timestamps_gps1, timestamps_gps2 = self.get_correspondence_info(get_gps(seq_base), get_gps(seqlist[idx]), seq_base[len(self.dataset_path)+1:-12], seqlist[idx][len(self.dataset_path)+1:-12])
            match_list.append(match_x1_to_x2)
            timestamp_list.append(timestamps_gps2)
        
        
        
            
               
        return torch.utils.data.ConcatDataset(self.db_data)
        
    def get_train_dataset(self):
        raise NotImplementedError

    def get_val_dataset(self):
        raise NotImplementedError

    def get_test_dataset(self):
        raise NotImplementedError
        
if __name__ == "__main__":
    dsec_dir = '/mnt/data/Event-VPR/'
    dataset_provider = DatasetProvider(Path(dsec_dir))
    db_dirname = dataset_provider.get_whole_db_dirname()
    db_data = dataset_provider.get_whole_db_data()
    
    batch_size = 8
    num_workers = 4
    train_loader = DataLoader(dataset=db_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)   
    
        
