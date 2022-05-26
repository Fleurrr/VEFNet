import numpy as np
from tools.correspondence_event_camera_frame_camera import traverse_to_name, name_to_consumervideo, video_beginning
from os import listdir, symlink
from os.path import join, isfile, basename, abspath
from pathlib import Path

def get_image_paths(folder1):
    return sorted([join(folder1, f) for f in listdir(folder1) if f.endswith('.png')])

def get_correspondence_info(x1, x2, query_traverse, reference_traverse):
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
    tsg1 = np.array([t + video_beginning[traverse_to_name[query_traverse]] for t in t_raw1])
    tsg2 = np.array([t + video_beginning[traverse_to_name[reference_traverse]] for t in t_raw2])
    return match_x1_to_x2, tsg1, tsg2
        