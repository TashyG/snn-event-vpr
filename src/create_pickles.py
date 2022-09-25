from fileinput import close
import re
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
import os

import lava.lib.dl.slayer as slayer
import torch
from torch.utils.data import Dataset, DataLoader

import IPython.display as display
from matplotlib import animation
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.spatial.distance import cdist

from utils import (
    get_gps_speed,
    load_event_streams,
    sync_event_streams,
    get_short_traverse_name,
    get_gps,
    print_duration,
    interpolate_gps,
    interpolate_gps_speed,
    interpolate_gps_distance,
    get_images_at_start_times
)
from constants import brisbane_event_traverses, path_to_gps_files, path_to_pickles


for traverse in [4]:
    gps_gt= []
    if os.path.isfile(path_to_gps_files + get_short_traverse_name(brisbane_event_traverses[traverse]) + ".nmea"):
        tqdm.write("Adding GPS")
        gps_gt.append(get_gps(path_to_gps_files + get_short_traverse_name(brisbane_event_traverses[traverse]) + ".nmea"))

    event_streams = load_event_streams([brisbane_event_traverses[traverse]])
    event_streams = sync_event_streams(event_streams, [brisbane_event_traverses[traverse]], gps_gt)
    event_stream  = event_streams[0]
    print_duration(event_stream)

    total_x_pixels = 346
    total_y_pixels = 260
    subselect_num = 34

    x_space = total_x_pixels/subselect_num
    y_space = total_y_pixels/subselect_num

    # Subselect 34 x 34 pixels evenly spaced out - Create a filter
    x_select  = [int(i*x_space + x_space/2) for i in range(subselect_num)]
    y_select  = [int(i*y_space + y_space/2) for i in range(subselect_num)]


    filter0x = event_stream['x'].isin(x_select)
    filter0y = event_stream['y'].isin(y_select)

    # Apply the filters
    event_stream = event_stream[filter0x & filter0y]

    # Now reset values to be between 0 and 33
    for i, x in zip(range(34), x_select):
        small_filt0x  = event_stream['x'].isin([x])
        event_stream['x'].loc[small_filt0x] = i

    for i, y in zip(range(34), y_select):
        small_filt0y  = event_stream['y'].isin([y])
        event_stream['y'].loc[small_filt0y] = i


    # Get GPS distance and time rows 
    gps_distance = gps_gt[0][:,2:4]
    new_row = [0,0]
    gps_distance = np.vstack([new_row,gps_distance])
    print(gps_distance)

    # Get times when we want the gps distance 
    times = event_stream['t'].to_numpy()
    desired_times = np.subtract(times, times[0])*0.000001
    print(desired_times.shape)

    # Interpolate gps distance for desired times
    gps_distances_interpolated = interpolate_gps_distance(gps_distance, desired_times)[:,0].T
    print(gps_distances_interpolated)

    # Add to dataframe
    event_stream = event_stream.assign(distance=gps_distances_interpolated)
    print(event_stream)

    # Pickle for later use
    event_stream.to_pickle(path_to_pickles + get_short_traverse_name(brisbane_event_traverses[traverse]) + ".pkl")

