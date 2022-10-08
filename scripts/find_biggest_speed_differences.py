import os
import sys
import numpy as np
from os import path
from scipy.spatial.distance import cdist

sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
from src.utils import (
    get_gps_speed,
    load_event_streams,
    sync_event_streams,
    get_short_traverse_name,
    get_gps,
    print_duration,
    interpolate_gps,
    interpolate_gps_speed,
    interpolate_gps_distance,
    get_images_at_start_times,
    print_distance
)
from src.constants import brisbane_event_traverses, path_to_gps_files

def get_speed_information(traversal):
    gps_gt_speed = []
    if os.path.isfile(path_to_gps_files + get_short_traverse_name(brisbane_event_traverses[traversal]) + ".nmea"):
        print("Adding GPS")
        gps_gt_speed.append(get_gps_speed(path_to_gps_files + get_short_traverse_name(brisbane_event_traverses[traversal]) + ".nmea"))
    
    return gps_gt_speed

def get_dist_and_coords_information(traversal):
    gps_gt= []
    if os.path.isfile(path_to_gps_files + get_short_traverse_name(brisbane_event_traverses[traversal]) + ".nmea"):
        print("Adding GPS")
        gps_gt.append(get_gps(path_to_gps_files + get_short_traverse_name(brisbane_event_traverses[traversal]) + ".nmea"))

    return gps_gt

# speed_and_time_sunset1 = np.array(get_speed_information(0)[0])
# speed_and_time_daytime = np.array(get_speed_information(2)[0])

speed_sunset1 = get_speed_information(0)[0][:,0]
speed_daytime = get_speed_information(2)[0][:,0]

# speed_diffs = np.subtract(speed_sunset1, speed_daytime)
# top_speed_diffs = np.argsort(speed_diffs)[-10:]

print(speed_sunset1)

dist_sunset1 = get_dist_and_coords_information(0)[0][:,3]
dist_daytime = get_dist_and_coords_information(0)[0][:,3]

print(dist_sunset1)





# # Get GPS distance and time rows 
# gps_distance = gps_gt[0][:,2:4]
# new_row = [0,0]
# gps_distance = np.vstack([new_row,gps_distance])
# print(gps_distance)

# # Get times when we want the gps distance 
# times = event_stream['t'].to_numpy()
# desired_times = np.subtract(times, times[0])*0.000001
# print(desired_times.shape)

# print(speed_sunset1)
# print(speed_daytime)