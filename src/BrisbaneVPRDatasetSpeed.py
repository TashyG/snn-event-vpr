# Code adapted  from https://lava-nc.org/lava-lib-dl/slayer/notebooks/nmnist/train.html

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
    get_images_at_start_times,
    print_distance
)
from constants import brisbane_event_traverses, path_to_gps_files, path_to_pickles

  
# train_traverse = brisbane_event_traverses[0]
# test_traverse = brisbane_event_traverses[3]

def chopData(event_stream, start_seconds, end_seconds, max_spikes):
    """
    Gets a specific time window of event data out of an event stream

    :param start_seconds: The start time of the window
    :param end_seconds: The end time of the window
    :param max_spikes: The maximum number of spikes that can be in the window
    :return: The event data within the specified time window
    """ 

    stream_start_time  = event_stream['t'].iloc[0]

    chop_start = stream_start_time + start_seconds*1000000
    chop_end = stream_start_time + end_seconds*1000000 -1

    btwn = event_stream['t'].between(chop_start, chop_end, inclusive='both')
    chopped_stream = event_stream[btwn]

    chopped_stream['t'] -= chop_start

    # Crop the data to the specified number of spikes
    if max_spikes != None:
        chopped_stream = chopped_stream.iloc[0:max_spikes]

    return chopped_stream


def filter_and_divide_training(event_stream, x_select, y_select, num_places, start_time, place_gap, place_duration, max_spikes):
    """
    Filters a long event stream to have only select pixels in the data and
    then divides the stream up into multiple substreams

    :param event_stream: The stream of events, a DataFrame object 
    :param x_select: The x pixel positions to be included (all others are fitlered out)
    :param y_select: The y pixel positions to be included (all others are filtered out)
    :param num_places: The number of place samples to extract (starting from the start of the event stream)
    :param start_time: Time at which to start extracting place samples in seconds
    :param place_gap: The time gap in seconds between the start of each place sample
    :param place_duration: The duration of each place sample in seconds
    :param max_spikes: The maximum number of spikes that can be in a place sample
    :return: The filtered substreams/place samples & The start times of each place sample
    """ 

    # Subselect 34 x 34 pixels evenly spaced out - Create the filters
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

    # Divide the test stream into 2 second windows
    sub_streams = []
    start_times = []
    for i in range(0,num_places):
        sub_streams.append(chopData(event_stream, start_time + i*place_gap, start_time + i*place_gap + place_duration, max_spikes))
        start_times.append((start_time + i*place_gap))

    return sub_streams, start_times

def filter_and_divide_testing(event_stream, x_select, y_select, start_times, place_duration, max_spikes):
    """
    Filters a long event stream to have only select pixels in the data and
    then divides the stream up into multiple substreams

    :param event_stream: The stream of events, a DataFrame object 
    :param x_select: The x pixel positions to be included (all others are fitlered out)
    :param y_select: The y pixel positions to be included (all others are filtered out)
    :param start_times: Times at which to start extracting place samples in seconds
    :param place_duration: The duration of each place sample in seconds
    :param max_spikes: The maximum number of spikes that can be in a place sample
    :return: The filtered substreams/place samples
    """ 

    # Subselect 34 x 34 pixels evenly spaced out - Create the filters
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

    # Divide the test stream into 2 second windows
    sub_streams = []
    for start_time in start_times:
        sub_streams.append(chopData(event_stream, start_time, start_time + place_duration, max_spikes))

    return sub_streams

def find_closest_matches(training_locations, test_gps_data):
    """
    Finds the closest matching GPS locations (and their corresponding times) in the 
    testing data to the selected GPS locations in the training data

    :param training_locations: the lat and long coords of the chosen places in the training data 
    :param test_gps_data: the gps data of the test dataset 
    :return: the times of the estimated closest matches and their gps coords
    """ 

    # Separate test gps coordinates and times
    test_locations = test_gps_data[:, :2]
    test_times = test_gps_data[:,2]

    # Find the two closest test gps locations and their corresponding times for each training location
    distance_matrix = cdist(training_locations, test_locations)
    closest_inds = np.argsort(distance_matrix,axis=1)[:,:2]
    closest_ranges = test_times[closest_inds]
    
    # Search between the closest gps locations for a more accurate match
    times = []
    closest_test_locs = []
    for closest_range, training_location in zip(closest_ranges, training_locations):
        # Make sure earlier time is at the start
        closest_range = np.sort(closest_range)

        # Break down the time range found into smaller time steps
        locs_within_range = interpolate_gps(test_gps_data, np.arange(closest_range[0], closest_range[1]+0.01, 0.01))[:,:2]

        # Find the closest location match out of the smaller time steps and its corresponing time
        distance_matrix = cdist([training_location], locs_within_range)
        closest_match = np.argsort(distance_matrix,axis=1)[:,:1][0][0]
        time_of_closest_match = closest_range[0] + closest_match*0.01
        loc_of_closest_match = locs_within_range[closest_match]
        times.append(time_of_closest_match)
        closest_test_locs.append(loc_of_closest_match)

    return times, np.array(closest_test_locs)


def chopDataDistTime(event_stream, start_distance, place_duration, max_spikes):
    """
    Gets a specific time window of event data out of an event stream

    :param start_seconds: The start time of the window
    :param end_seconds: The end time of the window
    :param max_spikes: The maximum number of spikes that can be in the window
    :return: The event data within the specified time window
    """ 

    stream_start_time  = event_stream['t'].iloc[0]

    # Get closest time for the starting distance
    time_at_start_dist = event_stream.loc[event_stream['distance'] >= start_distance]['t'].iloc[0]

    chop_start = time_at_start_dist
    chop_end = chop_start + place_duration*1000000 -1

    btwn = event_stream['t'].between(chop_start, chop_end, inclusive='both')
    chopped_stream = event_stream[btwn]

    chopped_stream['t'] -= chop_start

    # Crop the data to the specified number of spikes
    if max_spikes != None:
        chopped_stream = chopped_stream.iloc[0:max_spikes]

    return chopped_stream, (chop_start - stream_start_time)/1000000

def chopDataByDist(event_stream, start_distance, end_distance, max_spikes):
    """
    Gets a specific time window of event data out of an event stream

    :param start_seconds: The start time of the window
    :param end_seconds: The end time of the window
    :param max_spikes: The maximum number of spikes that can be in the window
    :return: The event data within the specified time window
    """ 

    stream_start_time  = event_stream['t'].iloc[0]

    # Get closest time for the starting distance
    time_at_start_dist = event_stream.loc[event_stream['distance'] >= start_distance]['t'].iloc[0]

    btwn = event_stream['distance'].between(start_distance, end_distance, inclusive='both')
    chopped_stream = event_stream[btwn]

    chopped_stream['distance'] -= start_distance

    # Crop the data to the specified number of spikes
    if max_spikes != None:
        chopped_stream = chopped_stream.iloc[0:max_spikes]

    return chopped_stream, (time_at_start_dist - stream_start_time)/1000000

def chopDataTimeDist(event_stream, start_time, place_length, max_spikes):
    """
    Gets a specific time window of event data out of an event stream

    :param start_seconds: The start time of the window
    :param end_seconds: The end time of the window
    :param max_spikes: The maximum number of spikes that can be in the window
    :return: The event data within the specified time window
    """ 

    stream_start_time  = event_stream['t'].iloc[0]

    # Get closest time for the starting distance
    dist_at_start_time = event_stream.loc[event_stream['t'] >= (stream_start_time + start_time*1000000)]['distance'].iloc[0]
    end_distance = dist_at_start_time + place_length

    btwn = event_stream['distance'].between(dist_at_start_time, end_distance, inclusive='both')
    chopped_stream = event_stream[btwn]

    chopped_stream['distance'] -= dist_at_start_time

    # Crop the data to the specified number of spikes
    if max_spikes != None:
        chopped_stream = chopped_stream.iloc[0:max_spikes]

    return chopped_stream


def divide_training(event_stream, num_places, start_dist, place_gap, place_length, max_spikes):

    # Divide the test stream into 2 second windows
    sub_streams = []
    start_times = []
    times = []
    for i in range(0,num_places):
        sub_stream, start_time = chopDataByDist(event_stream, start_dist + i*place_gap, start_dist + i*place_gap + place_length, max_spikes)
        print("Place: " + str(i))
        print_distance(sub_stream)
        print_duration(sub_stream)
        sub_streams.append(sub_stream)
        start_times.append(start_time)
        times.append(sub_stream.iloc[-1]["t"] - sub_stream.iloc[0]["t"])
    return sub_streams, start_times, times

def divide_testing(event_stream, start_times, place_length, max_spikes):

    # Divide the test stream into 2 second windows
    sub_streams = []
    times = []
    counter = 0
    for start_time in start_times:
        sub_stream = chopDataTimeDist(event_stream, start_time, place_length, max_spikes)
        print("Place: " + str(counter))
        counter+=1
        print_distance(sub_stream)
        print_duration(sub_stream)
        sub_streams.append(sub_stream)
        times.append(sub_stream.iloc[-1]["t"] - sub_stream.iloc[0]["t"])
    return sub_streams, times


class BrisbaneVPRDatasetSpeed(Dataset):
    """NMNIST dataset method
    Parameters
    ----------
    train : bool, optional
        train/test flag, by default True
    sampling_time : int, optional
        sampling time of event data, by default 2
    stream_length : int, optional
        the length in seconds of traversal that you want to use
    transform : None or lambda or fx-ptr, optional
        transformation method. None means no transform. By default None.
    """
    def __init__(
        self,
        traverse_name,
        train=True,
        training_locations=None, 
        sampling_time=1, 
        samples_per_sec = 1000,
        num_places = 45,
        start_dist = 200,
        place_gap=100, 
        place_length= 25,
        max_spikes=None,
        transform=None,
    ):
        super(BrisbaneVPRDatasetSpeed, self).__init__()

        if train:
            print("Loading training event streams ...")

            # Get GPS long/lat data associated with chosen training stream
            gps_gt = []
            if os.path.isfile(path_to_gps_files + get_short_traverse_name(traverse_name) + ".nmea"):
                tqdm.write("Adding GPS")
                gps_gt.append(get_gps(path_to_gps_files + get_short_traverse_name(traverse_name) + ".nmea"))

            # Load the training stream itself and synchronise 
            event_stream = pd.read_pickle(path_to_pickles + get_short_traverse_name(traverse_name) + ".pkl")
            print_duration(event_stream)
            print_distance(event_stream)

            # Get the place samples 
            sub_streams, start_times, times = divide_training(event_stream, num_places, start_dist, place_gap, place_length, max_spikes)
            
            # # Get the interpolated reference gps locations at each start time
            self.place_locations = interpolate_gps(gps_gt[0], start_times)[:, :2]
            

            # # Get the closest CMOS images at each start time
            self.place_images = get_images_at_start_times(start_times, traverse_name)

            # # Get the speeds at each start time each 
            # self.place_speeds = interpolate_gps_speed(gps_gt_speed[0], start_times)[:, :2]
            # print(self.place_speeds)

            self.samples = sub_streams
            print("The number of training substreams is: " + str(len(self.samples)))
            
            
        else:
            assert training_locations is not None, "Must provide training locations"

            # Get GPS data associated with chosen testing stream
            print("Loading testing event streams ...")
            gps_gt = []
            if os.path.isfile(path_to_gps_files + get_short_traverse_name(traverse_name) + ".nmea"):
                tqdm.write("Adding GPS")
                gps_gt.append(get_gps(path_to_gps_files + get_short_traverse_name(traverse_name) + ".nmea"))


            # Estimate the closest locations in test dataset to training dataset and get their start_times 
            start_times, self.place_locations = find_closest_matches(training_locations, gps_gt[0])

            # Get the closest CMOS images at each start time
            self.place_images = get_images_at_start_times(start_times, traverse_name)

            # Get the speeds at each start time each 
            # self.place_speeds = interpolate_gps_speed(gps_gt_speed[0], start_times)[:, :2]
            # print(self.place_speeds)

            # Load the test stream and synchronise
            event_stream = pd.read_pickle(path_to_pickles + get_short_traverse_name(traverse_name) + ".pkl")
            print_duration(event_stream)
            print_distance(event_stream)

            # Get the place samples 
            sub_streams, times = divide_testing(event_stream, start_times, place_length, max_spikes)
            
            self.samples = sub_streams
            print("The number of testing substreams is: " + str(len(self.samples)))

        self.place_duration = place_length # The duration of a place in seconds 
        self.place_gap = place_gap # The time between each place window
        self.num_places = num_places # the number of places
        self.sampling_time = sampling_time # Default sampling time is 1
        self.samples_per_sec = samples_per_sec
        self.transform = transform
        #self.num_time_bins = int(sample_length/sampling_time)
       

    def __getitem__(self, i):
        
        # make sure we aren't calling an index out of range
        assert i < self.num_places, "Index out of range! There are not that many place samples"

        # Find the place label
        label = int(i % (self.num_places))

        # Find the number of time bins
        num_dist_bins = 3000 #int(self.place_duration*self.samples_per_sec)
        #time_divider = int(1000000/self.samples_per_sec)

        # Turn the sample stream into events
        x_event = self.samples[i]['x'].to_numpy()
        y_event = self.samples[i]['y'].to_numpy()
        c_event = self.samples[i]['p'].to_numpy()
        d_event = self.samples[i]['distance'].to_numpy()
        #event = slayer.io.Event(x_event, y_event, c_event, t_event/1000)
        event = slayer.io.Event(x_event, y_event, c_event, d_event*(1000/9))

        # Transform event
        if self.transform is not None:
            event = self.transform(event)

        # Turn the events into a tensor 
        spike = event.fill_tensor(
                torch.zeros(2, 34, 34, num_dist_bins),
                sampling_time=self.sampling_time,
            )
        
        return spike.reshape(-1, num_dist_bins), label

    def __len__(self):
        return len(self.samples)


# gps_gt_speed = []
# if os.path.isfile(path_to_gps_files + get_short_traverse_name(brisbane_event_traverses[1]) + ".nmea"):
#     tqdm.write("Adding GPS")
#     gps_gt_speed.append(get_gps_speed(path_to_gps_files + get_short_traverse_name(brisbane_event_traverses[1]) + ".nmea"))

# desired_times = [50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 260, 270, 280, 290, 300, 310, 320, 330]
# gps_speed = interpolate_gps_speed(gps_gt_speed[0], desired_times)

# print(gps_speed)


# gps_gt= []
# if os.path.isfile(path_to_gps_files + get_short_traverse_name(brisbane_event_traverses[0]) + ".nmea"):
#     tqdm.write("Adding GPS")
#     gps_gt.append(get_gps(path_to_gps_files + get_short_traverse_name(brisbane_event_traverses[0]) + ".nmea"))

# print(gps_gt[0])

# event_streams = load_event_streams([brisbane_event_traverses[0]])
# event_streams = sync_event_streams(event_streams, [brisbane_event_traverses[0]], gps_gt)
# print_duration(event_streams[0])
# event_stream  = event_streams[0]

# total_x_pixels = 346
# total_y_pixels = 260
# subselect_num = 34

# x_space = total_x_pixels/subselect_num
# y_space = total_y_pixels/subselect_num

# # Subselect 34 x 34 pixels evenly spaced out - Create a filter
# x_select  = [int(i*x_space + x_space/2) for i in range(subselect_num)]
# y_select  = [int(i*y_space + y_space/2) for i in range(subselect_num)]


# filter0x = event_stream['x'].isin(x_select)
# filter0y = event_stream['y'].isin(y_select)

# # Apply the filters
# event_stream = event_stream[filter0x & filter0y]

# # Now reset values to be between 0 and 33
# for i, x in zip(range(34), x_select):
#     small_filt0x  = event_stream['x'].isin([x])
#     event_stream['x'].loc[small_filt0x] = i

# for i, y in zip(range(34), y_select):
#     small_filt0y  = event_stream['y'].isin([y])
#     event_stream['y'].loc[small_filt0y] = i

# times = event_stream['t'].to_numpy()


# # GPS only distance
# gps_distance = gps_gt[0][:,2:4]
# new_row = [0,0]
# gps_distance = np.vstack([new_row,gps_distance])
# print(gps_distance)
# desired_times = np.subtract(times, times[0])*0.000001
# print(desired_times.shape)

# gps_distances_interpolated = interpolate_gps_distance(gps_distance, desired_times)[:,0].T
# print(gps_distances_interpolated)

# event_stream = event_stream.assign(distance=gps_distances_interpolated)

# event_stream.to_pickle("my_data.pkl")

# event_stream = pd.read_pickle("my_data.pkl")

# print(event_stream)

# gps_speed = interpolate_gps_speed(gps_gt[0])[:668,:1].T
# gps_coords = interpolate_gps(gps_gt_normal[0])[:,:2].T

# x_tr, y_tr = gps_coords
# plt.scatter(x_tr, y_tr, c=gps_speed, cmap='viridis')

# plt.savefig(os.path.dirname(os.path.abspath(__file__)) + "/../results/scatter4.png")

# gps_gt = []
# if os.path.isfile(path_to_gps_files + get_short_traverse_name(brisbane_event_traverses[2]) + ".nmea"):
#     tqdm.write("Adding GPS")
#     gps_gt.append(get_gps_speed(path_to_gps_files + get_short_traverse_name(brisbane_event_traverses[2]) + ".nmea"))

# gps_gt_normal = []
# if os.path.isfile(path_to_gps_files + get_short_traverse_name(brisbane_event_traverses[2]) + ".nmea"):
#     tqdm.write("Adding GPS")
#     gps_gt_normal.append(get_gps(path_to_gps_files + get_short_traverse_name(brisbane_event_traverses[2]) + ".nmea"))

# gps_speed = interpolate_gps_speed(gps_gt[0])[:701,:1].T
# gps_coords = interpolate_gps(gps_gt_normal[0])[:,:2].T

# x_tr, y_tr = gps_coords
# plt.scatter(x_tr, y_tr, c=gps_speed, cmap='viridis')

# plt.savefig(os.path.dirname(os.path.abspath(__file__)) + "/../results/scatter2.png")

# start_times = [0, 10, 20, 30, 40]
# image_paths = get_images_at_start_times(start_times, train_traverse)
# print(image_paths)

# Ultimate test- loading the data
# train_traverse = brisbane_event_traverses[0]
# test_traverse = brisbane_event_traverses[2]
# training_set = BrisbaneVPRDatasetTime(train=True, traverse_name=train_traverse)
# testing_set  = BrisbaneVPRDatasetTime(train=False, traverse_name=test_traverse, training_locations=training_set.place_locations)

           
# train_loader = DataLoader(dataset=training_set, batch_size=32, shuffle=True)
# test_loader  = DataLoader(dataset=testing_set , batch_size=32, shuffle=True)


            # # Create filters using the subselected pixels
            # filter0x = event_streams[0]['x'].isin(x_select)
            # filter0y = event_streams[0]['y'].isin(y_select)
            # filter1x = event_streams[1]['x'].isin(x_select)
            # filter1y = event_streams[1]['y'].isin(y_select)

            # # Apply the filters
            # event_streams[0] = event_streams[0][filter0x & filter0y]
            # event_streams[1] = event_streams[1][filter1x & filter1y]
            
            # # Now reset values to be between 0 and 33
            # for i, x in zip(range(34), x_select):
            #     small_filt0x  = event_streams[0]['x'].isin([x])
            #     small_filt1x  = event_streams[1]['x'].isin([x])   
            #     event_streams[0]['x'].loc[small_filt0x] = i
            #     event_streams[1]['x'].loc[small_filt1x] = i  

            # for i, y in zip(range(34), y_select):
            #     small_filt0y  = event_streams[0]['y'].isin([y])
            #     small_filt1y  = event_streams[1]['y'].isin([y])
            #     event_streams[0]['y'].loc[small_filt0y] = i
            #     event_streams[1]['y'].loc[small_filt1y] = i

            # # Divide the event streams into 2 second windows
            # sub_streams0 = []
            # sub_streams1 = []
            # for i in range(0,num_places):
            #     sub_streams0.append(chopData(event_streams[0], i*place_gap, i*place_gap + place_duration, max_spikes))
            #     sub_streams1.append(chopData(event_streams[1], i*place_gap, i*place_gap + place_duration, max_spikes))




# for i, (data, labels) in enumerate(train_loader):
#     print(data.shape, labels.shape)
#     print(data,labels)
#     break;


#print(training_set[0])
# spike_tensor, label = training_set[0]
# spike_tensor = spike_tensor.reshape(2, 34, 34, -1)
# print(spike_tensor)
# event = slayer.io.tensor_to_event(spike_tensor.cpu().data.numpy(), sampling_time=2)

# for i in range(5):
#     spike_tensor, label = testing_set[np.random.randint(len(testing_set))]
#     spike_tensor = spike_tensor.reshape(2, 34, 34, -1)
#     event = slayer.io.tensor_to_event(spike_tensor.cpu().data.numpy())
#     anim = event.anim(plt.figure(figsize=(5, 5)), frame_rate=240)
#     anim.save(f'gifs/input{i}.gif', animation.PillowWriter(fps=24), dpi=300)

# gif_td = lambda gif: f'<td> <img src="{gif}" alt="Drawing" style="height: 250px;"/> </td>'
# header = '<table><tr>'
# images = ' '.join([gif_td(f'gifs/input{i}.gif') for i in range(5)])
# footer = '</tr></table>'
# display.HTML(header + images + footer)
