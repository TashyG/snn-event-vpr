# Code adapted  from https://lava-nc.org/lava-lib-dl/slayer/notebooks/nmnist/train.html

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

from utils import (
    load_event_streams,
    sync_event_streams,
    get_short_traverse_name,
    get_gps,
    print_duration,
    interpolate_gps
)
from constants import brisbane_event_traverses, path_to_gps_files

  
train_traverse = brisbane_event_traverses[0]
test_traverse = brisbane_event_traverses[4]
train_gps_locations = None  

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


def filter_and_divide(event_stream, x_select, y_select, num_places, place_gap, place_duration, max_spikes):
    """
    Filters a long event stream to have only select pixels in the data and
    then divides the stream up into multiple substreams

    :param event_stream: The stream of events, a DataFrame object 
    :param x_select: The x pixel positions to be included (all others are fitlered out)
    :param y_select: The y pixel positions to be included (all others are filtered out)
    :param num_places: The number of place samples to extract (starting from the start of the event stream)
    :param place_gap: The time gap in seconds between the start of each place sample
    :param place_duration: The duration of each place sample in seconds
    :param max_spikes: The maximum number of spikes that can be in a place sample
    :return: The filtered substreams/place samples
    :return: The start times of each place sample
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
        sub_streams.append(chopData(event_stream, i*place_gap, i*place_gap + place_duration, max_spikes))
        start_times.append(i*place_gap*1000000)

    return sub_streams, start_times

def get_testing_labels(test_gps_locations):
    global train_gps_locations

    print(train_gps_locations)
    print(test_gps_locations)


class BrisbaneVPRDataset(Dataset):
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
        train=True,
        sampling_time=1, 
        samples_per_sec = 1000,
        num_places = 30,
        place_gap=2, 
        place_duration = 0.5,
        max_spikes=None,
        subselect_num = 34,
        transform=None,
    ):
        super(BrisbaneVPRDataset, self).__init__()
        global train_gps_locations
        total_x_pixels = 346
        total_y_pixels = 260

        x_space = total_x_pixels/subselect_num
        y_space = total_y_pixels/subselect_num

        # Subselect 34 x 34 pixels evenly spaced out - Create a filter
        x_select  = [int(i*x_space + x_space/2) for i in range(subselect_num)]
        y_select  = [int(i*y_space + y_space/2) for i in range(subselect_num)]


        if train:
            print("Loading training event streams ...")
            gps_gt = []
            if os.path.isfile(path_to_gps_files + get_short_traverse_name(train_traverse) + ".nmea"):
                tqdm.write("Adding GPS")
                gps_gt.append(get_gps(path_to_gps_files + get_short_traverse_name(train_traverse) + ".nmea"))

            # Load the training stream
            event_streams = load_event_streams([train_traverse])
            event_streams = sync_event_streams(event_streams, [train_traverse], gps_gt)

            # Print length of full stream
            print_duration(event_streams[0])

            # Get the place samples 
            sub_streams, start_times = filter_and_divide(event_streams[0], x_select, y_select, num_places, place_gap, place_duration, max_spikes)
            
            # Get the interpolated reference gps locations at each start time
            train_gps_locations = interpolate_gps(gps_gt[0], start_times)[:, :2]

            self.samples = sub_streams
            print("The number of training substreams is: " + str(len(self.samples)))
            
            
        else:
            #assert train_gps_locations is not None, "Training data must be loaded first"
            print("Loading testing event streams ...")
            gps_gt = []
            if os.path.isfile(path_to_gps_files + get_short_traverse_name(test_traverse) + ".nmea"):
                tqdm.write("Adding GPS")
                gps_gt.append(get_gps(path_to_gps_files + get_short_traverse_name(test_traverse) + ".nmea"))
            print(gps_gt[0])
            # Find the closest  GPS locations to the training locations
            gps_gt = gps_gt[0]
            test_gps_locations = interpolate_gps(gps_gt, np.arange(0, 667, 0.01))
            print(test_gps_locations)
            # Load the test stream
            # event_streams = load_event_streams([test_traverse])
            # event_streams = sync_event_streams(event_streams, [test_traverse], gps_gt)

            # # Print length of full stream
            # print_duration(event_streams[0])

            # # Get the place samples 
            # sub_streams, start_times = filter_and_divide(event_streams[0], x_select, y_select, num_places, place_gap, place_duration, max_spikes)

            # # Get the interpolated query gps locations at each start time
            # test_gps_locations = interpolate_gps(gps_gt[0], start_times)[:, :2]

            # # Find the ground truth labels for the testing dataset based on the training dataset
            # #testing_labels = 
            # get_testing_labels(test_gps_locations)

            # self.samples = sub_streams
            # print("The number of testing substreams is: " + str(len(self.samples)))

        self.place_duration = place_duration # The duration of a place in seconds 
        self.place_gap = place_gap # The time between each place window
        self.num_places = num_places
        self.sampling_time = sampling_time # Default sampling time is 1
        self.samples_per_sec = samples_per_sec
        self.transform = transform
        self.subselect_num = subselect_num
        #self.num_time_bins = int(sample_length/sampling_time)
       

    def __getitem__(self, i):

        # Find the place label
        #num_places = self.stream_length/self.place_gap
        label = int(i % (self.num_places))
        #print("The sample number is: " + str(i) + " with a label of: " + str(label))

        # Find the sample length and number of time bins
        time_divider = int(1000000/self.samples_per_sec)
        num_time_bins = int(self.place_duration*self.samples_per_sec)

        #sample_length = len(self.samples[i])
        #num_time_bins = int(sample_length/self.sampling_time)
        #print("The sample Length is " + str(num_time_bins))

        # Turn the sample stream into events
        x_event = self.samples[i]['x'].to_numpy()
        y_event = self.samples[i]['y'].to_numpy()
        c_event = self.samples[i]['p'].to_numpy()
        t_event = self.samples[i]['t'].to_numpy()
        # print(num_time_bins)
        # print(t_event/100)
        #event = slayer.io.Event(x_event, y_event, c_event, t_event/1000)
        event = slayer.io.Event(x_event, y_event, c_event, t_event/time_divider)

        # Transform event
        if self.transform is not None:
            event = self.transform(event)

        # Turn the events into a tensor 
        spike = event.fill_tensor(
                torch.zeros(2, self.subselect_num, self.subselect_num, num_time_bins),
                sampling_time=self.sampling_time,
            )
        
        return spike.reshape(-1, num_time_bins), label

    def __len__(self):
        return len(self.samples)


# Ultimate test- loading the data
#training_set = BrisbaneVPRDataset(train=True)
testing_set  = BrisbaneVPRDataset(train=False)
            
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
