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
    load_event_streams_full,
    sync_event_streams,
    print_duration,
    get_images_at_start_times,
    filter_and_divide_with_params,
    filter_and_divide_with_global_times
)
from constants import qcr_traverses, path_to_gps_files, synced_times




class QCRVPRDataset(Dataset):
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
        # relative_place_times=None,
        # training_duration=None,
        sampling_time=1, 
        samples_per_sec = 1000,
        num_places = 30,
        start_time = 0,
        place_gap=2, 
        place_duration = 0.5,
        max_spikes=None,
        subselect_num = 34,
        transform=None,
    ):
        super(QCRVPRDataset, self).__init__()

        # Check input parameters 

        total_x_pixels = 346
        total_y_pixels = 260

        x_space = total_x_pixels/subselect_num
        y_space = total_y_pixels/subselect_num

        # Subselect 34 x 34 pixels evenly spaced out - Create a filter
        x_select  = [int(i*x_space + x_space/2) for i in range(subselect_num)]
        y_select  = [int(i*y_space + y_space/2) for i in range(subselect_num)]

        if train:
            print("Loading training event streams ...")

            # Load the training stream itself and synchronise 
            event_streams = load_event_streams_full([traverse_name])
            event_streams = sync_event_streams(event_streams, [traverse_name])
            duration = print_duration(event_streams[0])

            # Get the place samples 
            sub_streams, start_times = filter_and_divide_with_params(event_streams[0], x_select, y_select, num_places, start_time, place_gap, place_duration, max_spikes)
            print(start_times)
            
            # Get the start 
            # print(start_times)
            # self.relative_place_times = start_times/duration
            # self.training_duration = duration

            # # Get the closest CMOS images at each start time
            self.place_images = get_images_at_start_times(start_times, traverse_name, event_streams[0]['t'].iloc[0]/1e6)

            self.samples = sub_streams
            #self.speed_ratio = 1
            print("The number of training substreams is: " + str(len(self.samples)))
            
            
        else:
            # assert relative_place_times is not None, "Must provide relative place times"
        	
            # Load the test stream and synchronise
            event_streams = load_event_streams_full([traverse_name])
            event_streams = sync_event_streams(event_streams, [traverse_name])
            duration = print_duration(event_streams[0])

            # # Estimate the closest locations in test dataset to training dataset and get their start_times 
            # start_times = relative_place_times*duration
            # print(start_times)

            # # Calculate speed ratio if using speed information to determine place duration
            # if training_duration:
            #     speed_ratio = duration/training_duration
            #     new_place_duration = place_duration #place_duration*speed_ratio
            #     self.speed_ratio = speed_ratio
            # else:
            #     new_place_duration = place_duration
            #     self.speed_ratio = 1
            # print("Place duration " + str(new_place_duration))
            # print("Speed ratio " + str(self.speed_ratio))
            


            # Get the place samples 
            sub_streams, start_times = filter_and_divide_with_params(event_streams[0], x_select, y_select, num_places, start_time, place_gap, place_duration, max_spikes)
            print(start_times)

            # # Get the closest CMOS images at each start time
            self.place_images = get_images_at_start_times(start_times, traverse_name, event_streams[0]['t'].iloc[0]/1e6)

            
            self.samples = sub_streams
            print("The number of testing substreams is: " + str(len(self.samples)))

        self.place_duration = place_duration # The duration of a place in seconds 
        self.place_gap = place_gap # The time between each place window
        self.num_places = num_places # the number of places
        self.sampling_time = sampling_time # Default sampling time is 1
        self.samples_per_sec = samples_per_sec
        self.transform = transform
        self.subselect_num = subselect_num
        
        #self.num_time_bins = int(sample_length/sampling_time)
       

    def __getitem__(self, i):
        
        # make sure we aren't calling an index out of range
        assert i < self.num_places, "Index out of range! There are not that many place samples"

        # Find the place label
        label = int(i % (self.num_places))

        # Find the number of time bins
        num_time_bins = int(self.place_duration*self.samples_per_sec)
        time_divider = int(1000000/self.samples_per_sec)

        # Turn the sample stream into events
        x_event = self.samples[i]['x'].to_numpy()
        y_event = self.samples[i]['y'].to_numpy()
        c_event = self.samples[i]['p'].to_numpy()
        t_event = self.samples[i]['t'].to_numpy()
        #event = slayer.io.Event(x_event, y_event, c_event, t_event/1000)
        event = slayer.io.Event(x_event, y_event, c_event, (t_event/time_divider))

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


class QCRVPRSyncDataset(Dataset):
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
        training_duration=None,
        sampling_time=1, 
        samples_per_sec = 1000,
        place_duration = 0.5,
        max_spikes=None,
        subselect_num = 34,
        transform=None,
    ):
        super(QCRVPRSyncDataset, self).__init__()

        # Check input parameters 
        total_x_pixels = 346
        total_y_pixels = 260

        x_space = total_x_pixels/subselect_num
        y_space = total_y_pixels/subselect_num

        # Subselect 34 x 34 pixels evenly spaced out - Create a filter
        x_select  = [int(i*x_space + x_space/2) for i in range(subselect_num)]
        y_select  = [int(i*y_space + y_space/2) for i in range(subselect_num)]

        if train:
            print("Loading training event streams ...")

            # Load the training stream itself and synchronise 
            event_streams = load_event_streams_full([traverse_name])
            event_streams = sync_event_streams(event_streams, [traverse_name])
            duration = print_duration(event_streams[0])

            # Get the start times for the traversal
            start_times = synced_times[traverse_name]
            print(start_times)

            # Get the place samples 
            sub_streams = filter_and_divide_with_global_times(event_streams[0], x_select, y_select, start_times, place_duration, max_spikes)
            
            # Save the duration
            self.training_duration = duration

            # # Get the closest CMOS images at each start time
            self.place_images = get_images_at_start_times(start_times, traverse_name, times_are_relative_to_start=False)

            self.samples = sub_streams
            self.speed_ratio = 1
            print("The number of training substreams is: " + str(len(self.samples)))
            
            
        else:

            # Load the test stream and synchronise
            event_streams = load_event_streams_full([traverse_name])
            event_streams = sync_event_streams(event_streams, [traverse_name])
            duration = print_duration(event_streams[0])

            # Get the start times for the traversal
            start_times = synced_times[traverse_name]
            print(start_times)

            # Calculate speed ratio if using speed information to determine place duration
            if training_duration:
                speed_ratio = duration/training_duration
                new_place_duration = place_duration*speed_ratio
                self.speed_ratio = 1 #speed_ratio
            else:
                new_place_duration = place_duration
                self.speed_ratio = 1
            print("Place duration " + str(new_place_duration))
            print("Speed ratio " + str(self.speed_ratio))
            

            # # Get the closest CMOS images at each start time
            self.place_images = get_images_at_start_times(start_times, traverse_name, times_are_relative_to_start=False)

            # Get the place samples 
            sub_streams = filter_and_divide_with_global_times(event_streams[0], x_select, y_select, start_times, new_place_duration, max_spikes)
            place_duration = new_place_duration
            
            self.samples = sub_streams
            print("The number of testing substreams is: " + str(len(self.samples)))

        self.place_duration = place_duration # The duration of a place in seconds 
        self.num_places = len(start_times) # the number of places
        self.sampling_time = sampling_time # Default sampling time is 1
        self.samples_per_sec = samples_per_sec
        self.transform = transform
        self.subselect_num = subselect_num
        self.train = train
        
        #self.num_time_bins = int(sample_length/sampling_time)
       

    def __getitem__(self, i):
        
        # make sure we aren't calling an index out of range
        assert i < self.num_places, "Index out of range! There are not that many place samples"

        # Find the place label
        label = int(i % (self.num_places))

        # Find the number of time bins
        num_time_bins = int(self.place_duration*self.samples_per_sec)
        time_divider = int(1000000/self.samples_per_sec)



        # Turn the sample stream into events
        x_event = self.samples[i]['x'].to_numpy()
        y_event = self.samples[i]['y'].to_numpy()
        c_event = self.samples[i]['p'].to_numpy()
        t_event = self.samples[i]['t'].to_numpy()
        #event = slayer.io.Event(x_event, y_event, c_event, t_event/1000)
        event = slayer.io.Event(x_event, y_event, c_event, (t_event/time_divider)/self.speed_ratio)

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
# training_set = QCRVPRDataset(traverse_name=qcr_traverses[0], train=True)
# testing_set  = QCRVPRDataset(traverse_name=qcr_traverses[3], train=False, relative_place_times=training_set.relative_place_times)

