# Code adapted  from https://lava-nc.org/lava-lib-dl/slayer/notebooks/nmnist/train.html

from fileinput import close
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
import os

import lava.lib.dl.slayer as slayer
import torch
from torch.utils.data import Dataset, DataLoader



from utils import (
    get_short_traverse_name,
    get_gps,
    print_duration,
    interpolate_gps,
    get_images_at_start_times,
    print_distance,
    divide_training,
    divide_testing,
    find_closest_matches

)
from constants import  path_to_gps_files, path_to_pickles





class BrisbaneVPRDatasetTime(Dataset):
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
        place_duration = 2,
        max_spikes=None,
        transform=None,
        
    ):
        super(BrisbaneVPRDatasetTime, self).__init__()

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
            sub_streams, start_times, distances = divide_training(event_stream, num_places, start_dist, place_gap, place_duration, max_spikes)

            # Get the interpolated reference gps locations at each start time
            self.place_locations = interpolate_gps(gps_gt[0], start_times)[:, :2]
            
            # Get the closest CMOS images at each start time
            self.place_images = get_images_at_start_times(start_times, traverse_name)

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


            # Load the test stream and synchronise
            event_stream = pd.read_pickle(path_to_pickles + get_short_traverse_name(traverse_name) + ".pkl")
            print_duration(event_stream)
            print_distance(event_stream)

            # Get the place samples 
            sub_streams, distances = divide_testing(event_stream, start_times, place_duration, max_spikes)
            
            self.samples = sub_streams
            print("The number of testing substreams is: " + str(len(self.samples)))

        self.place_duration = place_duration # The duration of a place in seconds 
        self.place_gap = place_gap # The time between each place window
        self.num_places = num_places # the number of places
        self.sampling_time = sampling_time # Default sampling time is 1
        self.samples_per_sec = samples_per_sec
        self.transform = transform
       

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
        event = slayer.io.Event(x_event, y_event, c_event, t_event/time_divider)

        # Transform event
        if self.transform is not None:
            event = self.transform(event)

        # Turn the events into a tensor 
        spike = event.fill_tensor(
                torch.zeros(2, 34, 34, num_time_bins),
                sampling_time=self.sampling_time,
            )
        
        return spike.reshape(-1, num_time_bins), label

    def __len__(self):
        return len(self.samples)

