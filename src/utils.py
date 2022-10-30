# Code adapted from https://github.com/Tobias-Fischer/salient-event-vpr

import os
import re
import glob
import numpy as np
import torch

import pandas as pd
import pyarrow.parquet as pq
from PIL import Image

# import terality as pd
import cv2

# from scipy.spatial.distance import cdist
from scipy import interpolate
import sys
import tonic
import pynmea2

from tqdm.auto import tqdm
from scipy.spatial.distance import cdist

from os import path
sys.path.append( path.dirname( path.abspath(__file__) ) )
from constants import (
    time_windows_overwrite,
    path_to_event_files,
    path_to_frames,
    path_to_image_files,
    gt_times,
    video_beginning,
    qcr_traverses_first_times,
    qcr_traverses_last_times,
    brisbane_event_traverses_aliases,
    qcr_traverses_aliases,
)


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {"false", "f", "0", "no", "n"}:
        return False
    elif value.lower() in {"true", "t", "1", "yes", "y"}:
        return True
    raise ValueError(f"{value} is not a valid boolean value")


def none_or_str(value):
    if value == "None":
        return None
    return value


def none_or_int(value):
    if value == "None":
        return None
    return int(value)


def get_short_traverse_name(traverse_name):
    m = re.search(r"(\d)\D*$", traverse_name)
    traverse_short = traverse_name[: m.start() + 1]
    return traverse_short


def get_traverse_alias(traverse_name_short):
    if traverse_name_short in brisbane_event_traverses_aliases:
        return brisbane_event_traverses_aliases[traverse_name_short]
    elif traverse_name_short in qcr_traverses_aliases:
        return qcr_traverses_aliases[traverse_name_short]
    else:
        return traverse_name_short


def load_event_streams(event_streams_to_load, dir_to_load_from=path_to_event_files):
    event_streams = []
    for event_stream in tqdm(event_streams_to_load):
        parquet_file = pq.ParquetFile(os.path.join(dir_to_load_from, event_stream))
        parquet_subset = parquet_file.read_row_groups([0,1,2,3])
        dataframe = parquet_subset.to_pandas()
        event_streams.append(dataframe)
        # for batch in parquet_file.iter_batches():
        #     print("RecordBatch")
        #     batch_df = batch.to_pandas()
        #     print("batch_df:", batch_df)
        #     print("\n\n")

        # event_streams.append(pd.read_parquet(os.path.join(dir_to_load_from, event_stream)))
    return event_streams

def load_event_streams_full(event_streams_to_load):
    event_streams = []
    for event_stream in tqdm(event_streams_to_load):
        event_streams.append(pd.read_parquet(os.path.join(path_to_event_files, event_stream)))
    return event_streams


def get_size(event_stream):
    im_width, im_height = int(event_stream["x"].max() + 1), int(event_stream["y"].max() + 1)
    return im_width, im_height


def print_duration(event_stream):
    duration = (event_stream.iloc[-1]["t"] - event_stream.iloc[0]["t"]) / 1e6
    print(f'Duration: {duration:.2f}s (which is {len(event_stream)} events)')
    return duration

def print_distance(event_stream):
    print(f'Distance: {((event_stream.iloc[-1]["distance"] - event_stream.iloc[0]["distance"])):.2f}m (which is {len(event_stream)} events)')


# Turns numpy array of events into event frames and combines positive and negative polarities
def create_event_frames(event_streams_numpy, traverses_to_compare, sensor_size, frame_length=1e6, event_count=None, overlap=0, time_slices=None):
    if time_slices is None:
        time_slices = [None] * len(event_streams_numpy)

    event_frames = [
        tonic.functional.to_frame_numpy(
            event_stream_numpy,
            sensor_size,
            time_window=int(frame_length * time_windows_overwrite.get(get_short_traverse_name(traverse), 1)) if frame_length is not None else None,
            event_count=event_count,
            overlap=int(overlap * time_windows_overwrite.get(get_short_traverse_name(traverse), 1)) if frame_length is not None else overlap,
            event_slices=event_slices,
        )
        for traverse, event_stream_numpy, event_slices in zip(traverses_to_compare, event_streams_numpy, time_slices)
    ]
    event_frames_pos = [event_frame[:, 0, ...] for event_frame in event_frames]
    event_frames_neg = [event_frame[:, 1, ...] for event_frame in event_frames]

    # Combine negative and positive polarities
    event_frames_total = [event_frame_pos + event_frame_neg for event_frame_pos, event_frame_neg in zip(event_frames_pos, event_frames_neg)]
    return event_frames_total


def get_times_for_streams_const_time(event_stream, time_window, overlap, include_incomplete=False):
    times = event_stream["t"]
    stride = time_window - overlap

    last_time = times.iloc[-1] if isinstance(times, pd.Series) else times[-1]
    begin_time = times.iloc[0] if isinstance(times, pd.Series) else times[0]

    if include_incomplete:
        n_slices = int(np.ceil(((last_time - begin_time) - time_window) / stride) + 1)
    else:
        n_slices = int(np.floor(((last_time - begin_time) - time_window) / stride) + 1)

    window_start_times = np.arange(n_slices) * stride
    # window_end_times = window_start_times + time_window

    return window_start_times


def get_times_for_streams_const_count(event_stream, event_count, overlap, include_incomplete=False):
    n_events = len(event_stream)
    event_count = min(event_count, n_events)

    stride = event_count - overlap
    if stride <= 0:
        raise Exception("Inferred stride <= 0")

    if include_incomplete:
        n_slices = int(np.ceil((n_events - event_count) / stride) + 1)
    else:
        n_slices = int(np.floor((n_events - event_count) / stride) + 1)

    times = event_stream["t"].to_numpy()
    begin_time = times.iloc[0] if isinstance(times, pd.Series) else times[0]
    indices_start = (np.arange(n_slices) * stride).astype(int)

    return times[indices_start] - begin_time


def get_gopro_frames(traverses_to_compare):
    gopro_frames = []
    for traverse in tqdm(traverses_to_compare):
        gopro_frames.append([])
        for filename in tqdm(sorted(glob.glob(path_to_event_files + "/" + get_short_traverse_name(traverse) + "-gopro/*.png"))):
            gopro_frames[-1].append(cv2.resize(cv2.imread(filename, cv2.IMREAD_GRAYSCALE), (85, 48)))
        gopro_frames[-1] = np.array(gopro_frames[-1])
        assert len(gopro_frames[-1]) > 0
    return gopro_frames


def get_conventional_frames(traverse, desired_times):
    filenames = sorted(glob.glob(path_to_frames + "/" + get_short_traverse_name(traverse) + "/frames/*.png"))
    timestamps = np.array([float(os.path.basename(filename).replace(".png", "")) for filename in filenames])
    idx_to_load = []
    for time in desired_times:
        idx_to_load.append((np.abs(time - timestamps)).argmin())

    conventional_frames = []
    for idx in idx_to_load:
        conventional_frames.append(cv2.imread(filenames[idx], cv2.IMREAD_GRAYSCALE))

    return np.array(conventional_frames)


def sync_event_streams(event_streams, traverses_to_compare, gps_gt=None):
    event_streams_synced = []
    for event_stream_idx, (event_stream, name) in enumerate(zip(event_streams, traverses_to_compare)):
        short_name = get_short_traverse_name(name)
        start_time = event_stream.iloc[0]["t"]
        # end_time = event_stream.iloc[-1]["t"]
        if short_name.startswith("bags_"):
            if short_name in qcr_traverses_first_times:
                first_idx = event_stream["t"].searchsorted(start_time + qcr_traverses_first_times[short_name])
            else:
                first_idx = 0
            if short_name in qcr_traverses_last_times:
                last_idx = event_stream["t"].searchsorted(start_time + qcr_traverses_last_times[short_name])
            else:
                last_idx = None
            event_streams_synced.append(event_stream[first_idx:last_idx].reset_index(drop=True))
        elif short_name.startswith("dvs_vpr_"):
            first_idx = event_stream["t"].searchsorted(video_beginning[short_name] * 1e6)
            if event_stream_idx < len(gps_gt):
                last_idx = event_stream["t"].searchsorted((video_beginning[short_name] + gps_gt[event_stream_idx][-1, 2]) * 1e6)
            else:
                last_idx = None
            event_streams_synced.append(event_stream[first_idx:last_idx].reset_index(drop=True))
        else:
            event_streams_synced.append(event_stream)
    return event_streams_synced

def get_images_at_start_times(start_times, traverse_name, traverse_start_time=None, times_are_relative_to_start=True):
    short_name = get_short_traverse_name(traverse_name)

    if times_are_relative_to_start: 
        if traverse_start_time:
            start_of_recording = traverse_start_time
        else:  
            start_of_recording = video_beginning[short_name]

    path_to_images = path_to_image_files + short_name + '/frames/'

    images_at_start_times = [] 
    for start_time in start_times:
        if times_are_relative_to_start:
            actual_time = start_of_recording + start_time
        else:
            actual_time = start_time
        actual_time_string = '{0:12.1f}'.format(actual_time)

        # Get image files starting the start time
        closest_timestamp_paths = glob.glob(path_to_images + actual_time_string + '*.png')
        if len(closest_timestamp_paths) == 0:
            actual_time_string = actual_time_string[:-1]
            closest_timestamp_paths = glob.glob(path_to_images + actual_time_string + '*.png')
        closest_timestamps = []
        for close_timestamp in closest_timestamp_paths:
            close_timestamp = re.findall('[0-9]+[.][0-9]+', close_timestamp)
            assert len(close_timestamp) == 1, "The path to the images is making regex fail"
            closest_timestamps.append(float(close_timestamp[0]))
        
        images_at_start_times.append(closest_timestamp_paths[min(range(len(closest_timestamps)), key = lambda i: abs(closest_timestamps[i]-actual_time))])

    return images_at_start_times

   

def remove_random_bursts(event_frames, threshold):
    print(event_frames[event_frames > threshold])
    event_frames[event_frames > threshold] = threshold
    return event_frames


def get_distance_matrix(ref_traverse: np.ndarray, qry_traverse: np.ndarray, metric="cityblock", device=None):
    # scipy_dist = cdist(ref_traverse.reshape(ref_traverse.shape[0], -1), qry_traverse.reshape(query_traverse.shape[0], -1), metric=metric)
    a = torch.from_numpy(ref_traverse.reshape(ref_traverse.shape[0], -1).astype(np.float32)).unsqueeze(0).to(device)
    b = torch.from_numpy(qry_traverse.reshape(qry_traverse.shape[0], -1).astype(np.float32)).unsqueeze(0).to(device)
    if metric == "cityblock":
        torch_dist = torch.cdist(a, b, 1)[0]
    elif metric == "euclidean":
        torch_dist = torch.cdist(a, b, 2)[0]
    elif metric == "cosine":

        def cosine_distance_torch(x1, x2=None, eps=1e-8):
            x2 = x1 if x2 is None else x2
            w1 = x1.norm(p=2, dim=1, keepdim=True)
            w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
            return 1 - torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)

        torch_dist = cosine_distance_torch(a.squeeze(0), b.squeeze(0))
    else:
        raise ValueError("Distance not supported")

    return torch_dist


def get_score_ratio_test(dist_matrix, neighborhood_exclusion_radius=3):
    match_scores_revised = np.empty(dist_matrix.shape[1], dtype=np.float32)
    for query in range(dist_matrix.shape[1]):
        refs_sorted = dist_matrix[:, query].argsort()
        best_match = refs_sorted[0]
        second_best_match = refs_sorted[np.abs(refs_sorted - best_match) >= neighborhood_exclusion_radius][0]
        if dist_matrix[second_best_match, query] == 0:  # Ignore division by zero
            match_scores_revised[query] = 1.0
        else:
            match_scores_revised[query] = dist_matrix[best_match, query] / dist_matrix[second_best_match, query]
    return match_scores_revised


def get_gps(nmea_file_path):
    nmea_file = open(nmea_file_path, encoding="utf-8")

    latitudes, longitudes, timestamps, distances = [], [], [], []

    first_timestamp = None
    previous_lat, previous_lon = None, None

    for line in nmea_file.readlines():
        try:
            msg = pynmea2.parse(line)
            if msg.sentence_type not in ["GSV", "VTG", "GSA"]:
                if first_timestamp is None:
                    first_timestamp = msg.timestamp
                    previous_lat, previous_lon = msg.latitude, msg.longitude
                    prev_dist = 0

                # print(msg.timestamp, msg.latitude, msg.longitude)
                # print(repr(msg.latitude))
                dist_to_prev = np.linalg.norm(np.array([msg.latitude, msg.longitude]) - np.array([previous_lat, previous_lon]))*100000
                if msg.latitude != 0 and msg.longitude != 0 and msg.latitude != previous_lat and msg.longitude != previous_lon and dist_to_prev > 0.0001:
                    timestamp_diff = (msg.timestamp.hour - first_timestamp.hour) * 3600 + (msg.timestamp.minute - first_timestamp.minute) * 60 + (msg.timestamp.second - first_timestamp.second)
                    latitudes.append(msg.latitude)
                    longitudes.append(msg.longitude)
                    timestamps.append(timestamp_diff)
                    next_dist = prev_dist + dist_to_prev
                    distances.append(next_dist)  # noqa
                    previous_lat, previous_lon = msg.latitude, msg.longitude
                    prev_dist = next_dist

        except pynmea2.ParseError as e:  # noqa
            # print('Parse error: {} {}'.format(msg.sentence_type, e))
            continue

    return np.array(np.vstack((latitudes, longitudes, timestamps, distances))).T



def get_gps_speed(nmea_file_path):
    nmea_file = open(nmea_file_path, encoding="utf-8")

    speeds, timestamps = [], []

    first_timestamp = None

    for line in nmea_file.readlines():
        try:
            msg = pynmea2.parse(line)
            
            if msg.sentence_type == 'RMC':
                if first_timestamp is None:
                    first_timestamp = msg.timestamp

                timestamp_diff = (msg.timestamp.hour - first_timestamp.hour) * 3600 + (msg.timestamp.minute - first_timestamp.minute) * 60 + (msg.timestamp.second - first_timestamp.second)
                timestamps.append(timestamp_diff)
            if msg.sentence_type == "VTG":
                speeds.append(msg.spd_over_grnd_kmph)  # noqa

        except pynmea2.ParseError as e:  # noqa
            # print('Parse error: {} {}'.format(msg.sentence_type, e))
            continue

    return np.array(np.vstack((speeds, timestamps))).T


def interpolate_gps(gps_data, desired_times=None): 
    f_time_to_lat = interpolate.interp1d(gps_data[:, 2], gps_data[:, 0], fill_value="extrapolate")
    f_time_to_lon = interpolate.interp1d(gps_data[:, 2], gps_data[:, 1], fill_value="extrapolate")

    if desired_times is None:
        desired_times = np.arange(gps_data[-1, 2])

    new_lat = np.array([f_time_to_lat(t) for t in desired_times])
    new_lon = np.array([f_time_to_lon(t) for t in desired_times])
    return np.array(np.vstack((new_lat, new_lon, desired_times))).T

def interpolate_gps_distance(gps_data, desired_times): 
    f_time_to_distance = interpolate.interp1d(gps_data[:, 0], gps_data[:, 1], fill_value="extrapolate")

    new_distance = np.array([f_time_to_distance(t) for t in desired_times])
    return np.array(np.vstack((new_distance, desired_times))).T


def interpolate_gps_speed(gps_data, desired_times=None): 
    f_time_to_speed = interpolate.interp1d(gps_data[:, 1], gps_data[:, 0], fill_value="extrapolate")

    if desired_times is None:
        desired_times = np.arange(gps_data[-1, 1])

    new_speed = np.array([f_time_to_speed(t) for t in desired_times])
    return np.array(np.vstack((new_speed, desired_times))).T


def get_precomputed_convweight(seq_length, diff_query_ref_factor, device):
    precomputed_convWeight_rows = []

    for i in range(seq_length):
        if i == 0:
            print("0", (seq_length - 1) * diff_query_ref_factor)
            # weights_list = []
            # for j in range(diff_query_ref_factor):
            #     weights_list.append((diff_query_ref_factor - j) / diff_query_ref_factor)
            # print(len(weights_list))
            precomputed_convWeight_rows.append(
                torch.cat(
                    [
                        torch.ones(1, device=device),
                        torch.zeros((seq_length - 1) * diff_query_ref_factor, device=device),
                    ],
                    dim=0,
                ).unsqueeze(0)
            )
        elif i == seq_length - 1:
            print((seq_length - 1) * diff_query_ref_factor, "0")

            precomputed_convWeight_rows.append(
                torch.cat(
                    [
                        torch.zeros((seq_length - 1) * diff_query_ref_factor, device=device),
                        torch.ones(1, device=device),
                    ],
                    dim=0,
                ).unsqueeze(0)
            )
        else:
            print(i * diff_query_ref_factor, (seq_length - i - 1) * diff_query_ref_factor)

            precomputed_convWeight_rows.append(
                torch.cat(
                    [
                        torch.zeros(i * diff_query_ref_factor, device=device),
                        torch.ones(1, device=device),
                        torch.zeros((seq_length - i - 1) * diff_query_ref_factor, device=device),
                    ],
                    dim=0,
                ).unsqueeze(0)
            )
        # print(precomputed_convWeight_rows[-1])

    precomputed_convWeight_adapted = torch.cat(precomputed_convWeight_rows, dim=0).unsqueeze(0).unsqueeze(0)

    # torch.set_printoptions(profile="full")
    # print(precomputed_convWeight_adapted.shape)
    # print(precomputed_convWeight_adapted)
    # torch.set_printoptions(profile="default")

    return precomputed_convWeight_adapted


def save_video(event_frames, out_path, max_count, do_reshape, use_binary_only=False):
    os.makedirs(out_path, exist_ok=True)
    print("Save videos to", out_path)
    for idx, event_frame in enumerate(tqdm(event_frames)):
        if not use_binary_only:
            out_image = (event_frame * 255 / max_count).astype(np.uint8)
        else:
            out_image = ((event_frame > 0) * 255).astype(np.uint8)
        if do_reshape:
            out_image = out_image.reshape(1, -1)
        img = Image.fromarray(out_image)
        if do_reshape:
            img = img.resize((out_image.shape[-1] * 5, 5), Image.NEAREST)
        img.save(f"{out_path}/{idx:06d}.png")


def shift_image(X, dx, dy):
    X = np.roll(X, dy, axis=0)
    X = np.roll(X, dx, axis=1)
    if dy > 0:
        X[:dy, :] = 0
    elif dy < 0:
        X[dy:, :] = 0
    if dx > 0:
        X[:, :dx] = 0
    elif dx < 0:
        X[:, dx:] = 0
    return X


def chopDataTimeTime(event_stream, start_seconds, end_seconds, max_spikes):
    """
    Gets a specific time window of event data out of an event stream

    :param event_stream: Dataframe object containing a stream of events
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

def chopDataDistTime(event_stream, start_distance, place_duration, max_spikes):
    """
    Gets a specific time window of event data out of an event stream starting at
    a particular distance.
    
    :param event_stream: Dataframe object containing a stream of events. Dataframe must have a distance parameter
    :param start_distance: The start distance of when to start extracting events
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


def chopDataGlobal(event_stream, start_seconds, end_seconds, max_spikes):
    """
    Gets a specific time window of event data out of an event stream using
    global start times rather than relative start times

    :param start_seconds: The start timestamp of the window
    :param end_seconds: The end timestamp of the window
    :param max_spikes: The maximum number of spikes that can be in the window
    :return: The event data within the specified time window
    """ 

    chop_start = start_seconds*1000000
    chop_end = end_seconds*1000000

    btwn = event_stream['t'].between(chop_start, chop_end, inclusive='both')
    chopped_stream = event_stream[btwn]

    chopped_stream['t'] -= chop_start

    # Crop the data to the specified number of spikes
    if max_spikes != None:
        chopped_stream = chopped_stream.iloc[0:max_spikes]
        print(chopped_stream['t'].iloc[max_spikes-1]/1000000)


    return chopped_stream


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



def divide_training(event_stream, num_places, start_dist, place_gap, place_duration, max_spikes):
    """
    Divide up an event stream (already filtered) into place samples
    
    :param event_stream: Dataframe object containing a stream of events. Dataframe must have a distance parameter
    :num_places: The number of place samples to extract
    :param start_dist: The first distance at which to start extracting place samples
    :param place_gap: The distance between places
    :param place_duration: The event accumulation time window for the place samples
    :param max_spikes: maximum amount of spike allowed in a place sample
    :return: The event data within the specified time window
    """ 

    # Divide the test stream into 2 second windows
    sub_streams = []
    start_times = []
    distances = []
    for i in range(0,num_places):
        sub_stream, start_time = chopDataDistTime(event_stream, start_dist + i*place_gap, place_duration, max_spikes)
        print("Place: " + str(i))
        print_distance(sub_stream)
        sub_streams.append(sub_stream)
        start_times.append(start_time)
        distances.append(sub_stream.iloc[-1]["distance"] - sub_stream.iloc[0]["distance"])

    return sub_streams, start_times, distances

def divide_testing(event_stream, start_times, place_duration, max_spikes):
    """
    Divide up an event stream (already filtered) into place samples
    
    :param event_stream: Dataframe object containing a stream of events. 
    :num_places: The number of place samples to extract
    :param start_times: The start times of the place samples
    :param place_duration: The event accumulation time window for the place samples
    :param max_spikes: maximum amount of spike allowed in a place sample
    :return: The event data within the specified time window
    """ 

    # Divide the test stream into 2 second windows
    sub_streams = []
    distances = []
    counter = 0
    for start_time in start_times:
        sub_stream = chopDataTimeTime(event_stream, start_time, start_time + place_duration, max_spikes)
        print("Place: " + str(counter))
        counter+=1
        print_distance(sub_stream)
        sub_streams.append(sub_stream)
        distances.append(sub_stream.iloc[-1]["distance"] - sub_stream.iloc[0]["distance"])

    return sub_streams, distances




def filter_and_divide_with_params(event_stream, x_select, y_select, num_places, start_time, place_gap, place_duration, max_spikes):
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
        sub_stream = chopDataTimeTime(event_stream, start_time + i*place_gap, start_time + i*place_gap + place_duration, max_spikes)
        print_duration(sub_stream)
        sub_streams.append(sub_stream)
        start_times.append((start_time + i*place_gap))

    return sub_streams, start_times

def filter_and_divide_with_times(event_stream, x_select, y_select, start_times, place_duration, max_spikes):
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
        sub_streams.append(chopDataTimeTime(event_stream, start_time, start_time + place_duration, max_spikes))

    return sub_streams


def filter_and_divide_with_global_times(event_stream, x_select, y_select, start_times, place_duration, max_spikes):
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
    average_spikes = 0
    for start_time in start_times:
        substream = chopDataGlobal(event_stream, start_time, start_time + place_duration, max_spikes)
        average_spikes += len(substream)
        sub_streams.append(substream)

    print("average spikes = " + str(average_spikes/len(start_times)))

    return sub_streams
