import os

brisbane_event_traverses = [
    "dvs_vpr_2020-04-21-17-03-03_no_hot_pixels_nobursts_denoised.parquet",  # sunset1
    "dvs_vpr_2020-04-22-17-24-21_no_hot_pixels_nobursts_denoised.parquet",  # sunset2
    "dvs_vpr_2020-04-24-15-12-03_no_hot_pixels_nobursts_denoised.parquet",  # daytime
    "dvs_vpr_2020-04-28-09-14-11_no_hot_pixels_nobursts_denoised.parquet",  # morning
    "dvs_vpr_2020-04-29-06-20-23_no_hot_pixels_nobursts_denoised.parquet",  # sunrise
]

brisbane_event_traverses_aliases = {
    "dvs_vpr_2020-04-21-17-03-03": "Sunset 1",
    "dvs_vpr_2020-04-22-17-24-21": "Sunset 2",
    "dvs_vpr_2020-04-24-15-12-03": "Daytime",
    "dvs_vpr_2020-04-28-09-14-11": "Morning",
    "dvs_vpr_2020-04-29-06-20-23": "Sunrise",
}


video_beginning = {
    "dvs_vpr_2020-04-21-17-03-03": 1587452593.35,
    "dvs_vpr_2020-04-22-17-24-21": 1587540271.65,
    "dvs_vpr_2020-04-24-15-12-03": 1587705136.80,
    "dvs_vpr_2020-04-28-09-14-11": 1588029271.73,
    "dvs_vpr_2020-04-29-06-20-23": 1588105240.91,
    "night": 1587975221.10,
}

qcr_traverses = [
    "bags_2021-08-19-08-25-42_denoised.parquet",  # S11 side-facing, slow
    "bags_2021-08-19-08-28-43_denoised.parquet",  # S11 side-facing, slow
    "bags_2021-08-19-09-45-28_denoised.parquet",  # S11 side-facing, slow
    "bags_2021-08-20-10-19-45_denoised.parquet",  # S11 side-facing, fast
    "bags_2021-08-20-09-52-59_denoised.parquet",  # S11 down-facing, slow
    "bags_2021-08-20-09-49-58_denoised.parquet",  # S11 down-facing, slow
    "bags_2021-10-21-10-32-55_denoised.parquet",  # S11 down-facing, slow, with GoPro
    "bags_2021-10-21-10-36-59_denoised.parquet",  # S11 down-facing, slow, with GoPro
    "bags_2021-10-25-13-16-08_denoised.parquet",  # QUT down-facing, slow, with GoPro
    "bags_2021-10-25-13-32-00_denoised.parquet",  # QUT down-facing, slow, with GoPro
    "bags_2021-10-26-15-41-37_denoised.parquet",  # QUT down-facing, slow, with GoPro
    "bags_2021-10-26-15-56-15_denoised.parquet",  # QUT down-facing, fast, with GoPro
    "bags_2021-10-27-08-22-34_denoised.parquet",  # QUT down-facing, slow, with GoPro
    "bags_2022-03-28-11-51-26_denoised.parquet",  # S11 side-facing, slow
    "bags_2022-03-28-12-01-42_denoised.parquet",  # S11 side-facing, fast
    "bags_2022-03-28-12-03-44_denoised.parquet",  # S11 side-facing, extra slow
]

qcr_traverses_aliases = {
    "bags_2022-03-28-11-51-26": "Normal",
    "bags_2022-03-28-12-01-42": "Fast",
    "bags_2022-03-28-12-03-44": "Slow",
}


gt_times = {
    "bags_2022-03-28-11-51-26_denoised.parquet": [0, 6.7, 13.2, 31, 57, 74, 97, 119, 141, 148, 154],
    "bags_2022-03-28-12-01-42_denoised.parquet": [0, 2.9, 5.8, 12.9, 23.5, 32, 41.5, 50, 59.3, 62, 64.5],
    "bags_2022-03-28-12-03-44_denoised.parquet": [0, 15.5, 30, 63.5, 110, 141, 185, 217, 246.5, 256, 263],
}


qcr_traverses_first_times = {
    "bags_2021-08-19-08-25-42": 2e6,
    "bags_2021-08-19-09-45-28": 2e6,
    "bags_2022-03-28-11-51-26": 8e6,
    "bags_2022-03-28-12-01-42": 7.7e6,
    "bags_2022-03-28-12-03-44": 12e6,
}

qcr_traverses_last_times = {
    "bags_2021-10-21-10-32-55": 165e6,
    "bags_2021-08-19-08-25-42": 166.2e6,
    "bags_2021-08-19-09-45-28": 166.2e6,
    "bags_2022-03-28-11-51-26": 8e6 + gt_times["bags_2022-03-28-11-51-26_denoised.parquet"][-1] * 1e6,
    "bags_2022-03-28-12-01-42": 7.7e6 + gt_times["bags_2022-03-28-12-01-42_denoised.parquet"][-1] * 1e6,
    "bags_2022-03-28-12-03-44": 12e6 + gt_times["bags_2022-03-28-12-03-44_denoised.parquet"][-1] * 1e6,
}

gt_percentage_travelled = {}
for traverse, traverse_gt_times in gt_times.items():
    gt_percentage_travelled[traverse] = [(gt_time / traverse_gt_times[-1]) * 100 for gt_time in traverse_gt_times]

time_windows_overwrite = {  # roughly normalise time windows for traverses where robot was going faster
    "bags_2021-08-20-10-19-45": 0.4,
    "bags_2021-10-26-15-56-15": 0.4,
}

path_to_data = os.path.dirname(os.path.abspath(__file__)) + "/../data/"
path_to_event_files = os.path.dirname(os.path.abspath(__file__)) + "/../data/"
path_to_frames = os.path.dirname(os.path.abspath(__file__)) + "/../data/input_frames/"
path_to_frames_event_vlad = os.path.dirname(os.path.abspath(__file__)) + "/../data/input_frames_event_vlad/"
path_to_denoised_event_vlad = os.path.dirname(os.path.abspath(__file__)) + "/../data/denoised_frames_event_vlad/"
path_to_gps_files = os.path.dirname(os.path.abspath(__file__)) + "/../data/gps_data/"
path_to_image_files = os.path.dirname(os.path.abspath(__file__)) + "/../data/image_data/"
