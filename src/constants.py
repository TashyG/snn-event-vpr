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
    "bags_2021-08-19-08-25-42_denoised.parquet",  # S11 side-facing, slow 0
    "bags_2021-08-19-08-28-43_denoised.parquet",  # S11 side-facing, slow 1
    "bags_2021-08-19-09-45-28_denoised.parquet",  # S11 side-facing, slow 2
    "bags_2021-08-20-10-19-45_denoised.parquet",  # S11 side-facing, fast 3
    "bags_2021-08-20-09-52-59_denoised.parquet",  # S11 down-facing, slow 4
    "bags_2021-08-20-09-49-58_denoised.parquet",  # S11 down-facing, slow 5
    "bags_2021-10-21-10-32-55_denoised.parquet",  # S11 down-facing, slow, with GoPro 6
    "bags_2021-10-21-10-36-59_denoised.parquet",  # S11 down-facing, slow, with GoPro 7
    "bags_2021-10-25-13-16-08_denoised.parquet",  # QUT down-facing, slow, with GoPro 8
    "bags_2021-10-25-13-32-00_denoised.parquet",  # QUT down-facing, slow, with GoPro 9
    "bags_2021-10-26-15-41-37_denoised.parquet",  # QUT down-facing, slow, with GoPro 10
    "bags_2021-10-26-15-56-15_denoised.parquet",  # QUT down-facing, fast, with GoPro 11
    "bags_2021-10-27-08-22-34_denoised.parquet",  # QUT down-facing, slow, with GoPro 12
    "bags_2022-03-28-11-51-26_denoised.parquet",  # S11 side-facing, slow 13
    "bags_2022-03-28-12-01-42_denoised.parquet",  # S11 side-facing, fast 14
    "bags_2022-03-28-12-03-44_denoised.parquet",  # S11 side-facing, extra slow 15
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

#stream start times
# 1629325544.892744
# 1629325723.482761
# 1648432297.040480
# 1648432912.934640
# 1648433034.569249


synced_times = {
    "bags_2021-08-19-08-25-42_denoised.parquet": [1629325543.81,1629325548.89, 1629325553.95, 1629325559.16, 1629325564.37, 1629325569.58, 1629325574.79, 1629325579.69, 1629325584.89, 1629325590.17, 1629325595.37, 1629325600.58, 1629325605.78, 1629325610.98, 1629325616.28, 1629325621.59, 1629325626.79, 1629325632.00, 1629325637.28, 1629325642.43, 1629325647.63, 1629325652.84, 1629325658.178670406342, 1629325663.38, 1629325668.58, 1629325673.78, 1629325678.98, 1629325684.32, 1629325689.64, 1629325694.84, 1629325699.88, 1629325705.25],
    "bags_2021-08-19-08-28-43_denoised.parquet": [1629325723.50,1629325728.56, 1629325733.12, 1629325738.60, 1629325743.60, 1629325749.03, 1629325754.23, 1629325759.14, 1629325764.30, 1629325769.59, 1629325774.79, 1629325779.99, 1629325785.37, 1629325789.83, 1629325795.25, 1629325800.62, 1629325805.40, 1629325810.47, 1629325815.80, 1629325821.15, 1629325826.35, 1629325831.56, 1629325836.761175394058, 1629325841.97, 1629325847.60, 1629325852.57, 1629325857.75, 1629325863.08, 1629325868.98, 1629325872.88, 1629325877.94, 1629325883.45],

    "bags_2022-03-28-11-51-26_denoised.parquet": [1648432297.02, 1648432302.25, 1648432309.62, 1648432316.95, 1648432323.89, 1648432331.39, 1648432339.04, 1648432346.59, 1648432353.07, 1648432357.98, 1648432364.56, 1648432371.68, 1648432379.52, 1648432386.62, 1648432394.42, 1648432401.75, 1648432409.40, 1648432416.61, 1648432424.41, 1648432431.15, 1648432438.55, 1648432445.49 ],
    "bags_2022-03-28-12-01-42_denoised.parquet": [1648432912.91, 1648432915.03, 1648432918.10, 1648432921.15, 1648432924.06, 1648432927.01, 1648432930.13, 1648432933.15, 1648432936.10, 1648432939.18, 1648432942.01, 1648432945.04, 1648432948.16, 1648432951.02, 1648432954.11, 1648432957.06, 1648432960.14, 1648432963.08, 1648432966.15, 1648432969.01, 1648432972.12, 1648432975.02 ],     
    "bags_2022-03-28-12-03-44_denoised.parquet": [1648433034.33, 1648433048.06, 1648433063.64, 1648433077.47, 1648433091.34, 1648433105.65, 1648433120.19, 1648433133.24, 1648433144.55, 1648433151.05, 1648433163.63, 1648433177.62, 1648433193.01, 1648433206.73, 1648433220.38, 1648433232.07, 1648433242.70, 1648433253.00, 1648433263.31, 1648433272.60, 1648433282.51, 1648433291.85 ]
}

qcr_traverses_first_times = {
    "bags_2021-08-19-08-25-42": 1e6, # changed from 2e6 to 1e6
    "bags_2021-08-19-09-45-28": 2e6,
    "bags_2022-03-28-11-51-26": 8e6,
    "bags_2022-03-28-12-01-42": 7.7e6,
    "bags_2022-03-28-12-03-44": 8e6, #changed from 12e6 to 8e6
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
path_to_event_files = os.path.dirname(os.path.abspath(__file__)) + "/../data/event_data/"
path_to_frames = os.path.dirname(os.path.abspath(__file__)) + "/../data/input_frames/"
path_to_frames_event_vlad = os.path.dirname(os.path.abspath(__file__)) + "/../data/input_frames_event_vlad/"
path_to_denoised_event_vlad = os.path.dirname(os.path.abspath(__file__)) + "/../data/denoised_frames_event_vlad/"
path_to_gps_files = os.path.dirname(os.path.abspath(__file__)) + "/../data/gps_data/"
path_to_image_files = os.path.dirname(os.path.abspath(__file__)) + "/../data/image_data/"
path_to_pickles = os.path.dirname(os.path.abspath(__file__)) + "/../data/pickles/"
