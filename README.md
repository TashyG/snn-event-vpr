# snn-event-vpr
A Spiking Neural Network (SNN) pipeline for Visual Place Recognition (VPR) using data from an event-based camera

# Acknowledgements 
- Code is adapted from https://lava-nc.org/lava-lib-dl/slayer/notebooks/nmnist/train.html
- Code from https://github.com/Tobias-Fischer/salient-event-vpr is also used
- Backpropagation method SLAYER is used. Sumit Bam Shrestha and Garrick Orchard. "SLAYER: Spike Layer Error Reassignment in Time." In Advances in Neural Information Processing Systems, pp. 1417-1426. 2018.

## Create suitable conda environment
1. Create a new conda environment with the dependencies: `conda env create --name snn_event_vpr -f environment.yml`

# Use the pipeline on the Brisbane-Event-VPR Dataset
1. Clone this repository: `git clone git@github.com:TashyG/snn-event-vpr.git`

1. Download the [Brisbane-Event-VPR dataset](https://zenodo.org/record/4302805).

1. Put the parquet files containing the event data under [data/event_data/](./data/event_data/), the GPS files (.nmea) files under [data/gps_data/](./data/gps_data/), and the images under [data/image_data/](./data/image_data/)

1. Run [create_pickles](./scripts/create_pickles.py) to filter the event data and add GPS distance information

1. You can now run the code in [software_pipeline_brisbane.ipynb](./src/software_pipeline_brisbane.ipynb) to train the SNN on place samples from the Brisbane-Event-VPR dataset and see results in [results/](./results/)

# Use the pipeline on the QCR-Event-VPR Dataset
1. Clone this repository: `git clone git@github.com:TashyG/snn-event-vpr.git`

1. Download the QCR-Event-VPR dataset.

1. Put the parquet files containing the event data under [data/event_data/](./data/event_data/) and the images under [data/image_data/](./data/image_data/)

1. You can now run the code in [software_pipeline_qcr.ipynb](./src/software_pipeline_qcr.ipynb) to train the SNN on place samples from the QCR-Event-VPR dataset and see results in [results/](./results/)


