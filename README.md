# MERLIN And DAMP Refined Iteratively on Demand (MADRID)

While stable, this implementation is still a work in progress and can still be further optimized for Python. All credit to the Python DAMP implementation is listed in the respective portion of the README. Please support their code too by visiting the original fork!

This repository contains an unofficial Python implementation of MERLIN And DAMP Refinder Iteratively On Demand (MADRID), introduced in ["MADRID: A Hyper-Anytime Algorithm to Find Time Series Anomalies of all Lengths](https://www.dropbox.com/scl/fi/hd9gt0xs8v8mrsx3upwd3/ICDM23_Madrid_023.pdf?rlkey=s5s95y2eeyk159lx69qn1469e&e=1&dl=0) by Lu, Yue, et al. The official MADRID implementation in MATLAB is available [here](https://sites.google.com/view/madrid-icdm-23/home?authuser=0).

## Run

You can run the code using the following command.

```
python madrid.py
```

You can also import the MADRID function into your own code as a module for use in Jupyter notebooks or other Python scripts.

# Discord Aware Matrix Profile (DAMP)

Authors:

- Siho Han ([@sihohan](https://github.com/sihohan/DAMP))
- Jihwan Min ([@rtm-jihwan-min](https://github.com/rtm-jihwan-min))
- Taeyeong Heo ([@htyvv](https://github.com/htyvv))
- JuI Ma ([@iju298](https://github.com/iju298))

This repository contains an unofficial Python implementation of Discord Aware Matrix Profile (DAMP), introduced in ["Matrix Profile XXIV: Scaling Time Series Anomaly Detection to Trillions of Datapoints and Ultra-fast Arriving Data Streams" (KDD '22)](https://dl.acm.org/doi/abs/10.1145/3534678.3539271). The official MATLAB implementation can be found [here](https://sites.google.com/view/discord-aware-matrix-profile/documentation).

## Project Organization

    ├── data
    |   └── samples
    |       └── BourkeStreetMall.txt
    ├── .gitignore
    ├── README.md
    ├── damp.py
    └── utils.py

## Requirements

- Python >= 3.6
- matplotlib
- numpy

## Datasets

This repository includes Bourke Street Mall as the default dataset (see the `data` directory), which can be downloaded [here](https://sites.google.com/view/discord-aware-matrix-profile/documentation).

## Run

You can run the code using the following command.

```
python damp.py
```

With `--enable_output`, the resulting plot and DAMP values will be saved in the `./figures` and `./outputs` directories, respectively.

Note that the input time series and its corresponding DAMP scores on the plot are scaled for visualization purposes.

## References

- Lu, Yue, et al. "Matrix Profile XXIV: Scaling Time Series Anomaly Detection to Trillions of Datapoints and Ultra-fast Arriving Data Streams." Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining. 2022.
