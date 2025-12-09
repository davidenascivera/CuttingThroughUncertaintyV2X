# CuttingThroughUncertaintyV2X
![Simulated Scenario](images/simulated_situation.png)


CTU is a pedestrian tracking simulation for V2X collaborative perception. The core logic lives in `src/CTU` where `main.py` drives a time-stepped loop that renders roadside units, vehicles and pedestrians; performs raycasting, clutter injection, extended Kalman filtering and greedy data association. All scenario parameters are declared in `constants.py` for easy tuning.

## Setup

```bash
conda env create -f environment.yml
conda activate v2x-perception
```

## Run

```bash
python src/CTU/main.py
```


# Content:
To execute the simulation, simply run the main file. The SBE module contains the dedicated components for Set-Based Estimation, including the linear operators used to compute segment intersections and the Minkowski sum in a linear form.