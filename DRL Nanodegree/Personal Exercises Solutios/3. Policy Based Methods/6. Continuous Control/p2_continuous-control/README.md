[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"

# Project 2: Continuous Control

## Introduction

This project was developed on the environment [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher). Actually, on the version of the environment used was the one with 20 different arms.

![Trained Agent][image1]

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1. The environment is considered solved when the agent perform an average of 30 points for the last 100 episodes.

## Getting Started

All the required files can be installed using [Anaconda](https://www.anaconda.com/distribution/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

1. Download the environment from one of the links below, unzip and place the file in this folder. You need only select the environment that matches your operating system:
    - **_Version 2: Twenty (20) Agents_**
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)

2. The commands below create a new environment and install all the requirements to run this project:

    ```shell
    conda create --name mlagents python=3.6
    conda activate mlagents
    conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
    pip -q install ./python
    ```

3. Run the `jupyter notebook` command and execute the file `report.ipynb`.

**OBS.: This notebook was tested on Windows 64-bit with a CUDA capable GPU.**

## Instructions

Follow the instructions in `report.ipynb` to train the agent!