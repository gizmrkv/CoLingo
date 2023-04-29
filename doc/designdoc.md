# CoLingo (Cooperative Language Emergence Library)

## Authors
- gizmrkv

## Purpose
This is a library designed to support experiments in language emergence within environments containing multiple agents.

## Background
There is already an open-source library available for conducting language emergence experiments between two agents, Sender and Receiver ([EGG, 2019](https://github.com/facebookresearch/EGG)). However, this library is not suitable for implementing experiments with three or more agents. On the other hand, there is growing interest in simulating language emergence in environments with a large number of agents, and there is a demand for tools that support this type of experimentation.

## Architecture
The experiments conducted using this library are expected to follow the following process:

1. Generate the agents present in the environment, the dataset to be used in the experiment, and the tasks to be performed in each iteration.
2. Execute the generated tasks repeatedly in an alternating fashion.

It is possible to have multiple agents, datasets, and tasks generated. If multiple tasks are generated, each task is executed once per iteration.



## Components

### Agent
Each agent has a model, a loss function, and an optimizer. Agents are given various types of input depending on the task and are expected to produce various types of output.

### Model
The model is responsible for processing this input and output.

### Loss
The loss function calculates the loss by combining the previous model call result with the reward calculated by the task.

### Baseline
Baseline is a function used to speed up learning convergence, and is a common technique in reinforcement learning.

### Dataset
The dataset is the input data that the task gives to the agent. Various types of data are assumed, including concepts combined with one-hot vectors, strings, and images.

### DataLoader
DataLoader provides a way to create mini-batches by sampling from a specified Dataset in an arbitrary manner.

### Network
Network defines the communication path between Agents. It is a graph structure, managed by an adjacency list and an edge list. Edges can have arbitrary information. Some Tasks require Network.

### Task
The task processing can be freely defined. It is assumed that any task can be executed as a task, including agent learning, metric calculation, model saving, and any other processing that is periodically executed.

### Reward
Reward calculates the reward. Reward allows Tasks not to know the dimensions of the data being passed between Agents.(It might as well be combined with Loss...).

### Logger
Logger logs various values.

### TaskRunner
TaskRunner executes tasks.

## Test
TODO: how to test