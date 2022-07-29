# Red vs Blue COCO-Q

This is the repo for solving the Red vs. Blue Scenario 2.  This has used 
COCO-Q and a Tensorflow 2 neural net to solve the various versions of 
scenario 2.

## Setup
The environment.yml file has all required dependencies and assumes 
anaconda is installed.  To set up the environment, execute the following 
command:

`conda env create -f environment.yml`

Once the environment is created, the coco environment can be activated
with the following command:

`conda activate coco`

## Files
| Syntax | Description |
| ----------- | ----------- |
| configs | Directory containing all configurations for training and tests |
| environment.yml | Set up file for creating the anaconda environment |
| Memory.py | Manages previously seen data samples from the environment |
| RvBEnvironment.py | OpenAI Gym type environment that wraps RvB_env's Game |
| RvBFieldSetup.py | Places components on the field with given rules |
| RvBLearner.py | Responsible for creating and updating of models using DDQN |
| RvBSimulation.py | Responsible for playing games and gathering data samples for the Memory |
| RvBUtils.py | Utilities used such as A* pathfinding |
| test_network.py | Runs many simulations for each model in a given directory |
| test_runner.py | Runs and visualizes a simulation with a given model |
| utils.py | COCO methods |

