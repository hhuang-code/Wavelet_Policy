### Setup environment

- Install conda environment
```angular2html
conda env create --file=wavelet_policy.yaml
```
- Activate the conda environment
```angular2html
conda activate wavelet_policy
```
- Clone the Relay Policy Learning repo (for Kitchen tasks):
```angular2html
git clone https://github.com/google-research/relay-policy-learning
```
- Install MuJoCo 2.1.0
```angular2html
Follow this section: https://github.com/openai/mujoco-py?tab=readme-ov-file#install-mujoco
```
- Install CARLA server
```angular2html
Follow this section: https://carla.readthedocs.io/en/0.9.13/start_quickstart/#a-debian-carla-installation
or
follow the instructions listed in NOTE.md
```
- To enable wandb logging, set the environment variable `WANDB_MODE`
```angular2html
export WANDB_MODE=online    # To disable, set to disabled
```

### Download training dataset

- For simulation tasks, we use the same dataset provoided by [BeT](https://github.com/notmahi/bet), and the dataset can be downloaded from [here](https://osf.io/983qz/).
- Unzip the file and run `python process_carla.py carla` to preprocess the CARLA dataset into tensors.
- In `./config/env_vars/env_vars.yaml`, set the dataset paths accordingly:
  - `relay_kitchen`: Franka kitchen environment
  - `carla_multipath_town04_merge`: CARLA environment

### Experiments with Wavelet Policy

Assuming the current directory is the root directory of the project, after training, the model checkpoints are saved in `./exp_local/`.

During evaluation, set the `load_dir` in each environment's configuration file (e.g., `./config/evn/relay_kitchen_traj.yaml`) to the absolute path of the checkpoint directory as mentioned above, **OR**, set the `env.load_dir` in the evaluation bash scripts as described below.


#### Franka kitchen
- Train
```angular2html
bash scripts/kitchen_train.sh
```

- Evaluation
```angular2html
bash scripts/kitchen_test.sh
```

#### CARLA
- Train
```angular2html
bash scripts/carla_train.sh

# Change `eval_path` in `analysis/parse_log.py` to the parent path of the folder storing the `run_on_env.log`.
python analysis/parse_log.py
# Similarly, run `python analysis/parse_log_modes.py` to get the mode analysis results.
```
- Evaluation
```angular2html
bash scripts/carla_test.sh

# Similarly, run `python analysis/parse_log_bet.py` and `python analysis/parse_log_bet_modes.py`.
```

### Errors and Solutions

- X11: The DISPLAY environment variable is missing
```angular2html
export DISPLAY=:0
```

- The GLFW library is not initialized and GLEW initalization error: Missing GL version
```angular2html
sudo apt install libglfw3-dev
sudo apt install libglew-dev
sudo apt install libgl1-mesa-dev
```