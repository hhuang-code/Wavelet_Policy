### Setup environment

- Install conda environment
```angular2html
conda env create --file=wavelet_policy.yaml
```

- Activate the conda environment
```angular2html
conda activate wavelet_policy
```

### Download training dataset
Under the project root, create `data` subfolder:
```angular2html
[project]$ mkdir data && cd data
```

Download `pusht.zip` and `robomimic_lowdim.zip` files from [https://diffusion-policy.cs.columbia.edu/data/training/](https://diffusion-policy.cs.columbia.edu/data/training/)
```angular2html
[data]$ wget https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip
[data]$ wget https://diffusion-policy.cs.columbia.edu/data/training/robomimic_lowdim.zip
```

Extract training data:
```angular2html
[data]$ unzip pusht.zip && cd ..
[data]$ unzip robomimic_lowdim.zip && cd ..
```

### Experiments with Wavelet Policy

Assuming the current directory is the root directory of the project, after training, the model checkpoints are saved in `./outputs/`.

Modify the conda environment in the following base files if necessary.

#### Push-T
- Train
```angular2html
# Set `TASK_NAME` in `scripts/pusht_train.sh` to `lowdim_pusht_ph` for `lowdim` input
bash scripts/pusht_train.sh
```
- Evaluation
```angular2html
bash scripts/pusht_test.sh
```

#### Transport - ph
- Train
```angular2html
# Set `TASK_NAME` in `scripts/transport_ph_train.sh` to `lowdim_transport_ph` for `lowdim` input
bash scripts/transport_ph_train.sh
```
- Evaluation
```angular2html
bash scripts/transport_ph_test.sh
```

#### Transport - mh
- Train
```angular2html
# Set `TASK_NAME` in `scripts/transport_mh_train.sh` to `lowdim_transport_mh` for `lowdim` input
bash scripts/transport_mh_train.sh
```
- Evaluation
```angular2html
bash scripts/transport_mh_test.sh
```