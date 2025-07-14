### Setup environment

- Install conda environment
```angular2html
conda env create --file=wavelet_policy.yaml
```

- Activate the conda environment
```angular2html
conda activate wavelet_policy.yaml
```

- To enable wandb logging, set the environment variable `WANDB_MODE`
```angular2html
export WANDB_MODE=online    # To disable, set to disabled
```

- Install D3il-Sim environment
```angular2html
cd environments/d3il && pip install -e .
```

### Download datasets
- Donwload the zip file from https://drive.google.com/file/d/1SQhbhzV85zf_ltnQ8Cbge2lsSWInxVa8/view?usp=drive_link
```angular2html
cd environments/dataset/data
gdown 1SQhbhzV85zf_ltnQ8Cbge2lsSWInxVa8
```

- Extract the data into the folder `environments/dataset/data/`
```angular2html
unzip data.zip
```

### Experiments with WaveletPolicy
- Avoiding (T1) task
```angular2html
bash scripts/avoiding/wavelet_benchmark.sh
```

- Aligning (T2) task
```angular2html
bash scripts/aligning/wavelet_benchmark.sh
```

- Pushing (T3) task
```angular2html
bash scripts/pushing/wavelet_benchmark.sh
```

- Sorting-2 (T4) task
```angular2html
bash scripts/sorting/wavelet_benchmark.sh
```

- Stacking-1 (T5) tasks
```angular2html
bash scripts/stacking/wavelet_benchmark.sh
```