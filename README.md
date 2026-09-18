# Regularized Brownian Bridges for Deterministic Medical Image Translation
Official implementation for Reg-BBrg.

## Pre-requeist and Installation

### Required dependencies
* python >= 3.10
* pytorch >= 2.3
* torchmanager >= 1.4
* torchmanager-diffusion >= 1.2

### Optional dependencies for example code
* imageio >= 2.34
* itk >= 5.3
* monai >= 1.3
* nibabel >= 5.2
* tensorboard >= 2.17

To install all requirements:
```bash
pip install -r environments.txt
```

### Installation
To install this work as a package:
```bash
pip install .
```

## Example usage
Use the main `example` package to run datasets in our paper:

```bash
python -m example iseg <data_dir> <output_model_path>
```

### BraTS 2021

Use `brats2021` with a dataset root containing your `Train`, `Val`, and `Test` folders:

```text
BraTS2021/
  Train/
    BraTS2021_01165/
      BraTS2021_01165_flair.nii
      BraTS2021_01165_t1.nii
      BraTS2021_01165_t1ce.nii
      BraTS2021_01165_t2.nii
      BraTS2021_01165_seg.nii
    BraTS2021_.../
  Val/
    BraTS2021_.../
  Test/
    BraTS2021_.../
```

Both `.nii` and `.nii.gz` are supported. Every subject must have all four
modalities and a segmentation map; duplicate compressed/uncompressed copies
are rejected. Channels follow BraTS 2018: FLAIR=0, T1=1, T1CE=2, T2=3.

```bash
python -m example brats2021 <data_dir> <output_model_path> --input_modality 1 --target_modality 3 -b 4
```

Pass the parent of `Train`, `Val`, and `Test` as `<data_dir>`. The loader uses
these existing assignments directly and sorts subject IDs only within each
folder. It does not repartition, exclude, or move subjects, and it imposes no
fixed split counts. All three folders must contain complete subjects; a
subject ID appearing in more than one split is rejected. Nonconsecutive IDs
are supported. Omit `--train_split` for BraTS 2021; that argument remains
available for the other datasets.

Preprocessing follows the existing BraTS 2018 pipeline: RAS orientation,
240x240 in-plane size, segmentation-guided random slabs, and the existing
shared-modality min-max normalization in `MedicalTranslationDataset`.
`-b` controls the slab depth with random sampling enabled.
`--disable_random_sampling` enumerates the full 155 slices per subject;
testing always uses this full-depth setting. Standard 240x240x155 volumes
after orientation are expected. Validation is passed to training and uses
the same sampling policy as training. BraTS 2018 remains available as `brats2018`.

### Batch size and epochs
Use `-b/--batch_size` and `-e/--epochs` to assign the number of batch size and epochs:

```bash
python -m example iseg <data_dir> <output_model_path> -b <batch_size> -e <epochs>
```

### Diffusion and HiFi-BBrg arguments
Use `-t/--time_steps` to control the total number of time steps and `-cycle/--lambda_cycle` for cycle-consistency lambda:

```bash
python -m example iseg <data_dir> <output_model_path> -t <time_steps> -cycle <lambda_cycle>
```

### Display progress bar and specify GPU
Use `--show_verbose` to show training/testing progress bar:

```bash
python -m example iseg <data_dir> <output_model_path> --show_verbose
```

Use `--device` to specify one or more GPUs to run. For distributed parallel training, `--use_multi_gpus` flag is needed:

```bash
python -m example iseg <data_dir> <output_model_path> --device cuda:1  # use cuda:1 to train
```

```bash
python -m example iseg <data_dir> <output_model_path> --device cuda:2 cuda:3 --use_multi_gpus  # use cuda:2 and cuda:3 to train
```

If no devices selected, all available GPUs will be used for distributed parallel training:

```bash
python -m example iseg <data_dir> <output_model_path> --use_multi_gpus  # use all available GPUs to train
```

### Experiment name
Use `-exp/--experiment` for the training experiment name:

```bash
python -m example iseg <data_dir> <output_model_path> -exp <experiment_name>  # all logs and checkpoints will be saved to experiments/<experiment_name>.exp folder
```

If experiment folder exists, an error will raise to protect overwritten. Use `--replace_experiment` to override the duplicate experiment protection:

```bash
python -m example iseg <data_dir> <output_model_path> -exp <exist_experiment_name> --replace_experiment  # all logs and checkpoints will be saved to experiments/<exist_experiment_name>.exp folder
```

### Example scripts
The `example/train.py` and `example/eval.py` scripts are two running scripts for training and evaluation respectively. HiFi-BBrg package need to be installed first before running these scripts.

## API Usage
HiFi-BBrg architecture has been splitted into two parts: a HiFi-BBrg model and an adversarial training manager. To train with HiFi-BBrg, initialize `Manager` with `nn.HiFiBBrgModule`.

### HiFi-BBrg model
The HiFi-BBrg model and pixelwised discriminator can be directly built with `networks.build` method:

```python
input_channels: int = ...
output_channels: int = ...
model, discriminator = networks.build(input_channels, output_channels, time_steps=training_cfgs.time_steps)
```

### Use wrapped training and testing function
The `train` function can be directly used to train with HiFi-BBrg:

```python
# initialize configs
cfgs: configs.TrainingConfigs = ...

# load training and validation dataset
training_dataset: data.TranslationDataset = ...
validation_dataset: data.TranslationDataset = ...

# training function
train(cfgs, training_dataset, model, discriminator, validation_dataset=validation_dataset)
```

The `eval` function can be directly used to train with HiFi-BBrg:

```python
# initialize configs
cfgs: configs.EvalConfigs = ...

# load training and validation dataset
testiung_dataset: data.TranslationDataset = ...

# training function
eval(cfgs, testing_dataset)
```

### Customize training
Compile a `Manager` with customized optimizer and loss functions:

```python
# initialize optimizer and loss function for HiFi-BBrg
optimizer: torch.optim.Optimizer = ...
l_hifibbrg: dict[str, losses.Loss] = ...

# initialize optimizer and loss function for adversarial training
adv_optimizer: torch.optim.Optimizer = ...
l_adv: torchmanager.losses.Loss = ...

# intialize manager
manager = Manager(model, optimizer=optimizer, loss=l_hifibbrg, adversarial_optimizer=adv_optimizer, adversarial_loss_fn=l_adv)
```
