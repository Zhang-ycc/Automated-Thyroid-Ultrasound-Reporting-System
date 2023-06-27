# CV-Project

Project for CV, SJTU SE

## Requirements

### Windows

#### Conda

A suitable conda environment named `CV-Project` can be created and activated with:

```bash
conda env create -f env.yaml
conda activate CV-Project
```

#### Python (**NOT RECOMMAND**)

```bash
pip install -r requirements.txt
```

### MacOS

#### Conda

A suitable conda environment named `CV-Project` can be created and activated with:

```bash
conda env create -f env_OSX.yaml
conda activate CV-Project
```

#### Python (**NOT RECOMMAND**)

```bash
pip install -r requirements_OSX.txt
```

## Usage

All `.ui` files should be generated to `.py` first using

```bash
scripts\gen_all_pyuic.ps1   # Windows
scripts\gen_all_pyuic.sh    # MacOS
```

Run with python

```bash
python main.py
```
