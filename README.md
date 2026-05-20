# Neural rhythms as priors of speech computations

[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Generic badge](https://img.shields.io/badge/DOI-10.17605/OSF.IO/HV7JA-orange.svg)](https://www.doi.org/)

## Introduction 


This repository contains the implementation to reproduce the experiments of the paper [Neural rhythms as priors of speech computations](https://www.sciencedirect.com/science/article/pii/S2666389926000723)


Abstract of the paper: 

>Endogenous rhythms of auditory neural circuits have a striking resemblance to the temporal modulations of incoming speech signals. Here, we show that these rhythms may serve as priors for speech recognition, encoding knowledge of speech structure in the dynamics of network computations. In a network of coupled oscillators, we find that speech is readily identified when characteristic frequencies of the oscillators match low-frequency circuit rhythms in the auditory cortex. When signal and circuit rhythms are mismatched, speech identification is impaired. Compared to a baseline recurrent neural network without intrinsic oscillations, the coupled oscillatory network has significantly higher performance in speech recognition across languages but not in the recognition of signals that lack speech-like structure, such as urban sounds. Our results suggest a central computational role of brain rhythms in speech processing.

## Code

Each experiment lives in its own folder and is run by executing `task.py` from
within that folder, passing the appropriate `--experiment_id`.

**English Spoken Digits**

```bash
cd English
python3 task.py --experiment_id 1
```

**Arabic Spoken Digits**

```bash
cd Arabic
python3 task.py --experiment_id 1
```

**Bengali Spoken Digits**

```bash
cd Bengali
python3 task.py --experiment_id 1
```

**UrbanSounds8k**

```bash
cd UrbanSounds
python3 task.py --experiment_id 1
```

Before running an experiment, download the raw audio data from the
[Zenodo repository](https://zenodo.org/records/20315071) and place each
language's data in its corresponding folder (e.g., `English/data/`).



## Datasets

This repository contains the code to reproduce the results of the following experiments:


- **English Spoken Digits**
- **Arabic Spoken Digits**
- **Bengali Spoken Digits**
- **UrbanSounds8k**

The raw audio data can be found at the the following [Zenodo repository](https://zenodo.org/records/20315071).


## Citation
If you found this work useful, please consider citing:

```bibtex
@article{chandravadia2025neural,
  title={Neural rhythms as priors of speech computations},
  author={Chandravadia, Nand and Imam, Nabil},
  journal={bioRxiv},
  pages={2025--05},
  year={2025}
}
