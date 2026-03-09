# Neural rhythms as priors of speech computations

[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Generic badge](https://img.shields.io/badge/DOI-10.17605/OSF.IO/HV7JA-orange.svg)](https://www.doi.org/)

## Introduction 


This repository contains the implementation to reproduce the experiments of the paper [Neural rhythms as priors of speech computations](https://www.biorxiv.org/content/10.1101/2025.05.06.652542v2.abstract)


Abstract of the paper: 

>The transformation of speech into discrete linguistic representations forms the basis of speech recognition. Natural speech encodes cues at distinct timescales: phonetic features have modulation frequencies of $30$-$50$ Hz, syllables and words around $4$-$7$ Hz, and phrases $1$-$2$ Hz. Strikingly, these frequencies mirror frequencies of endogenous network rhythms of the brain and synaptic time constants of the underlying neural circuits. Here, we suggest that endogenous brain rhythms serve as priors for speech recognition, encoding knowledge of speech structure in the dynamics of network computations. In a network of coupled oscillators, we find that speech is readily identified when characteristic frequencies of the oscillators match frequencies of circuit rhythms in the brain.  When signal and circuit rhythms are mismatched, speech identification is impaired. Compared to a baseline recurrent neural network without intrinsic oscillations, the coupled oscillatory network has significantly higher performance in speech recognition across languages, but not in the recognition of signals that lack speech-like structure, such as urban sounds. Our results suggest a central computational role of brain rhythms in speech processing.


## Code



## Datasets

This repository contains the code to reproduce the results of the following experiments:


- **English Spoken Digits**
- **Arabic Spoken Digits**
- **Bengali Spoken Digits**
- **UrbanSounds8k**


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
