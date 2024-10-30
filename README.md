## Info

This project implements DDQN for optimizing MRI k-space sampling, inspired by the [research paper](https://arxiv.org/abs/2007.10469).

### Overview

This implementation provides a standalone, flexible framework for training and testing DDQN for MRI k-space sampling optimization. The codebase has been updated to work with PyTorch 2.5+ and includes comprehensive comments.

### Key Features

- **Standalone Implementation**: Can be run independently 
- **Flexible Architecture**: Modular design allowing easy modification of network architectures and training parameters only in training and testing scripts
- **Modern Packages**: Updated to work with latest packages 
- **Well-Documented**: Clear documentation of roles and responsibilities for each component (class and function)

### Results

For normal acceleration (normal-acc, Scenario30L):
- Training completed in 48,669 seconds
- Convergence achieved after 712 episodes
- Results show promising performance but differ from the original study



