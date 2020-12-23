# NYU ECE-GY 9163 Project (CSAW-HackML-2020 Bad Network Repair)
Group members: Xiaoyu Su (xs1060), Bingchen Wang (bw1839), Zibo Zhang (zz2492) Guandong Kou (gk1675).


## Structure of this repository
```bash
├── data/ # `.h5` data files (too large to upload to GitHub)
├── models/ # backdoored models and their repaired counterpart
├── architecture.py # the original DNN architecture file
├── eval.py # the script for evaluating repaired models
├── repair_model.py # the script for repairing the bad nets
└── repair.sh # bash script for generating all repaired models
```
The validation and test datasets are from [here](https://drive.google.com/drive/folders/13o2ybRJ1BkGUvfmQEeZqDo1kskyFywab?usp=sharing) and should be stored under `data/` directory.


## Dependencies
   1. Python 3.6.9
   2. Keras 2.3.1
   3. Numpy 1.16.3
   4. Matplotlib 2.2.2
   5. H5py 2.9.0
   6. TensorFlow-gpu 1.15.2
   
## Evaluating the Repaired Model
   Evaluate the repaired models `g1`, `g2` and `g3`, execute `eval.py` by running:
      `python3 eval.py <clean evaluation data directory> <repaired model name>`.      
      E.g., `python3 eval.py data/clean_validation_data.h5 g1`.

## Methodology

   1. Iteratively prune the `cov_3` layer (the outmost layer of feature extraction) of the backdoored network ordered by ascending average activations, until the accuracy falls below a specific threshold
   2. Leverage clean validation data to fine tune the pruned network
   3. Evaluate the final network with clean test data and poisoned data

