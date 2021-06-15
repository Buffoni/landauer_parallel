# Landauer

This repository contains the code to run scaling experiments on the emergence of Landauer principle at the microscopic level as described in this [paper](https://arxiv.org/abs/2106.07570).

The code is based on the class object Ising and the erasure protocol.
The code is fully jitted and parallelized by using numba.
Dependencies:
```
numpy 
matplotlib 
numba
```

To run the code edit the parameters in the ``` config.py ``` file and then run
```
python3 landauer.py
```
This will save the results of each run in a pickle file that can be later 
visualized using the notebook.
