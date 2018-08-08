# Vector-Quantization-LBG
Python Implementation of Vector Quantization with Linde–Buzo–Gray algorithm proposed by Y. Linde, A. Buzo and R. Gray in the paper "An Algorithm for Vector Quantizer Design"[[ref](https://ieeexplore.ieee.org/document/1094577/)].
# Running
Just import the module and run, enjoy
# Example
## Import the module
```python
from Vector_Quantization import VQ_LGB
```
## Importing a dataset
```python
from sklearn import datasets
digits = datasets.load_digits()
dataset = digits.data
```
## Set the parameters (dataset, codebook_size, alpha, t_iteration_max)
```python
vq_lg = VQ_LGB(dataset,64,0.00005,3000)
```
## Run and get the codebook
```python
vq_lg.run()
print(vq_lg.get_codebook())
```
