# Vector-Quantization---LBG
Python Implementation of Vector Quantization with Linde–Buzo–Gray algorithm
#Running
Just import the module and run, enjoy
# Example
from Vector_Quantization import VQ_LGB
from sklearn import datasets
digits = datasets.load_digits()
dataset = digits.data
#-dataset -codebook_size -alpha -t_iteration_max
vq_lg = VQ_LGB(dataset,64,0.00005,3000)
vq_lg.run()
print(vq_lg.get_codebook())
