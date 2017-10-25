# Vector-Quantization-LBG<br />
Python Implementation of Vector Quantization with Linde–Buzo–Gray algorithm <br />
#Running <br />
Just import the module and run, enjoy <br />
# Example <br />
from Vector_Quantization import VQ_LGB <br />
from sklearn import datasets <br />
digits = datasets.load_digits() <br />
dataset = digits.data <br />
#-dataset -codebook_size -alpha -t_iteration_max <br />
vq_lg = VQ_LGB(dataset,64,0.00005,3000) <br />
vq_lg.run() <br />
print(vq_lg.get_codebook()) <br />
