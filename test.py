import torch
t1=torch.tensor(.4)
print("t1", t1)
print(type(t1))
print(t1.dtype)
#comments 
#Pytorch is an open-source library for deep learning.
#it is a framework for building deep learning models.
#we can get more information about the pytorch library from the official website:
#www.pytorch.org/
#huggingface.co/transformers/ is for learning the NLP

#Below code is for Pytorch for ANN   
#Implementation of Linear Regression using PyTorch
import numpy as np

#imput data defining
input=np.array([[12,55,77],
                [34,54,43],
                [34,66,21],
                [92,72,48],
                [12,99,12]],dtype="float32")
print(input)

#target data
targets=np.array([[55,77],
                    [34,63],
                    [23,67],
                    [89,45],
                    [67,87]],dtype="float32")


#convert input to tensor
inputs=torch.from_numpy(input)
targets=torch.from_numpy(targets)

print("inputs",inputs)
print("targets",targets)


#initialize weights
w=torch.randn(2,3,requires_grad=True) #initializing weights as 2 x 3 matrix based on number of target columns X number of input columns
                   #of input columns.
b=torch.randn(2,requires_grad=True) #initializing bias as 2 x 1 matrix based on number of target columns X number of input columns

print("w",w)
print("b",b)

print("Cuda available: ", torch.cuda.is_available())
print("torch.version.cuda", torch.version.cuda)