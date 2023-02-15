import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

# ROOT=""
# os.chdir(ROOT)
a=torch.Tensor([1,2,3])
print(a)
print("Cuda available: ", torch.cuda.is_available())
print("torch.version.cuda", torch.version.cuda)
class Config:
    def __init__(self):
        self.ROOT_DATA_DIR="FashionMNISTDir"
        self.EPOCH=10
        self.BATCH_SIZE=32
        self.LEARNING_RATE=0.001
        self.IMAGE_SIZE=(28,28)
        self.DEVICE="cuda" if torch.cuda.is_available() else "cpu"
        print(f"This code runs on {self.DEVICE}")
        self.SEED=2022

config=Config()
#Fashion-MNIST is a dataset of Zalando's article images consisting of a training set of 60,000
#examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated 
# with a label from 10 classes.
#dowonloading the train and test datasets from the datasets class
train_dataset=datasets.FashionMNIST(root=config.ROOT_DATA_DIR, train=True,download=True,
transform=transforms.ToTensor()) # To download the data we need to define the dataset.
# Root is the directory where the downloaded dataset is stored.It will be processed trhough
#configuration to get the root directory. Wheather you want to train or test the model, 
# you need to mention the train or test. What are the transformations you want to apply to 
# to the data set you need to define the transform.
#FashionMNISTDir is the directory where the downloaded dataset is stored.
# It will be processed throught configuration to get the root directory.
#ToTensor is a function that transforms the data set into a tensor.
test_dataset=datasets.FashionMNIST(root=config.ROOT_DATA_DIR, train=False,download=True,
transform=transforms.ToTensor()) # To download the data we need to define the dataset.

print("train_dataset.shape", train_dataset.data.shape)
print("test_dataset.shape", test_dataset.data.shape)

print("train_dataset.class_to_idx", train_dataset.class_to_idx) #gives the class to index dictionary
                                                     #i.e.name of the class or categories in the
                                                     #target column of the dataset


print("train_dataset.train_labels", train_dataset.train_labels)

print("train_dataset.targets", train_dataset.targets) #gives the target column data present in the
                                                      #training data set

given_labels=train_dataset.class_to_idx #this gives the key value pair in reverse order

label_map={val:key for key,val in given_labels.items()} #to reverse the key to value and viseversa
print("label_map", label_map)


#visualize one of the sample image
plt.imshow(train_dataset.data[0])
plt.show()
print(label_map[train_dataset.targets[0].item()])
print(train_dataset.targets[0])
print(train_dataset.targets[0].item())

def imgshow(dataset,index,labelmap):
    plt.imshow(dataset.data[index])
    plt.title(f"data_label:{labelmap[dataset.targets[index].item()]}")
    plt.axis("off")
    plt.show()
imgshow(train_dataset,1,label_map)

#create Data loader. This will help us to pass the data to the model or algorithm
train_loader=DataLoader(dataset=train_dataset,batch_size=config.BATCH_SIZE,shuffle=True)
test_loader=DataLoader(dataset=test_dataset,batch_size=config.BATCH_SIZE,shuffle=False)
#This will create a iterator to iterate over the data set and gets the matrix data for the 
# given batch size. i.e. here we will get matrix data for 32 images per iteration.


for images, labels in train_loader:
    print(images.shape)
    print(labels.shape)
    
    break #Here we are only interested in the first batch of data


print("images",images) #contains 32 images data.
print("images.shape",images.shape) # here we have image data set having shape (32,1,28,28)
# this indicates that the each image has size of 28X28 pixels. so each 28 X 28 makes one set
#of data for one image, so we have 32 numbers of such set, that makes 32 image data.
#How to read the shape (32,1,28,28) read from the right hand side, 28 indicates the number of 
# columns makes one row, 28 indicates the number of such rows. 1 indicates the number of 
# 28 X 28 that makes 1 set, 32 indacates how many such 1 set




print("images[0].shape",images[0].shape) # this gives the shape of the first image in the batch
                                        # as 1 X 28 X 28
#But we will not be able to show the image with this shape.
#for that we need to reshape the image data into 28 X 28 
#Squeeze is a function that removes the last dimension of the tensor.


plt.imshow(images[0].squeeze())
plt.show()

#to convert a tensor into a numpy array
print(images[0].squeeze().numpy())

#Similar to squeeze we have unsqueezed the last dimension of the tensor.
print("images[0].unsqueeze.shape",images[0].unsqueeze(1).shape)


#CNN Model building Architecture

class CNN(nn.Module):
    def __init__(self,in_,out_):
        super(CNN,self).__init__()
        self.conv_pool_01=nn.Sequential(
            nn.Conv2d(in_channels=in_,out_channels=8,kernel_size=5,stride=1,padding=0), #The
                #function of Conv2d is to create a 2D convolution layer. in_channels is the number
                # of inputs we want to convolve, out_channels is the number of 
                # outputs we want to generate. Kernel_size is the size of the filter (5 X 5)
                #stride is the steps by which we want to move the filter.
                # padding is used to pad the output with zeros, so the corner information should
                # be the same as the input. 
         ##   nn.BatchNorm2d(32),
            nn.ReLU(), #ReLU is a non-linear activation function. The function of the Relu is
            # to remove the negative values of the input.


            nn.MaxPool2d(kernel_size=2,stride=2)) #The function of MaxPool2d is to create
            # 2d max pooling layer. The max pooling is done by taking the max 
            # value of each pooling window. Here kernel_size indicates the size 
            # of the pooling window and stride indicates the steps by which we want to 
            # move the pooling window.

 #Where do we use Conv1d? We use Conv1d in case of audio signal data set or in text message
 # data set where the data is in a single dimension. Example Wavenet.
    
        self.conv_pool_02=nn.Sequential(
            nn.Conv2d(in_channels=8,out_channels=16,kernel_size=5,stride=1,padding=0),
   #         nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2))

        self.Flatten=nn.Flatten() #Flatten is a function that flattens the tensor.
        # this will make the tensor into a 1D vector by multiplying the all the dimensions
        # of the tensor to flatten it.
        #
        self.FC_01=nn.Linear(in_features=16*4*4,out_features=128)
        self.FC_02=nn.Linear(in_features=128,out_features=64)
        self.FC_03=nn.Linear(in_features=64,out_features=out_ )


    def forward(self,x):
        x=self.conv_pool_01(x)
        x=self.conv_pool_02(x)
        x=self.Flatten(x)
        x=self.FC_01(x)
        x=F.relu(x)
        x=self.FC_02(x)
        x=F.relu(x)
        x=self.FC_03(x)
        x=F.relu(x)
        return x

# The pytorch is similar to the keras only. The advantage of the pytorch is that 
# we can code the each layres from scratch, which gives us a transperancy in understanding
# the model architecture. Some time in keras it is not possible to get the transperancy of 
# the model architecture.

model=CNN(1,10) #1 is the number of input channels, 10 is the number of output classes
print(model)

# Even we can access the each layer of the model by using the name of the layer.
print("model.FC_01()",model.FC_01)

print("model.conv_pool_01()",model.conv_pool_01)

#To check if the model using cuda or not
print("model.cuda()",next(model.parameters()).is_cuda)

model.to(config.DEVICE) #to move the model to the device i.e cuda here 
print("model.cuda()",next(model.parameters()).is_cuda)



#to check the number of countable parameters in the model
def count_parameters(model):
    model_parameters = {"Modules":list(),"Parameters":list()}
    total_parameters = {"trainable":0,"non_trainable":0}
    for name,param in model.named_parameters():
        if not param.requires_grad:
            total_parameters["non_trainable"]+= parameters
            continue
        parameters = param.numel()
        model_parameters["Modules"].append(name)
        model_parameters["Parameters"].append(parameters)
        total_parameters["trainable"]+= parameters

    df=pd.DataFrame(model_parameters)
    print(df)
  #  df=df.style.set_caption(f"Number of Parameters: {total_parameters}" )
    print(df)

    return df

count_parameters(model)
 


#Sage maker is nothing but like a jupyter notebook in AWS

#Training loop

criterion=nn.CrossEntropyLoss() #this is the loss function we want to minimize.

optimizer=torch.optim.Adam(model.parameters(),lr=config.LEARNING_RATE) # this is the optimizer 
                                    #we want to use.
print(len(train_dataset)) #steps per epoch

#number_of_ephochs=60000/32=1875 # 60000 number of records in train data set
for epoch in range(1,(config.EPOCH +1)):
    with tqdm(train_loader) as tqdm_epoch:
        for images, labels in tqdm_epoch:
            tqdm_epoch.set_description(f"Epoch {epoch+1}/{config.EPOCH}")

            #put the images on to the device to use Cuda otherwise it will use CPU


            images=images.to(config.DEVICE)
            labels=labels.to(config.DEVICE)
            #forward pass

         #   optimizer.zero_grad()
            outputs=model(images)
            loss=criterion(outputs,labels) #passing the outputs (pred) and labels(target) to 
                                            #the loss function

            #backward pass
            optimizer.zero_grad() #past gradients from previous epoch

            loss.backward() #calculating the gradients

            optimizer.step() #update the weights of the model

            tqdm_epoch.set_postfix(loss=loss.item()) #item() is used to get the numeric value 
                       #of the loss other wise it will print the tensor output format 

            tqdm_epoch.update(1)

#saving the model

os.makedirs("MODEL_SAVE_PATH",exist_ok=True)
model_file=os.path.join("MODEL_SAVE_PATH","CNN_model.pth")
torch.save(model.state_dict(),model_file)

#To load the model

model2=torch.load(model_file)

#Ecaluation of the model
pred=np.array([])
target=np.array([])

with torch.no_grad():   # this means that we don't want to calculate the gradients and update the
                        # gradients while we evaluating the model.
    for batch, data in enumerate(test_loader):  #enumerate gives the index and the data 
        images=data[0].to(config.DEVICE)
        labels=data[1].to(config.DEVICE)
        y_pred=model(images)

        pred=np.concatenate((pred,torch.argmax(y_pred,1).cpu().numpy())) #concatenate predictions to
                                                #pred array. then store them to CPU and convert them 
                                                #to numpy array.

        target=np.concatenate((target,labels.cpu().numpy())) #concatenate targets to target array.
                                        # then store them to CPU and convert them
                                                            # 
                                                            # 
                                                            # 
confsion_matrix=confusion_matrix(target,pred)
print(confsion_matrix)
plt.figure(figsize=(12,14))
sns.heatmap(confsion_matrix,annot=True,fmt="d",xticklabels=label_map.values(),
              yticklabels=label_map.values(),cbar=False)

plt.show()


#Predicting the model

data=next(iter(test_loader))
print(data)
len(data)
images,labels=data

print("images.shape",images.shape)

idx=2
img=images[idx]
label=labels[idx]
label_map[label.item()]

print("img.shape",img.shape)
plt.imshow(img.squeeze())
plt.show()

#pass this image to the model and get the prediction

logit=model(img.unsqueeze(0).to(config.DEVICE)) #model is on cuda so we need to put the image on the device.
print("logit",logit) # the above gives us a prediction is terms of array of data. so we 
 # need to use softmax to understand what is the predicted image.

predicted_probability=F.softmax(logit,dim=1)

print("predicted_probability",sum(predicted_probability))


argmax=torch.argmax(predicted_probability,dim=1)

print("Predicted image is ",label_map[argmax.item()])


#Let us create a function to create the prediction

def predict(data,image,label_map,device,idx=0):
    images,labels=data
    img=images[idx]
    label=labels[idx]

    plt.imshow(img.squeeze())
    plt.show()
    logit=model(img.unsqueeze(0).to(device))
    predicted_probability=F.softmax(logit,dim=1)
    argmax=torch.argmax(predicted_probability,dim=1)
    predicted_label=label_map[argmax.item()]
    actual_label=label_map[label.item()]

    plt.title(f"actual_label: {actual_label}, predicted_label: {predicted_label}")
    plt.axis("off")
    
    return predicted_label, actual_label

pred,actual=predict(data,model,label_map,config.DEVICE,idx=2)

print("Predicted label",pred)
print("Actual label",actual) 


    








        
