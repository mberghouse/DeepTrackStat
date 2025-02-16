import os
from torch import nn
import torch.nn.functional as F
import gc
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gc
from torch.optim.lr_scheduler import StepLR
import torchvision as tv
import timm
import pandas as pd
import numpy as np
import cv2
import os
import torch
import re

device='cuda'
sim_dir = os.listdir('all_sims')
file_dir = os.listdir('all_traj_files')

    
def sort_key(item):
    if '_brown' in item:
        return (1, int(item[3:-6]))
    else:
        return (0, int(item[3:]))

# Sort the list using the custom key
sorted_sim_dir = sorted(sim_dir, key=sort_key)

print(sorted_sim_dir)

import os


# Custom sorting key function
def sort_key2(item):
    if '_brown' in item:
        return (1, int(item[6:-10]))  # Extract the numeric part for files with '_brown'
    else:
        return (0, int(item[6:-4]))  # Extract the numeric part for files without '_brown'

# Sort the list using the custom key
sorted_file_dir = sorted(file_dir, key=sort_key2)

# Split the sorted list into 'yc_' and 'xc_' files
yc_files = [f for f in sorted_file_dir if f.startswith('yc_')]
xc_files = [f for f in sorted_file_dir if f.startswith('xc_')]

# Create a list of lists with corresponding 'yc_' and 'xc_' files
sorted_file_pairs = [[yc, xc] for yc, xc in zip(yc_files, xc_files)]

print(sorted_file_pairs)

test_dirs = sorted_file_pairs[0::4]
print(len(test_dirs))

train_dirs = sorted_file_pairs[1::4]+sorted_file_pairs[2::4]+sorted_file_pairs[3::4]
print(len(train_dirs))

test_image_files = sorted_sim_dir[0::4]


train_image_files = sorted_sim_dir[1::4]+sorted_sim_dir[2::4]+sorted_sim_dir[3::4]


#torch.set_default_dtype(torch.float16)
k=40


s=384
images_train = torch.zeros(len(train_image_files),k,s, s)
images_test = torch.zeros(len(test_image_files),k,s, s)
for i in range(len(train_image_files)):
    base = 'all_sims/'+train_image_files[i]
    im_dir = os.listdir(base)
    im_dir = sorted(im_dir, key=lambda x: int(re.findall(r'\d+', x)[0]))
    for j in range(k):
        images_train[i,j,:,:] = torch.tensor(cv2.resize(cv2.imread(base+'/'+im_dir[j], cv2.IMREAD_GRAYSCALE),(s,s)))


for i in range(len(test_image_files)):
    base = 'all_sims/'+test_image_files[i]
    im_dir = os.listdir(base)
    im_dir = sorted(im_dir, key=lambda x: int(re.findall(r'\d+', x)[0]))
    for j in range(k):
        images_test[i,j,:,:] = torch.tensor(cv2.resize(cv2.imread(base+'/'+im_dir[j], cv2.IMREAD_GRAYSCALE),(s,s)))
        


base='all_traj_files/'
print(base)
target_length=500


#def calculate_angles(train_dirs):
target_length=500
angles = torch.zeros([len(train_dirs),target_length])
import matplotlib.pyplot as plt
from scipy.stats import exponweib,lognorm, johnsonsb,expon
import torch
import numpy as np

def calculate_angles(x_arr, y_arr):
    traj_stats=[]
    for n in range(x_arr.shape[1]):
        x_diff = np.diff(x_arr[:,n])
        y_diff = np.diff(y_arr[:,n])
        angles=np.zeros((len(x_diff),1))
        for i in range(len(x_diff)-1):
            theta_1=np.arctan(y_diff[i]/x_diff[i])
            theta_2=np.arctan(y_diff[i+1]/x_diff[i+1])
            angles[i]=(theta_2-theta_1)*(180/np.pi)
    
        traj_stats.append([n, angles])
    angle_array=[]
    vel_array=[]
    for i in range(len(traj_stats)):
        section1=traj_stats[i][1]
        section1=np.reshape(section1,(len(section1),))
        angle_array=np.hstack([section1,angle_array])
    
    angle_array
    angle_array=np.abs(angle_array[~np.isnan(angle_array)])
    angle_array = interpolate_vectors(np.sort(angle_array), target_length)
    angle_array[angle_array==0]=0.1
    return angle_array
    
def interpolate_vectors(vector, target_length):
    current_length = len(vector)
    # Create a new x-axis with the desired length
    new_x = np.linspace(0, current_length - 1, target_length)
    # Create the old x-axis based on the current length
    old_x = np.arange(current_length)
    # Perform linear interpolation
    interpolated_vector = np.interp(new_x, old_x, vector)
    return interpolated_vector

scales = torch.zeros(len(train_dirs),2)
# i=0
base='all_traj_files/'
target_length=500
angles = torch.zeros(len(train_dirs),target_length)
for i in range(len(train_dirs)):
    x_arr= np.array(pd.read_csv(base+train_dirs[i][1], header=None))
    y_arr= np.array(pd.read_csv(base+train_dirs[i][0], header=None))
    angle_array = calculate_angles(x_arr,y_arr)

    angles[i,:] = torch.tensor(angle_array)
    
    a,b = expon.fit(angle_array)
    
    # Generate points for plotting the fitted distribution
    xtorch= np.linspace(expon.ppf(0.01, a,b),
                    expon.ppf(0.99, a,b), target_length)
    
    distr_torch=expon.pdf(xtorch, a,b)
    scales[i,:] = torch.tensor([a,b])
    plt.plot(xtorch,distr_torch)
    #plt.show()
    
scales_test = torch.zeros(len(test_dirs),2)
# target_length=1000
angles_test = torch.zeros(len(test_dirs),target_length)
for i in range(len(test_dirs)):
    x_arr= np.array(pd.read_csv(base+test_dirs[i][1], header=None))
    y_arr= np.array(pd.read_csv(base+test_dirs[i][0], header=None))
    angle_array = calculate_angles(x_arr,y_arr)

    angles_test[i,:] = torch.tensor(angle_array)
    
    a,b = expon.fit(angle_array)
    
    # Generate points for plotting the fitted distribution
    xtorch= np.linspace(expon.ppf(0.01, a,b),
                    expon.ppf(0.99, a,b), target_length)
    
    distr_torch=expon.pdf(xtorch, a,b)
    scales_test[i,:] = torch.tensor([a,b])
    plt.plot(xtorch,distr_torch)
    #plt.show()
# torch.max(scales_test,dim=0)
images_train = images_train/255
images_test = images_test/255

#torch.cuda.empty_cache()
gc.collect()

model = timm.create_model('volo_d1_384', in_chans=40, drop_path_rate=.0,num_classes=500,pretrained=True)
model.load_state_dict(torch.load('turn_angle_model_Volod1_384_dispBrownv3'))
model.to(device)
print(model)

class TestDataset(Dataset):
    def __init__(self, images, x):#x,y,scales):
        self.images = images
        self.x = x
        # self.y = y

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index,:,:,:]
        #image  = tv.transforms.Resize(size=1000)(image)
        #image  = torch.tensor(image)
        x = self.x[index,:]
        # y = torch.tensor(self.y[index,:])
        # scales1 = self.scales[index,0]
        # scales2 = self.scales[index,1]
       # scales3 = self.scales[index,2]
        return image.float(), x.float()#,scales1.float(),scales2.float()#,y.float(),  scales.float()
import random
class TrainDataset(Dataset):
    def __init__(self, images, x):#x,y,scales):
        self.images = images
        self.x = x
        # self.y = y

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index,:,:,:]
        #image  = tv.transforms.Resize(size=1000)(image)
        # if random.random() < 0.5:
        #     scale_factor = random.uniform(.95, 1.05)
        #     image *= scale_factor
        if random.random() < 0.5:
            image = image.flip(dims=[-1]) 
        if random.random() < 0.5:
            image = image.flip(dims=[-2])  # Assuming W=50 is the middle axis for flipping

    # # Flip across the horizontal axis (H dimension) for input and label with a 50% chance
        if random.random() < 0.5:
            # # For input, assuming H=200 is the middle
            image = image.flip(dims=[-3])  # Flipping the last dimension (H)
            # # For label, assuming H=1000 is the middle
        

        x = self.x[index,:]
        # y = torch.tensor(self.y[index,:])
        # scales1 = self.scales[index,0]
        # scales2 = self.scales[index,1]
        #scales3 = self.scales[index,2]
        return image.float(), x.float()#,scales1.float(),scales2.float()#,y.float(),  scales.float()


# Define your model architecture

import torch
from torch.utils.data import DataLoader, Sampler
import numpy as np



# Assuming you have your grayscale images and labels loaded in memory
from torch.optim.lr_scheduler import ExponentialLR

dataset = TrainDataset(images_train, angles)#xtorch, distr_torch,scales2)


# Create a dataloader
batch_size = 32
torch.cuda.empty_cache()
gc.collect()

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,drop_last=False,num_workers=0)#,sampler=sampler)
dataset_test = TestDataset(images_test,angles_test)#  xtorch_test, distr_torch_test, scales_test2)
batch_size = 10
test_dataloader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0)

criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(),lr=0.0001,weight_decay=.001)
gamma = .9994
scheduler = ExponentialLR(optimizer, gamma=gamma)
torch.cuda.empty_cache()
gc.collect()

# Training loop
num_epochs = 400
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for im,speed in dataloader:
        # Forward pass
        optimizer.zero_grad()
        outputs= model(im.to(device))

        loss = criterion(outputs, speed.to(device))

        loss.backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()

    # Print the average loss for the epoch
    epoch_loss = running_loss / len(dataloader)
    model.eval()
    running_loss2=0
    running_loss3=0
    with torch.no_grad():
        for im_tst,speed_tst in test_dataloader:
        # Forward pass
            outputs_tst = model(im_tst.to(device))

            loss2 = criterion(outputs_tst, speed_tst.to(device))
 
            running_loss2 += loss2.item()
            running_loss3 += 0#loss3.item()
    epoch_loss2 = running_loss2 / len(test_dataloader)
    epoch_loss3 = running_loss3 / len(test_dataloader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.5f},  Test Loss: {epoch_loss2:.5f}, Test Loss Distribution Params: {epoch_loss3:.5f},Learning Rate: {optimizer.param_groups[0]['lr']:.7f}")
    torch.cuda.empty_cache()
    gc.collect()
print("Training finished!")
torch.save(model.state_dict(), 'turn_angle_model_Volod1_384_dispBrownv4')