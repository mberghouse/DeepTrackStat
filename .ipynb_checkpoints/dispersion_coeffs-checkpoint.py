import pandas as pd
import numpy as np
import cv2
import os
import torch
import re
#torch.set_default_dtype(torch.float16)
k=40


# train_image_files=['../500part_4xspeed_heterogeneous/','../2000part_4xspeed_heterogeneous/', '../2000part_16xspeed_heterogeneous/',
#                   '../2000part_32xspeed_heterogeneous/','../1000part_16xspeed_heterogeneous/','../500part_32xspeed_heterogeneous/',
#                   '../1000part_32xspeed_heterogeneous/', 'sim2/', 'sim11/', 'sim12/', 'sim13/', 'sim14/', 'sim19/','sim20/','sim21/',
#                    'sim22/','sim23/','sim24/','sim25/','sim35/','sim36/','sim46/','sim62/','sim63/','sim64/','sim65/','sim66/',
#                   'sim67/','sim16/']
train_image_files=['sim62/','sim63/','sim64/','sim66/',
                  'sim67/','sim68/','sim69/', 'sim72/','sim73/','sim74/','sim76/',
                  'sim77/','sim78/','sim79/', 'sim82/','sim83/','sim84/','sim86/',
                  'sim87/','sim88/','sim89/']
test_image_files=['sim61/','sim65/','sim70/','sim71/','sim75/','sim80/','sim81/','sim85/','sim90/']
images_train = torch.zeros(len(train_image_files),k,1000, 1000)
images_test = torch.zeros(len(test_image_files),k,1000, 1000)
for i in range(len(train_image_files)):
    base = train_image_files[i]
    im_dir = os.listdir(base)
    im_dir = sorted(im_dir, key=lambda x: int(re.findall(r'\d+', x)[0]))
    for j in range(k):
        images_train[i,j,:,:] = torch.tensor(cv2.resize(cv2.imread(base+im_dir[j], cv2.IMREAD_GRAYSCALE),(1000,1000)))


for i in range(len(test_image_files)):
    base = test_image_files[i]
    im_dir = os.listdir(base)
    im_dir = sorted(im_dir, key=lambda x: int(re.findall(r'\d+', x)[0]))
    for j in range(k):
        images_test[i,j,:,:] = torch.tensor(cv2.resize(cv2.imread(base+im_dir[j], cv2.IMREAD_GRAYSCALE),(1000,1000)))

import numpy as np
from scipy.stats import linregress
#x_positions, y_positions= x_arr,y_arr
def calculate_dispersion(x_positions, y_positions):
    num_frames, num_particles = x_positions.shape
    
    count=0    
    # Calculate the distance between each particle and the center of mass for each frame
    done=0
    #try:
    k=6
    slope2=0
    while k<40:

        #print(k)
        mask = ~np.isnan(x_positions[:k]).any(axis=0)
        x_positions = x_positions[:, mask]
        y_positions = y_positions[:, mask]
        # Calculate the center of mass for each frame
        center_of_mass_x = np.nanmean(x_positions, axis=1)
        center_of_mass_y = np.nanmean(y_positions, axis=1)
        var=[]
        for i in range(len(center_of_mass_x)):
            frame_dist=[]
            for j in range(x_positions.shape[1]):
                d = np.sqrt((x_positions[i,j] - center_of_mass_x[i])**2 +(y_positions[i,j] - center_of_mass_y[i])**2)
                frame_dist.append(d)
            var.append(np.nanvar(frame_dist))
        
        # Calculate the slope of the variance over time using linear regression
        frames = np.arange(num_frames)
        if count==0:
            slope, _, _, _, _ = linregress(frames[0:k-0], var[0:k-0])
            #plt.plot(var)
        else:
            slope2, _, _, _, _ = linregress(frames[0:k-0], var[0:k-0])
        
        if slope2>slope:
            slope=slope2
            k=k+1
        else:
            k=k+1
        count=1
    return slope

import pandas as pd
import numpy as np
import cv2
import os
import torch
import re
import matplotlib.pyplot as plt
from scipy.stats import exponweib,lognorm

def interpolate_vectors(vector, target_length):
    current_length = len(vector)
    # Create a new x-axis with the desired length
    new_x = np.linspace(0, current_length - 1, target_length)
    # Create the old x-axis based on the current length
    old_x = np.arange(current_length)
    # Perform linear interpolation
    interpolated_vector = np.interp(new_x, old_x, vector)
    return interpolated_vector
    
#k=100
# train_dirs=[["xc_500part_4xspeed.csv","yc_500part_4xspeed.csv"],["xc_2000part_4xspeed.csv","yc_2000part_4xspeed.csv"],
#             ["xc_2000part_16xspeed.csv","yc_2000part_16xspeed.csv"],["xc_2000part_32xspeed.csv","yc_2000part_32xspeed.csv"],
#            ["xc_1000part_16xspeed.csv","yc_1000part_16xspeed.csv"],["xc_500part_32xspeed.csv","yc_500part_32xspeed.csv"],
#            ["xc_1000part_32xspeed.csv","yc_1000part_32xspeed.csv"], ['yc_sim2.csv','xc_sim2.csv'], ['yc_sim11.csv','xc_sim11.csv'],
#             ['yc_sim12.csv','xc_sim12.csv'],['yc_sim13.csv','xc_sim13.csv'],['yc_sim14.csv','xc_sim14.csv'],['yc_sim15.csv','xc_sim15.csv'],
#             ['yc_sim19.csv','xc_sim19.csv'],['yc_sim20.csv','xc_sim20.csv'],['yc_sim21.csv','xc_sim21.csv'],['yc_sim22.csv','xc_sim22.csv'],
#             ['yc_sim23.csv','xc_sim23.csv'],['yc_sim24.csv','xc_sim24.csv'],['yc_sim25.csv','xc_sim25.csv'],
#             ['yc_sim35.csv','xc_sim35.csv'],['yc_sim36.csv','xc_sim36.csv'],['yc_sim46.csv','xc_sim46.csv'],['yc_sim62.csv','xc_sim62.csv'],
#             ['yc_sim63.csv','xc_sim63.csv'],['yc_sim64.csv','xc_sim64.csv'],['yc_sim65.csv','xc_sim65.csv'],['yc_sim66.csv','xc_sim66.csv'],
#             ['yc_sim67.csv','xc_sim67.csv'],
#             ['yc_sim16.csv','xc_sim16.csv']]
train_dirs=[['yc_sim62.csv','xc_sim62.csv'],['yc_sim63.csv','xc_sim63.csv'],['yc_sim64.csv','xc_sim64.csv'],['yc_sim66.csv','xc_sim66.csv'],
            ['yc_sim67.csv','xc_sim67.csv'],['yc_sim68.csv','xc_sim68.csv'],['yc_sim69.csv','xc_sim69.csv'],
            ['yc_sim72.csv','xc_sim72.csv'],['yc_sim73.csv','xc_sim73.csv'],['yc_sim74.csv','xc_sim74.csv'],['yc_sim76.csv','xc_sim76.csv'],
            ['yc_sim77.csv','xc_sim77.csv'],['yc_sim78.csv','xc_sim78.csv'],['yc_sim79.csv','xc_sim79.csv'],
            ['yc_sim82.csv','xc_sim82.csv'],['yc_sim83.csv','xc_sim83.csv'],['yc_sim84.csv','xc_sim84.csv'],['yc_sim86.csv','xc_sim86.csv'],
            ['yc_sim87.csv','xc_sim87.csv'],['yc_sim88.csv','xc_sim88.csv'],['yc_sim89.csv','xc_sim89.csv'],
]

train_dirs[0]

# xtorch = torch.zeros(len(train_dirs),1000)
# distr_torch = torch.zeros(len(train_dirs),1000)
dispersions = torch.zeros(len(train_dirs))
# i=0
target_length=1000
speeds = torch.zeros(len(train_dirs),target_length)
for i in range(21):#len(train_dirs)):
    x_arr= np.array(pd.read_csv(train_dirs[i][1], header=None))
    y_arr= np.array(pd.read_csv(train_dirs[i][0], header=None))
    for j in range(x_arr.shape[1]):
        x_arr[:,j] = x_arr[:,j] - x_arr[0,j]
    for j in range(y_arr.shape[1]):
        y_arr[:,j] = y_arr[:,j] - y_arr[0,j]
    dispersion = calculate_dispersion(x_arr, y_arr)
    dispersions[i] = dispersion
    print(f"Dispersion: {dispersion}")
    
    

# images_train = images_train/255
# images_test = images_test/255
#torch.max(scales,dim=0)[0]
torch.mean(dispersions)