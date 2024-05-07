import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gc
from torch.optim.lr_scheduler import StepLR
import torchvision as tv
import timm
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from joblib import Parallel, delayed
from PIL import Image


device='cuda'


class TestDataset2(Dataset):
    def __init__(self, images):
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]

        return image.float()
        
        
        
def interpolate_vectors(vector, target_length):
    current_length = len(vector)
    # Create a new x-axis with the desired length
    new_x = np.linspace(0, current_length - 1, target_length)
    # Create the old x-axis based on the current length
    old_x = np.arange(current_length)
    # Perform linear interpolation
    interpolated_vector = np.interp(new_x, old_x, vector)
    return interpolated_vector
    
def process_file(file_path, l):
    image = np.array(Image.open(file_path))
    resized_image = cv2.resize(image, [l, l])
    resized_image = (resized_image - np.min(resized_image)) / (np.max(resized_image) - np.min(resized_image))
    return resized_image

def parallel_image_processing(full_files, l):
    num_files = len(full_files)
    images = np.zeros([num_files, l, l])
    results = Parallel(n_jobs=16)(delayed(process_file)(file_path, l) for file_path in full_files)
    for i, resized_image in enumerate(results):
        images[i] = resized_image
    return images
    
directory='../tmate_trajectories/'
filenames = os.listdir(directory)
print(filenames)
f_idx = 0
print(filenames[f_idx])

directories = ['test_sims_homo/1000part_16xspeed/','Acid_1ulh_pre/', 'Acid_1ulh_pre/', 
'Acid_1ulh_pre/', 'acid_noflow/open_20x_20fps_001_frames/', 'Geo_1ulh_pre/', 
'Geo_5ulh_pre/', 'paen_noflow/paen_8040_20x_20fps_006_frames/', 'Paen_1ulh_pre/', 
'Paen_5ulh_pre/', 'shew_noflow/8040_20x_002_frames/', 'shew_noflow/808_20x_20fps_005_frames/', 
'test_sims/test1.csv', 'test_sims/test16.csv', 'test_sims/test17.csv', 
'test_sims/test2.csv', 'test_sims/test5.csv', 'test_sims/test8.csv']

base=directories[f_idx]
print(base)
#base='shew_noflow/8040_20x_002_frames/'
files=os.listdir(base)
print(files[0])
#5 for noflow
#3 ofr flow
full_files=[]
if 'noflow' in base:
    sorted_frames = sorted(files, key=lambda x: int(x[5:-4]))
elif 'homo' in base:
    sorted_frames = sorted(files, key=lambda x: int(x[4:-4]))
else:
    sorted_frames = sorted(files, key=lambda x: int(x[3:-4]))

for i in range(len(sorted_frames)):
    file = base+sorted_frames[i]
    full_files.append(file)
    
gc.collect()

l1=448
l2 = 384
l3 = 224


images1 = parallel_image_processing(full_files,l1)
images2 = parallel_image_processing(full_files,l1)

#plt.figure(figsize=(10,10),dpi=500)
#plt.imshow(images1[-1])


images = images1[:,:,:].reshape([(images1.shape[0])//40,40,448,448])
images = torch.tensor(images)

dataset_test = TestDataset2(images)#  xtorch_test, distr_torch_test, scales_test2)
batch_size = 2
test_dataloader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=0)

if (images2.shape[0]-20)%30==0:
    images2 = images2[20:,:,:].reshape([(images2.shape[0]-20)//30,30,448,448])
elif (images2.shape[0]-10)%30==0:
    images2 = images2[10:,:,:].reshape([(images2.shape[0]-10)//30,30,448,448])
else:
    images2 = images2.reshape([(images2.shape[0])//30,30,448,448])

images2 = torch.tensor(images2)

dataset_test = TestDataset2(images2)#  xtorch_test, distr_torch_test, scales_test2)
batch_size = 2
test_dataloader2 = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=0)

x = next(iter(test_dataloader))
#plt.imshow(x[0,29,:,:])


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        l=200
        self.fc2=nn.Conv2d(100, l, 1, stride=2)
        #self.m1 = nn.BatchNorm1d(1000)
        self.d1 = nn.Dropout(.2)
        self.fc1=nn.Conv2d(40, 100, 1, stride=2)
        #self.m2 = nn.BatchNorm1d(l)
        self.d2 = nn.Dropout(.2)
       # self.fc2 = nn.Linear(800,l)
       # self.fc3=nn.Linear(2000,800)
       # self.fc4 = nn.Linear(800,l)
        self.fc5 = nn.Conv2d(l, 5,1, stride=1)
        #self.m3 = nn.BatchNorm1d(100)
        self.fc6 = nn.Linear(312500,1000)
        self.d3 = nn.Dropout(.2)
        self.fc7 = nn.Linear(100,4)
        self.fc8 = nn.Linear(2000,1000)

    def forward(self, x):
        b = x.shape[0]
        x = x.squeeze()
        
        l=200
        x = F.relu(self.fc1(x))
        #x = self.m1(x)
        x = self.d1(x)
        x = F.relu(self.fc2(x))
        #x = self.m2(x)
        x = self.d2(x)
        #x = x.permute(0,2,1)
        #x = F.relu(self.fc3(x))
        #x = F.relu(self.fc4(x))
        
        #x = x.view(b,l*666)
        x = F.relu(self.fc5(x))
        #x = self.m3(x)
        x = x.reshape(b, -1)
        #x = F.relu(self.fc4(x))
        x = (self.fc6(x))
        # x = self.d3(x)
        # #x = torch.flatten(x)
        # x = (self.fc8(x))
        # x = x.reshape(2,200)
        #x = self.fc7(x)
        #output = nn.functional.log_softmax(x, dim=1)
        return x


class Patch_model(nn.Module):
    def __init__(self):
        super(Patch_model, self).__init__()
        self.m1 = timm.create_model('volo_d1_224', in_chans=40, drop_path_rate=0.5,num_classes=500,pretrained=True)
        self.m2 = timm.create_model('volo_d1_224', in_chans=40, drop_path_rate=0.5,num_classes=500,pretrained=True)
        self.m3 = timm.create_model('volo_d1_224', in_chans=40, drop_path_rate=0.5,num_classes=500,pretrained=True)
        self.m4 = timm.create_model('volo_d1_224', in_chans=40, drop_path_rate=0.5,num_classes=500,pretrained=True)


        self.fc0=nn.Linear(2000,200)
        self.fc1=nn.Linear(200,4)
        #self.fc2=nn.Linear(1000,500)


    def forward(self, x):
        l2=224
        x1 = x[:,:,0:l2,0:l2]
        x2 = x[:,:,0:l2,l2:]
        x3 = x[:,:,l2:,0:l2]
        x4 = x[:,:,l2:,l2:]
        #torch.cuda.empty_cache()
        #gc.collect()
        #x = F.gelu(self.m0(x))
        x1 = (self.m1(x1))
        x2 = (self.m2(x2))
        x3 = (self.m3(x3))
        x4 = (self.m4(x4))
        
        #x4 = F.gelu(self.m4(x4))
        x = torch.cat([x1,x2,x3,x4],dim=1)
       # del x1,x2, x3, x4
        #x0 = F.gelu(self.fc0(x0))
        #x = torch.cat([x,x0],dim=1)#x*x0
        #x = F.gelu(self.fc0(x))
        x = F.gelu(self.fc0(x))

        x = F.softmax(self.fc1(x),dim=1)


        return x#,x2,x3,x4\


class Net7(nn.Module):
    def __init__(self):
        super(Net7, self).__init__()
        self.m1 = timm.create_model('regnetx_160.pycls_in1k', in_chans=40,drop_path_rate=0.6, num_classes=4,pretrained=True)

    def forward(self, x):
        x = F.softmax(self.m1(x),dim=1)
        return x


class Net6(nn.Module):
    def __init__(self):
        super(Net6, self).__init__()
        self.m1 = timm.create_model('regnetx_064', in_chans=40,drop_path_rate=0.4, num_classes=4,pretrained=True)

    def forward(self, x):
        x = F.softmax(self.m1(x),dim=1)
        return x
    
class Net5(nn.Module):
    def __init__(self):
        super(Net5, self).__init__()
        self.m1 = timm.create_model('densenet169', in_chans=40, num_classes=4,pretrained=True)

    def forward(self, x):
        x = F.softmax(self.m1(x),dim=1)
        return x

class Net4(nn.Module):
    def __init__(self):
        super(Net4, self).__init__()
        self.m1 = timm.create_model('volo_d1_224', in_chans=40, drop_path_rate=0.9,num_classes=4,pretrained=True)

    def forward(self, x):
        x = F.softmax(self.m1(x),dim=1)
        return x
        
class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.m1 = timm.create_model('volo_d1_384', in_chans=40, drop_path_rate=0.6,num_classes=3,pretrained=True)

    def forward(self, x):
        x = F.softmax(self.m1(x),dim=1)
        return x
    
class Net0(nn.Module):
    def __init__(self):
        super(Net0, self).__init__()
        self.m1 = timm.create_model('volo_d3_448', in_chans=40, drop_path_rate=0.8,num_classes=4,pretrained=True)

    def forward(self, x):
        x = F.softmax(self.m1(x),dim=1)
        return x

class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.m1 = timm.create_model('volo_d1_384', in_chans=40, drop_path_rate=0.6,num_classes=4,pretrained=True)

    def forward(self, x):
        x = F.softmax(self.m1(x),dim=1)
        return x
    
class Net3(nn.Module):
    def __init__(self):
        super(Net3, self).__init__()
        l=80

        self.fc1=nn.Linear(1000,l)
        self.d1 = nn.Dropout(.1)
        self.fc2=nn.Linear(1000,l)
        self.d2 = nn.Dropout(.1)
        self.fc3 = nn.Linear(40*l*l,1000)
        self.d3 = nn.Dropout(.1)
        self.fc4 = nn.Linear(1000,250)
        self.fc4_2 = nn.Linear(250,1000)
#         self.fc4_3 = nn.Linear(800,300)
#         self.fc5 = nn.Linear(300,30)
# #         self.fc5_2 = nn.Linear(200,100)
# #         self.fc5_3 = nn.Linear(100,30)
#         self.fc6 = nn.Linear(30,3)

    def forward(self, x):
        b = x.shape[0]
        x = F.relu(self.fc1(x))
        x = self.d1(x)
        x = x.permute(0,1,3,2)
        x = F.relu(self.fc2(x))
        x = self.d2(x)
        x = torch.reshape(x, [b,-1])
        x = F.relu(self.fc3(x))
        x = self.d3(x)
        x = F.relu(self.fc4(x))
        #x = torch.reshape(x, [b,10000])
        x = (self.fc4_2(x))
#         x = F.rrelu(self.fc4_3(x))
#         x = F.rrelu(self.fc5(x))
# #         x = F.rrelu(self.fc5_2(x))
# #         x = F.rrelu(self.fc5_3(x))
#         x = self.fc6(x)
        #x = torch.reshape(x, [b,2,100])
        #x = self.fc7(x)
        #output = nn.functional.log_softmax(x, dim=1)
        return x
        
model19 = Net0()
model19.load_state_dict(torch.load('models/speed_classifier_volod3_448px_disp200'))
model19.to(device).eval()



model15 = Net7()
model15.load_state_dict(torch.load('models/speed_classifier_regnetx120_500px_disp_all'))
model15.to(device).eval()
    


model21 = Net4()
model21.load_state_dict(torch.load('models/speed_classifier_volod1_224px_disp_all'))
model21.to(device).eval()

model3 = timm.create_model('volo_d3_448', in_chans=30,drop_path_rate=.0, num_classes=4,pretrained=False)
model3.load_state_dict(torch.load('models/speed_classifier_volod1_448px_disp_all_5_04'))
model3.to(device).eval()

model0 = timm.create_model('volo_d1_384', in_chans=30,drop_path_rate=.0, num_classes=4,pretrained=False)
model0.load_state_dict(torch.load('models/speed_classifier_volod1_384px_disp_all_5_04_v3'))
model0.to(device).eval()

model1 = timm.create_model('volo_d1_384', in_chans=30,drop_path_rate=.0, num_classes=4,pretrained=False)
model1.load_state_dict(torch.load('models/speed_classifier_volod1_384px_disp_all_5_04_v2'))
model1.to(device).eval()

model2 = timm.create_model('volo_d1_384', in_chans=30,drop_path_rate=.0, num_classes=4,pretrained=False)
model2.load_state_dict(torch.load('models/speed_classifier_volod1_384px_disp_all_5_04_v1'))
model2.to(device).eval()

model4 = timm.create_model('volo_d4_448', in_chans=30,drop_path_rate=.0, num_classes=5,pretrained=False)
model4.load_state_dict(torch.load('models/speed_classifier_5class_volod4_448px_disp_all_5_04'))
model4.to(device).eval()

model5 = timm.create_model('volo_d3_448', in_chans=30,drop_path_rate=.0, num_classes=5,pretrained=False)
model5.load_state_dict(torch.load('models/speed_classifier_5class_volod3_448px_disp_all_5_05'))
model5.to(device).eval()

model6 = timm.create_model('volo_d1_384', in_chans=30,drop_path_rate=.0, num_classes=5,pretrained=False)
model6.load_state_dict(torch.load('models/speed_classifier_5class_volod1_384px_disp_all_5_06'))
model6.to(device).eval()

model7 = timm.create_model('volo_d1_384', in_chans=30,drop_path_rate=.0, num_classes=5,pretrained=False)
model7.load_state_dict(torch.load('models/speed_classifier_5class_volod1_384px_disp_all_5_06_v2'))
model7.to(device).eval()

model8 = timm.create_model('volo_d1_384', in_chans=30,drop_path_rate=.0, num_classes=5,pretrained=False)
model8.load_state_dict(torch.load('models/speed_classifier_5class_volod1_384px_disp_all_5_06_v3'))
model8.to(device).eval()

model9 = timm.create_model('volo_d1_384', in_chans=30,drop_path_rate=.0, num_classes=5,pretrained=False)
model9.load_state_dict(torch.load('models/speed_classifier_5class_volod1_384px_disp_all_5_06_v4'))
model9.to(device).eval()

model10 = timm.create_model('volo_d1_384', in_chans=30,drop_path_rate=.0, num_classes=5,pretrained=False)
model10.load_state_dict(torch.load('models/speed_classifier_5class_volod1_384px_disp_all_5_06_v5'))
model10.to(device).eval()

model11 = timm.create_model('volo_d1_384', in_chans=30,drop_path_rate=.0, num_classes=5,pretrained=False)
model11.load_state_dict(torch.load('models/speed_classifier_5class_volod1_384px_disp_all_5_06_v6'))
model11.to(device).eval()

model19.eval()

model15.eval()

model21.eval()
model0.eval()
model1.eval()
model2.eval()
# # model10.eval()
# model11.eval()
outputs1=[]
outputs2=[]
out0=0
out1=0
out2=0
out3=0
model4.eval()
model5.eval()
model6.eval()
model7.eval()
model8.eval()
model9.eval()
model10.eval()
model11.eval()
#outputs3=[]

with torch.no_grad():
    for x in test_dataloader2:
        #x1 = torchvision.transforms.functional.resize(x, 448)
        x2 = tv.transforms.functional.resize(x, 384)
        #x3 = torchvision.transforms.functional.resize(x, 224)


        out4 =F.softmax((model4((x).to(device)).detach().cpu()),1).numpy()
        out5 =F.softmax((model5((x).to(device)).detach().cpu()),1).numpy()
        out6 =F.softmax((model6((x2).to(device)).detach().cpu()),1).numpy()
        out7 =F.softmax((model7((x2).to(device)).detach().cpu()),1).numpy()
        out8 =F.softmax((model8((x2).to(device)).detach().cpu()),1).numpy()
        out9 =F.softmax((model9((x2).to(device)).detach().cpu()),1).numpy()
        out10 =F.softmax((model10((x2).to(device)).detach().cpu()),1).numpy()
        out11 =F.softmax((model11((x2).to(device)).detach().cpu()),1).numpy()

        #out=np.mean(np.vstack((out0+out1*3+out2+out3)/6),0)
        out = (out4*1+out5*2+out6*2+out7*6+out8+out9)/13

        outputs1.append(out)

print(np.mean(np.vstack(out6),0))
print(np.mean(np.vstack(out7),0))
print(np.mean(np.vstack(out8),0))
print(np.mean(np.vstack(out9),0))
print(np.mean(np.vstack(out10),0))
print(np.mean(np.vstack(out11),0))


mean_out = np.mean(np.vstack(outputs1),0)
class_num = np.argmax(mean_out)
print(class_num)
print(mean_out)
#class_num=1


class Patch_model2(nn.Module):
    def __init__(self):
        super(Patch_model2, self).__init__()
        self.m0 = timm.create_model('volo_d3_448', in_chans=40, drop_path_rate=0.0,num_classes=2000,pretrained=True)
        self.m1 = timm.create_model('volo_d1_224', in_chans=40, drop_path_rate=0.0,num_classes=500,pretrained=True)
        self.m2 = timm.create_model('volo_d1_224', in_chans=40, drop_path_rate=0.0,num_classes=500,pretrained=True)
        self.m3 = timm.create_model('volo_d1_224', in_chans=40, drop_path_rate=0.0,num_classes=500,pretrained=True)
        self.m4 = timm.create_model('volo_d1_224', in_chans=40, drop_path_rate=0.0,num_classes=500,pretrained=True)

        self.fc1=nn.Linear(2000, 1000)
        self.fc2=nn.Linear(1000,500)


    def forward(self, x):
        l2=224
        x1 = x[:,:,0:l2,0:l2]
        x2 = x[:,:,0:l2,l2:]
        x3 = x[:,:,l2:,0:l2]
        x4 = x[:,:,l2:,l2:]
        #torch.cuda.empty_cache()
        #gc.collect()
        x = F.gelu(self.m0(x))
        x1 = F.gelu(self.m1(x1))
        x2 = F.gelu(self.m2(x2))
        x3 = F.gelu(self.m3(x3))
        x4 = F.gelu(self.m4(x4))
        x0 = torch.cat([x1,x2,x3,x4],dim=1)
        del x1,x2, x3, x4
        x = x*x0
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return x#,x2,x3,x4

class Patch_model3(nn.Module):
    def __init__(self):
        super(Patch_model3, self).__init__()
        self.m1 = timm.create_model('volo_d1_224', in_chans=40, drop_path_rate=0.0,num_classes=500,pretrained=True)
        self.m2 = timm.create_model('volo_d1_224', in_chans=40, drop_path_rate=0.0,num_classes=500,pretrained=True)
        self.m3 = timm.create_model('volo_d1_224', in_chans=40, drop_path_rate=0.0,num_classes=500,pretrained=True)
        self.m4 = timm.create_model('volo_d1_224', in_chans=40, drop_path_rate=0.0,num_classes=500,pretrained=True)

        self.fc1=nn.Linear(2000, 1000)
        self.fc2=nn.Linear(1000,500)


    def forward(self, x):
        l2=224
        x1 = x[:,:,0:l2,0:l2]
        x2 = x[:,:,0:l2,l2:]
        x3 = x[:,:,l2:,0:l2]
        x4 = x[:,:,l2:,l2:]
        #torch.cuda.empty_cache()
        #gc.collect()
        x1 = F.gelu(self.m1(x1))
        x2 = F.gelu(self.m2(x2))
        x3 = F.gelu(self.m3(x3))
        x4 = F.gelu(self.m4(x4))
        x = torch.cat([x1,x2,x3,x4],dim=1)
        del x1,x2, x3, x4
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return x#,x2,x3,x4

class Patch_model4(nn.Module):
    def __init__(self):
        super(Patch_model4, self).__init__()
        self.m1 = timm.create_model('volo_d1_224', in_chans=40, drop_path_rate=0.0,num_classes=500,pretrained=True)
        self.m2 = timm.create_model('volo_d1_224', in_chans=40, drop_path_rate=0.0,num_classes=500,pretrained=True)
        self.m3 = timm.create_model('volo_d1_224', in_chans=40, drop_path_rate=0.0,num_classes=500,pretrained=True)
        self.m4 = timm.create_model('volo_d1_224', in_chans=40, drop_path_rate=0.0,num_classes=500,pretrained=True)

        self.fc0=nn.Linear(2000, 500)
        #self.fc2=nn.Linear(1000,500)


    def forward(self, x):
        l2=224
        x1 = x[:,:,0:l2,0:l2]
        x2 = x[:,:,0:l2,l2:]
        x3 = x[:,:,l2:,0:l2]
        x4 = x[:,:,l2:,l2:]
        #torch.cuda.empty_cache()
        #gc.collect()
        x1 = F.gelu(self.m1(x1))
        x2 = F.gelu(self.m2(x2))
        x3 = F.gelu(self.m3(x3))
        x4 = F.gelu(self.m4(x4))
        x = torch.cat([x1,x2,x3,x4],dim=1)
        del x1,x2, x3, x4
        #x = F.gelu(self.fc1(x))
        x = self.fc0(x)
        return x#,x2,x3,x4


class Patch_model(nn.Module):
    def __init__(self):
        super(Patch_model, self).__init__()
        self.m0 = timm.create_model('volo_d3_448', in_chans=40, drop_path_rate=0.0,num_classes=2000,pretrained=True)
        self.m1 = timm.create_model('volo_d1_224', in_chans=40, drop_path_rate=0.0,num_classes=500,pretrained=True)
        self.m2 = timm.create_model('volo_d1_224', in_chans=40, drop_path_rate=0.0,num_classes=500,pretrained=True)
        self.m3 = timm.create_model('volo_d1_224', in_chans=40, drop_path_rate=0.0,num_classes=500,pretrained=True)
        self.m4 = timm.create_model('volo_d1_224', in_chans=40, drop_path_rate=0.0,num_classes=500,pretrained=True)

        self.fc1=nn.Linear(4000, 1000)
        self.fc2=nn.Linear(1000,500)


    def forward(self, x):
        l2=224
        x1 = x[:,:,0:l2,0:l2]
        x2 = x[:,:,0:l2,l2:]
        x3 = x[:,:,l2:,0:l2]
        x4 = x[:,:,l2:,l2:]
        #torch.cuda.empty_cache()
        #gc.collect()
        x = F.gelu(self.m0(x))
        x1 = F.gelu(self.m1(x1))
        x2 = F.gelu(self.m2(x2))
        x3 = F.gelu(self.m3(x3))
        x4 = F.gelu(self.m4(x4))
        x0 = torch.cat([x1,x2,x3,x4],dim=1)
        del x1,x2, x3, x4
        x = torch.cat([x,x0],dim=1)#x*x0
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return x#,x2,x3,x4

class Net0(nn.Module):
    def __init__(self):
        super(Net0, self).__init__()
        self.m1 = timm.create_model('volo_d1_384', in_chans=40, drop_path_rate=0,num_classes=500,pretrained=True)



    def forward(self, x):
        x = self.m1(x)


        return x#,x2,x3,x4

class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.m1 = timm.create_model('volo_d3_448', in_chans=40, drop_path_rate=0,num_classes=1000,pretrained=True)



    def forward(self, x):
        x = self.m1(x)


        return x#,x2,x3,x4

class Net3(nn.Module):
    def __init__(self):
        super(Net3, self).__init__()
        self.m1 = timm.create_model('volo_d3_448', in_chans=40, drop_path_rate=0,num_classes=500,pretrained=True)



    def forward(self, x):
        x = self.m1(x)


        return x#,x2,x3,x4



model6 = Patch_model3()
model6.load_state_dict(torch.load('models/speed_model_volod1_patch_4x224px_dispBrown_4_21'))
model6.to(device).eval()

model7 = Patch_model()
model7.load_state_dict(torch.load('models/speed_model_Volo224-448_dispBrown_patch_v3'))
model7.to(device).eval()

model8 =Patch_model2()
model8.load_state_dict(torch.load('models/speed_model_Volo224-448_dispBrown_patch_v1'))
model8.to(device).eval()

model17 =Patch_model4()
model17.load_state_dict(torch.load('models/speed_model_patch448px_disp300_4_23_v2'))
model17.to(device).eval()

#model5 = timm.create_model('volo_d3_448', in_chans=40, drop_path_rate=.0,num_classes=1000,pretrained=False)
#model5.load_state_dict(torch.load('models/speed_model_volod3_448px_dispBrown_4_22_v2'))
#model5.to(device).eval()

#model1 = timm.create_model('volo_d3_448', in_chans=40, drop_path_rate=.0,num_classes=1000,pretrained=False)
#model1.load_state_dict(torch.load('models/speed_model_volod3_448px_dispBrown_4_22'))
#model1.to(device).eval()

model2 = timm.create_model('twins_svt_small', in_chans=40, drop_path_rate=.0,num_classes=500,pretrained=False)
model2.load_state_dict(torch.load('models/speed_model_TwinsSvtSmall_500px_opposite_dispBrown_4_21'))
model2.to(device).eval()

model3 = timm.create_model('volo_d2_224', in_chans=40, drop_path_rate=.0,num_classes=500,pretrained=False)
model3.load_state_dict(torch.load('models/speed_model_volod1_224px_disp_4_21'))
model3.to(device).eval()

model4 = timm.create_model('volo_d1_224', in_chans=40, drop_path_rate=.0,num_classes=500,pretrained=False)
model4.load_state_dict(torch.load('models/speed_model_volod1_224px_dispBrown_4_21'))
model4.to(device).eval()

model18 = timm.create_model('volo_d1_384', in_chans=40, drop_path_rate=.0,num_classes=500,pretrained=False)
model18.load_state_dict(torch.load('models/speed_model_384px_disp300_4_23'))
model18.to(device).eval()

model19 = timm.create_model('volo_d2_384', in_chans=40, drop_path_rate=.0,num_classes=500,pretrained=False)
model19.load_state_dict(torch.load('models/speed_model_384px_disp300_4_23_v2'))
model19.to(device).eval()

model20 = timm.create_model('volo_d3_448', in_chans=40, drop_path_rate=.0,num_classes=500,pretrained=False)
model20.load_state_dict(torch.load('models/speed_model_volod3_448px_disp300_4_25'))
model20.to(device).eval()

model9 = Net0()
model9.load_state_dict(torch.load('models/speed_model_Volo384_dispBrown_full'))
model9.to(device).eval()

#model10 = Net0()
#model10.load_state_dict(torch.load('models/speed_model_Volo384_dispBrown_fullv2'))
#model10.to(device).eval()

model11 = Net0()
model11.load_state_dict(torch.load('models/speed_model_Volo384_dispBrown_fullv3'))
model11.to(device).eval()

model12 = Net0()
model12.load_state_dict(torch.load('models/speed_model_Volo384_dispBrown_full_new_v1'))
model12.to(device).eval()

model13 = Net2()
model13.load_state_dict(torch.load('models/speed_model_Volo448_dispBrown'))
model13.to(device).eval()

model14 = Net3()
model14.load_state_dict(torch.load('models/speed_model_Volo448_dispBrown_full'))
model14.to(device).eval()

model15 = timm.create_model('swinv2_small_window16_256', in_chans=40, drop_path_rate=.0,num_classes=500,pretrained=False)
model15.load_state_dict(torch.load('models/speed_model_patch_swin_256px_dispBrown_4_23'))
model15.to(device).eval()

#model16 = timm.create_model('botnet26t_256', in_chans=40, drop_path_rate=.0,num_classes=500,pretrained=False)
#model16.load_state_dict(torch.load('models/speed_model_ByobNet26_256px_dispBrown_4_23'))
#model16.to(device).eval()

model21 = timm.create_model('volo_d1_384', in_chans=35, drop_path_rate=.0,num_classes=500,pretrained=False)
model21.load_state_dict(torch.load('models/speed_model_volod1_384px_disp_lbm_5_04_max216_7189'))
model21.to(device).eval()

model22 = timm.create_model('volo_d1_384', in_chans=30, drop_path_rate=.0,num_classes=500,pretrained=False)
model22.load_state_dict(torch.load('models/speed_model_volod1_384px_disp_lbm_5_04_v3'))
model22.to(device).eval()

model23 = timm.create_model('volo_d1_224', in_chans=30, drop_path_rate=.0,num_classes=500,pretrained=False)
model23.load_state_dict(torch.load('models/speed_model_volod1_384px_disp_all_5_04_v4'))
model23.to(device).eval()

model24 = timm.create_model('volo_d1_384', in_chans=40, drop_path_rate=.0,num_classes=500,pretrained=False)
model24.load_state_dict(torch.load('models/speed_model_volod2_384px_disp_lbm_5_04_max122_8786_v2'))
model24.to(device).eval()

import torchvision
outputs=[]
outputs2=[]
torch.cuda.empty_cache()
gc.collect()
def interpolate_vectors(vector, target_length):
    current_length = len(vector)
    # Create a new x-axis with the desired length
    new_x = np.linspace(0, current_length - 1, target_length)
    # Create the old x-axis based on the current length
    old_x = np.arange(current_length)
    # Perform linear interpolation
    interpolated_vector = np.interp(new_x, old_x, vector)
    return interpolated_vector
    
#model.eval()
#model1.eval()
model2.eval()
model3.eval()
model4.eval()
#model5.eval()
model6.eval()
model7.eval()
model8.eval()
model9.eval()
#model10.eval()
model11.eval()
model12.eval()
model13.eval()
model14.eval()
model15.eval()
#model16.eval()
model17.eval()
model18.eval()
model19.eval()
model20.eval()
model21.eval()
model22.eval()
model23.eval()
model24.eval()

max24 = 122.8786
max21 = 216.7189

with torch.no_grad():
    for x in test_dataloader:
        x=1-x
        x1 = torchvision.transforms.functional.resize(x, 448)
        x2 = torchvision.transforms.functional.resize(x, 224)
        x3 = torchvision.transforms.functional.resize(x, 384)
        x4 = torchvision.transforms.functional.resize(x, 256)
        #out1_1000 = (model1((1-x1).to(device)).detach().cpu().numpy())
        #out1 = np.zeros([batch_size,500])
        #for i in range(len(out1)):
        #    out1[i,:] = interpolate_vectors(out1_1000[i,:],500)
        out2 = np.sort(model2((x).to(device)).detach().cpu().numpy())
        out11 = np.sort(model11((x3).to(device)).detach().cpu().numpy())
        #out10 =(model10((x3).to(device)).detach().cpu().numpy())
        out9 = np.sort(model9((x3).to(device)).detach().cpu().numpy())
        out8 = np.sort(model8((x1).to(device)).detach().cpu().numpy())
        out7 =np.sort(model7((x1).to(device)).detach().cpu().numpy())
        out6 = np.sort(model6((1-x1).to(device)).detach().cpu().numpy())
        out3 = np.sort(model3((1-x2).to(device)).detach().cpu().numpy())
        out4 = np.sort(model4((x2).to(device)).detach().cpu().numpy())
        #out5 = np.sort(model5((x1).to(device)).detach().cpu().numpy())
        out12 = np.sort(model12((x3).to(device)).detach().cpu().numpy())
        #out0 = (model((1-x1).to(device)).detach().cpu().numpy())
        out13_1000 = np.sort(model13((x1).to(device)).detach().cpu().numpy())
        out13 = np.zeros([batch_size,500])
        for i in range(len(out13)):
            out13[i,:] = interpolate_vectors(out13_1000[i,:],500)
        out14 = np.sort(model14((x1).to(device)).detach().cpu().numpy())
        out15 = np.sort(model15((1-x4).to(device)).detach().cpu().numpy())
        #out16 = (model16((1-x4).to(device)).detach().cpu().numpy())
        out17 = np.sort(model17((1-x1).to(device)).detach().cpu().numpy())
        out18 = np.sort(model18((x3).to(device)).detach().cpu().numpy())
        out19 = np.sort(model19((1-x3).to(device)).detach().cpu().numpy())
        out20 = np.sort(model20((1-x1).to(device)).detach().cpu().numpy())
        out21 = np.sort(model21((1-x3[:,5:,:,:]).to(device)).detach().cpu().numpy())*max21
        out22 = np.sort(model22((1-x3[:,5:35,:,:]).to(device)).detach().cpu().numpy())
        out23 = np.sort(model23((1-x2[:,5:35,:,:]).to(device)).detach().cpu().numpy())
        out24 = np.sort(model24((1-x3).to(device)).detach().cpu().numpy())*max24

        out14_2 = np.sort(model14((1-x1).to(device)).detach().cpu().numpy())
        out15_2 = np.sort(model15((x4).to(device)).detach().cpu().numpy())
        #out16 = (model16((1-x4).to(device)).detach().cpu().numpy())
        out17_2 = np.sort(model17((x1).to(device)).detach().cpu().numpy())
        out18_2 = np.sort(model18((1-x3).to(device)).detach().cpu().numpy())
        out19_2 = np.sort(model19((x3).to(device)).detach().cpu().numpy())
        out20_2 = np.sort(model20((x1).to(device)).detach().cpu().numpy())

        
        if class_num==4:
            #1ulh
            #class 3 (x<3)
            out=(out4*6+(out14)*15+out17*25+out18*15+(out17_2)*20+(out18_2)*2+(out19_2)*25+(out20)*10+(out8)*2.5+(out9)*13.5+out23*2+out22*1)/148
        elif class_num==3:
            #class 2 (3<x<6)
            out=(out2*10+out3*4+out4*16+out6*4+out7*1+out11*10+out12*5+out13*4+out14*1+out15*2+out17*10+out18*8+out19*26+out20*2+out9*3+out21*3+out22*2+out23*4+out24*2)/106
        elif class_num==2:
            #5ulh
            #class 1 (6<x<10)
            out=(out3*8+out4*32+out8*15+out11*33+out13*50+out14*5+out18*17+out19*17+out6*5+out2*5+out21*3+out22*2+out23*4+out24*2)/166
            
        else:
            #class 0 (x>10)
            # out=(out3*20+out4*128+out8*20+out11*50+out13*70+out14*5+out18*17+out19*17+out6*5+out2*20+out21*3+out22*2+out23*2+out24*2)/320

            out=(out3*8+out4*64+out8*15+out11*33+out13*50+out14*5+out18*17+out19*17+out6*5+out2*10+out21*3+out22*2+out23*4+out24*2)/190

        outputs.append(out)
        #outputs2.append(out2)

stacked_out = np.vstack(outputs)
out = np.reshape(stacked_out,[stacked_out.shape[0]*stacked_out.shape[1], ])

directory='../tmate_trajectories/'
filename = os.listdir(directory)[f_idx]
print(filename)
df = pd.read_csv(os.path.join(directory,filename),header=0, skiprows=[1,2,3])
tracks = df.TRACK_ID.unique()
vels=1
for i in range(len(tracks)):
    idx = tracks[i]
    if df[df.TRACK_ID==idx].sort_values(by='FRAME').FRAME.iloc[-1] < (images.shape[0]*images.shape[1]):
        posx = df[df.TRACK_ID==idx].sort_values(by='FRAME').POSITION_X
        posy = df[df.TRACK_ID==idx].sort_values(by='FRAME').POSITION_Y

        vel = (np.sqrt(posx.diff()**2+posy.diff()**2)).dropna()
    
        vels = np.hstack([vels, vel])
        
vels=vels[~np.isnan(vels)]
vels = vels[vels<800]
vels = vels[vels>0]

criterion = nn.MSELoss()
speed_loss = criterion(torch.tensor(np.sort(interpolate_vectors(out, len(vels)))), torch.tensor(np.sort(vels)))
print('Speed Loss: ', speed_loss)
print('Absolute Error of Average Speed: ', np.abs(np.mean(vels)-np.mean(out)))