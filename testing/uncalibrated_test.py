import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gc
from torch.optim.lr_scheduler import StepLR
import torchvision as tv
import timm
import time
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from joblib import Parallel, delayed
from PIL import Image
from scipy.stats import exponweib,lognorm, beta, betaprime, norm, expon, wasserstein_distance
from sklearn.metrics import root_mean_squared_error as RMSE
from sklearn.metrics import r2_score as r2
mean_out=0
import torchvision
#class_num=4
target_length=500
device='cuda'
import warnings
warnings.filterwarnings('ignore')

class Patch_model2(nn.Module):
    def __init__(self):
        super(Patch_model2, self).__init__()
        self.m0 = timm.create_model('volo_d3_448', in_chans=40, drop_path_rate=0.0,num_classes=2000,pretrained=False)
        self.m1 = timm.create_model('volo_d1_224', in_chans=40, drop_path_rate=0.0,num_classes=500,pretrained=False)
        self.m2 = timm.create_model('volo_d1_224', in_chans=40, drop_path_rate=0.0,num_classes=500,pretrained=False)
        self.m3 = timm.create_model('volo_d1_224', in_chans=40, drop_path_rate=0.0,num_classes=500,pretrained=False)
        self.m4 = timm.create_model('volo_d1_224', in_chans=40, drop_path_rate=0.0,num_classes=500,pretrained=False)

        self.fc1=nn.Linear(2000, 1000)
        self.fc2=nn.Linear(1000,500)


    def forward(self, x):
        l2=224
        x1 = x[:,:,0:l2,0:l2]
        x2 = x[:,:,0:l2,l2:]
        x3 = x[:,:,l2:,0:l2]
        x4 = x[:,:,l2:,l2:]
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
        self.m1 = timm.create_model('volo_d1_224', in_chans=40, drop_path_rate=0.0,num_classes=500,pretrained=False)
        self.m2 = timm.create_model('volo_d1_224', in_chans=40, drop_path_rate=0.0,num_classes=500,pretrained=False)
        self.m3 = timm.create_model('volo_d1_224', in_chans=40, drop_path_rate=0.0,num_classes=500,pretrained=False)
        self.m4 = timm.create_model('volo_d1_224', in_chans=40, drop_path_rate=0.0,num_classes=500,pretrained=False)
        self.fc1=nn.Linear(2000, 1000)
        self.fc2=nn.Linear(1000,500)

    def forward(self, x):
        l2=224
        x1 = x[:,:,0:l2,0:l2]
        x2 = x[:,:,0:l2,l2:]
        x3 = x[:,:,l2:,0:l2]
        x4 = x[:,:,l2:,l2:]
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
        self.m1 = timm.create_model('volo_d1_224', in_chans=40, drop_path_rate=0.0,num_classes=500,pretrained=False)
        self.m2 = timm.create_model('volo_d1_224', in_chans=40, drop_path_rate=0.0,num_classes=500,pretrained=False)
        self.m3 = timm.create_model('volo_d1_224', in_chans=40, drop_path_rate=0.0,num_classes=500,pretrained=False)
        self.m4 = timm.create_model('volo_d1_224', in_chans=40, drop_path_rate=0.0,num_classes=500,pretrained=False)
        self.fc0=nn.Linear(2000, 500)

    def forward(self, x):
        l2=224
        x1 = x[:,:,0:l2,0:l2]
        x2 = x[:,:,0:l2,l2:]
        x3 = x[:,:,l2:,0:l2]
        x4 = x[:,:,l2:,l2:]
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
        self.m0 = timm.create_model('volo_d3_448', in_chans=40, drop_path_rate=0.0,num_classes=2000,pretrained=False)
        self.m1 = timm.create_model('volo_d1_224', in_chans=40, drop_path_rate=0.0,num_classes=500,pretrained=False)
        self.m2 = timm.create_model('volo_d1_224', in_chans=40, drop_path_rate=0.0,num_classes=500,pretrained=False)
        self.m3 = timm.create_model('volo_d1_224', in_chans=40, drop_path_rate=0.0,num_classes=500,pretrained=False)
        self.m4 = timm.create_model('volo_d1_224', in_chans=40, drop_path_rate=0.0,num_classes=500,pretrained=False)
        self.fc1=nn.Linear(4000, 1000)
        self.fc2=nn.Linear(1000,500)

    def forward(self, x):
        l2=224
        x1 = x[:,:,0:l2,0:l2]
        x2 = x[:,:,0:l2,l2:]
        x3 = x[:,:,l2:,0:l2]
        x4 = x[:,:,l2:,l2:]
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
        self.m1 = timm.create_model('volo_d1_384', in_chans=40, drop_path_rate=0,num_classes=500,pretrained=False)

    def forward(self, x):
        x = self.m1(x)
        return x#,x2,x3,x4

class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.m1 = timm.create_model('volo_d3_448', in_chans=40, drop_path_rate=0,num_classes=1000,pretrained=False)

    def forward(self, x):
        x = self.m1(x)
        return x#,x2,x3,x4

class Net3(nn.Module):
    def __init__(self):
        super(Net3, self).__init__()
        self.m1 = timm.create_model('volo_d3_448', in_chans=40, drop_path_rate=0,num_classes=500,pretrained=False)

    def forward(self, x):
        x = self.m1(x)
        return x#,x2,x3,x4
        
class Net4(nn.Module):
    def __init__(self):
        super(Net4, self).__init__()
        self.m1 = timm.create_model('volo_d1_384', in_chans=40, drop_path_rate=0.2,num_classes=500,pretrained=False) 
        self.m2 = timm.create_model('volo_d1_384', in_chans=40, drop_path_rate=0.2,num_classes=500,pretrained=False) 
        self.fc1=nn.Linear(2,2)
    def forward(self, x):
        vx = F.gelu(self.m1(x))
        vy = F.gelu(self.m2(x))
        vx = vx.unsqueeze(2)
        vy = vy.unsqueeze(2)
        x = torch.cat([vx,vy],dim=2)
        x = self.fc1(x)
        return x#,x2,x3,x4
        
model0_vx = timm.create_model('volo_d4_448', in_chans=40, drop_path_rate=.0,num_classes=500,pretrained=False)
model0_vx.load_state_dict(torch.load('../models/vx_model_volod4_448px_disp_all_max151_1078_min-49_9241'))
model0_vx.to(device).eval()

model1_vx = timm.create_model('volo_d2_384', in_chans=40, drop_path_rate=.0,num_classes=500,pretrained=False)
model1_vx.load_state_dict(torch.load('../models/vx_model_volod1_384px_noscale_disp_500_4_27'))
model1_vx.to(device).eval()

model2_vx = timm.create_model('volo_d1_384', in_chans=40, drop_path_rate=.0,num_classes=500,pretrained=False)
model2_vx.load_state_dict(torch.load('../models/vx_model_volod1_384px_opposite_disp_300_4_27'))
model2_vx.to(device).eval()

model3_vx = timm.create_model('regnetx_032', in_chans=40, drop_path_rate=.0,num_classes=500,pretrained=False)
model3_vx.load_state_dict(torch.load('../models/vx_model_regnetx32_384px_disp_all_4_27'))
model3_vx.to(device).eval()

model4_vx = timm.create_model('volo_d1_384', in_chans=40, drop_path_rate=.0,num_classes=500,pretrained=False)
model4_vx.load_state_dict(torch.load('../models/vx_model_volod1_384px_disp_all_4_27'))
model4_vx.to(device).eval()

model5_vx = timm.create_model('volo_d3_448', in_chans=40, drop_path_rate=.0,num_classes=500,pretrained=False)
model5_vx.load_state_dict(torch.load('../models/vx_model_volod3_448px_disp_all_4_27'))
model5_vx.to(device).eval()

model6_vx = timm.create_model('regnetx_016', in_chans=40, drop_path_rate=.0,num_classes=500,pretrained=False)
model6_vx.load_state_dict(torch.load('../models/vx_model_regnetx_384px_disp_all_4_27'))
model6_vx.to(device).eval()

model7_vx = Net4()
model7_vx.load_state_dict(torch.load('../models/vel_model_volod1_384px_disp_all_4_27'))
model7_vx.to(device).eval()

model8_vx   = timm.create_model('volo_d3_448', in_chans=30, drop_path_rate=.0,num_classes=500,pretrained=False)
model8_vx.load_state_dict(torch.load('../models/vx_model_volod3_448px_disp_all_max219_541_min-49_9241'))
model8_vx.to(device).eval()
    

    

model0_vy = timm.create_model('volo_d1_384', in_chans=40, drop_path_rate=.0,num_classes=500,pretrained=False)
model0_vy.load_state_dict(torch.load('../models/vy_model_volod1_384px_disp_all_max69_2162_min-66_9786_v3'))
model0_vy.to(device).eval()

model1_vy = timm.create_model('volo_d1_384', in_chans=40, drop_path_rate=.0,num_classes=500,pretrained=False)
model1_vy.load_state_dict(torch.load('../models/vy_model_volod1_384px_disp_all_max69_2162_min-66_9786_v2'))
model1_vy.to(device).eval()

model2_vy = timm.create_model('volo_d1_384', in_chans=40, drop_path_rate=.0,num_classes=500,pretrained=False)
model2_vy.load_state_dict(torch.load('../models/vy_model_volod1_384px_disp_all_max69_2162_min-66_9786'))
model2_vy.to(device).eval()

model3_vy = timm.create_model('volo_d1_224', in_chans=40, drop_path_rate=.0,num_classes=500,pretrained=False)
model3_vy.load_state_dict(torch.load('../models/vy_model_volod1_224px_disp_all_max69_2162_min-66_9786'))
model3_vy.to(device).eval()

model4_vy = timm.create_model('volo_d4_448', in_chans=40, drop_path_rate=.0,num_classes=500,pretrained=False)
model4_vy.load_state_dict(torch.load('../models/vy_model_volod4_448px_disp_all_max64_4513_min-66_1406'))
model4_vy.to(device).eval()

model5_vy = timm.create_model('volo_d1_384', in_chans=40, drop_path_rate=.0,num_classes=500,pretrained=False)
model5_vy.load_state_dict(torch.load('../models/vy_model_volod1_384px_disp_all_4_27'))
model5_vy.to(device).eval()

model6_vy = timm.create_model('volo_d3_448', in_chans=30, drop_path_rate=.0,num_classes=500,pretrained=False)
model6_vy.load_state_dict(torch.load('../models/vy_model_volod3_448px_disp_all_max132_73_min-95_1744'))
model6_vy.to(device).eval()

class Net1_ang(nn.Module):
        def __init__(self):
            super(Net1_ang, self).__init__()
            self.m1 = timm.create_model('volo_d1_384', in_chans=40, drop_path_rate=0.5,num_classes=2000,pretrained=True)
            self.fc1=nn.Linear(2000,1000)
            self.fc2 = nn.Linear(1000,500)

        def forward(self, x):
            x = self.m1(x)
            x = F.gelu(self.fc1(x))
            x = self.fc2(x)

            return x

model10_ang = Net1_ang()
model10_ang.load_state_dict(torch.load('../models/angle_model_net1_volod1_384px_disp_lbm'))
model10_ang.to(device).eval()

model11_ang = Net1_ang()
model11_ang.load_state_dict(torch.load('../models/angle_model_net1_volod1_384px_disp_lbm_v2'))
model11_ang.to(device).eval()

model12_ang = Net1_ang()
model12_ang.load_state_dict(torch.load('../models/angle_model_net1_volod1_384px_disp_lbm_v3'))
model12_ang.to(device).eval()

model13_ang = Net1_ang()
model13_ang.load_state_dict(torch.load('../models/angle_model_net1_volod1_384px_disp_lbm_v4'))
model13_ang.to(device).eval()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        l=200
        self.fc2=nn.Conv2d(100, l, 1, stride=2)
        #self.m1 = nn.BatchNorm1d(1000)
        self.d1 = nn.Dropout(.2)
        self.fc1=nn.Conv2d(40, 100, 1, stride=2)
        #self.m2 = nn.BatchNorm1d(l)

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

        x = F.relu(self.fc5(x))
        #x = self.m3(x)
        x = x.reshape(b, -1)
        #x = F.relu(self.fc4(x))
        x = (self.fc6(x))

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
        x1 = (self.m1(x1))
        x2 = (self.m2(x2))
        x3 = (self.m3(x3))
        x4 = (self.m4(x4))
        
        #x4 = F.gelu(self.m4(x4))
        x = torch.cat([x1,x2,x3,x4],dim=1)

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

        return x
        


model4_cl = timm.create_model('volo_d4_448', in_chans=30,drop_path_rate=.0, num_classes=5,pretrained=False)
model4_cl.load_state_dict(torch.load('../models/speed_classifier_5class_volod4_448px_disp_all_5_04'))
model4_cl.to(device).eval()

model5_cl = timm.create_model('volo_d3_448', in_chans=30,drop_path_rate=.0, num_classes=5,pretrained=False)
model5_cl.load_state_dict(torch.load('../models/speed_classifier_5class_volod3_448px_disp_all_5_05'))
model5_cl.to(device).eval()

model6_cl = timm.create_model('volo_d3_448', in_chans=30,drop_path_rate=.0, num_classes=5,pretrained=False)
model6_cl.load_state_dict(torch.load('../models/speed_classifier_5class_volod3_448px_disp_lbm_5_07'))
model6_cl.to(device).eval()

model7_cl = timm.create_model('volo_d3_448', in_chans=30,drop_path_rate=.0, num_classes=5,pretrained=False)
model7_cl.load_state_dict(torch.load('../models/speed_classifier_5class_volod3_448px_disp_homo_5_07'))
model7_cl.to(device).eval()

model10_cl = timm.create_model('volo_d1_384', in_chans=30,drop_path_rate=.0, num_classes=5,pretrained=False)
model10_cl.load_state_dict(torch.load('../models/speed_classifier_5class_volod1_384px_disp_all_5_06_v2'))
model10_cl.to(device).eval()

model11_cl = timm.create_model('volo_d1_384', in_chans=30,drop_path_rate=.0, num_classes=5,pretrained=False)
model11_cl.load_state_dict(torch.load('../models/speed_classifier_5class_volod1_384px_disp_all_5_06_v3'))
model11_cl.to(device).eval()

model13_cl = timm.create_model('volo_d1_384', in_chans=30,drop_path_rate=.0, num_classes=5,pretrained=False)
model13_cl.load_state_dict(torch.load('../models/speed_classifier_5class_volod1_384px_disp_all_5_06_v5'))
model13_cl.to(device).eval()

model14_cl = timm.create_model('volo_d1_384', in_chans=30,drop_path_rate=.0, num_classes=5,pretrained=False)
model14_cl.load_state_dict(torch.load('../models/speed_classifier_5class_volod1_384px_disp_all_5_06_v6'))
model14_cl.to(device).eval()

model15_cl = timm.create_model('efficientvit_b1', in_chans=30, num_classes=5,pretrained=False)
model15_cl.load_state_dict(torch.load('../models/speed_classifier_efficientvit_b1_mse_448px_all_5_08'))
model15_cl.to(device).eval()

model16_cl = timm.create_model('pvt_v2_b1', in_chans=30,num_classes=5,pretrained=False)
model16_cl.load_state_dict(torch.load('../models/speed_classifier_pvt_v2_b1_mse_448px_all_5_08'))
model16_cl.to(device).eval()

model17_cl = timm.create_model('regnetx_032', in_chans=30,num_classes=5,pretrained=False)
model17_cl.load_state_dict(torch.load('../models/speed_classifier_regnetx_032_mse_448px_all_5_08'))
model17_cl.to(device).eval()

model18_cl = timm.create_model('twins_svt_small', in_chans=30, num_classes=5,pretrained=False)
model18_cl.load_state_dict(torch.load('../models/speed_classifier_twins_svt_small_mse_448px_all_5_09'))
model18_cl.to(device).eval()

model19_cl = timm.create_model('davit_base', in_chans=30, num_classes=5,pretrained=False)
model19_cl.load_state_dict(torch.load('../models/speed_classifier_davit_base_small_mse_448px_all_5_09'))
model19_cl.to(device).eval()

model20_cl = timm.create_model('davit_base', in_chans=30, num_classes=5,pretrained=False)
model20_cl.load_state_dict(torch.load('../models/speed_classifier_davit_base_small_mse_448px_all_5_09_v2'))
model20_cl.to(device).eval()

model21_cl = timm.create_model('regnetx_064', in_chans=30, num_classes=3,pretrained=False)
model21_cl.load_state_dict(torch.load('../models/speed_classifier_regnetx064_mse_448px_all_5_12'))
model21_cl.to(device).eval()

model22_cl = timm.create_model('regnetx_064', in_chans=30, num_classes=3,pretrained=False)
model22_cl.load_state_dict(torch.load('../models/speed_classifier_regnetx064_mse_448px_all_5_12_v2'))
model22_cl.to(device).eval()

model23_cl = timm.create_model('regnetx_064', in_chans=30, num_classes=3,pretrained=False)
model23_cl.load_state_dict(torch.load('../models/speed_classifier_regnetx064_mse_448px_all_5_12_v3'))
model23_cl.to(device).eval()

model24_cl = timm.create_model('regnetx_160', in_chans=30, num_classes=3,pretrained=False)
model24_cl.load_state_dict(torch.load('../models/speed_classifier_regnetx160_mse_448px_all_5_12'))
model24_cl.to(device).eval()

model25_cl = timm.create_model('pvt_v2_b0', in_chans=30, num_classes=3,pretrained=False)
model25_cl.load_state_dict(torch.load('../models/speed_classifier_pvt_v2_b0_mse_384px_all_5_12'))
model25_cl.to(device).eval()

model26_cl = timm.create_model('pvt_v2_b1', in_chans=30, num_classes=3,pretrained=False)
model26_cl.load_state_dict(torch.load('../models/speed_classifier_pvt_v2_b1_mse_384px_all_5_12'))
model26_cl.to(device).eval()

model27_cl = timm.create_model('pvt_v2_b2', in_chans=30, num_classes=3,pretrained=False)
model27_cl.load_state_dict(torch.load('../models/speed_classifier_pvt_v2_b2_mse_384px_all_5_12'))
model27_cl.to(device).eval()


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
        
class Patch_model5(nn.Module):
    def __init__(self):
        super(Patch_model5, self).__init__()
        self.m1 = timm.create_model('volo_d1_224', in_chans=30, drop_path_rate=0.4,num_classes=500,pretrained=True)
        self.m2 = timm.create_model('volo_d1_224', in_chans=30, drop_path_rate=0.4,num_classes=500,pretrained=True)
        self.m3 = timm.create_model('volo_d1_224', in_chans=30, drop_path_rate=0.4,num_classes=500,pretrained=True)
        self.m4 = timm.create_model('volo_d1_224', in_chans=30, drop_path_rate=0.4,num_classes=500,pretrained=True)
        self.fc0=nn.Linear(2000,500)

    def forward(self, x):
        l2=224
        x1 = x[:,:,0:l2,0:l2]
        x2 = x[:,:,0:l2,l2:]
        x3 = x[:,:,l2:,0:l2]
        x4 = x[:,:,l2:,l2:]

        x1 = F.relu(self.m1(x1))
        x2 = F.relu(self.m2(x2))
        x3 = F.relu(self.m3(x3))
        x4 = F.relu(self.m4(x4))

        x = torch.cat([x1,x2,x3,x4],dim=1)
        x = self.fc0(x)
        return x#,x2,x3,x4



model6_speed = Patch_model3()
model6_speed.load_state_dict(torch.load('../models/speed_model_volod1_patch_4x224px_dispBrown_4_21'))
model6_speed.to(device).eval()


model8_speed =Patch_model2()
model8_speed.load_state_dict(torch.load('../models/speed_model_Volo224-448_dispBrown_patch_v1'))
model8_speed.to(device).eval()


model1_speed = timm.create_model('volo_d3_448', in_chans=40, drop_path_rate=.0,num_classes=1000,pretrained=False)
model1_speed.load_state_dict(torch.load('../models/speed_model_volod3_448px_dispBrown_4_22'))
model1_speed.to(device).eval()

model2_speed = timm.create_model('twins_svt_small', in_chans=40, drop_path_rate=.0,num_classes=500,pretrained=False)
model2_speed.load_state_dict(torch.load('../models/speed_model_TwinsSvtSmall_500px_opposite_dispBrown_4_21'))
model2_speed.to(device).eval()

model3_speed = timm.create_model('volo_d2_224', in_chans=40, drop_path_rate=.0,num_classes=500,pretrained=False)
model3_speed.load_state_dict(torch.load('../models/speed_model_volod1_224px_disp_4_21'))
model3_speed.to(device).eval()

model4_speed = timm.create_model('volo_d1_224', in_chans=40, drop_path_rate=.0,num_classes=500,pretrained=False)
model4_speed.load_state_dict(torch.load('../models/speed_model_volod1_224px_dispBrown_4_21'))
model4_speed.to(device).eval()

model18_speed = timm.create_model('volo_d1_384', in_chans=40, drop_path_rate=.0,num_classes=500,pretrained=False)
model18_speed.load_state_dict(torch.load('../models/speed_model_384px_disp300_4_23'))
model18_speed.to(device).eval()

model19_speed = timm.create_model('volo_d2_384', in_chans=40, drop_path_rate=.0,num_classes=500,pretrained=False)
model19_speed.load_state_dict(torch.load('../models/speed_model_384px_disp300_4_23_v2'))
model19_speed.to(device).eval()


model11_speed = Net0()
model11_speed.load_state_dict(torch.load('../models/speed_model_Volo384_dispBrown_fullv3'))
model11_speed.to(device).eval()


model25_speed = timm.create_model('volo_d3_448', in_chans=30, drop_path_rate=.0,num_classes=500,pretrained=False)
model25_speed.load_state_dict(torch.load('../models/speed_model_volod3_448px_disp_lbm_5_07'))
model25_speed.to(device).eval()



model27_speed = Patch_model5()
model27_speed.load_state_dict(torch.load('../models/speed_model_patch_448px_disp_lbm_5_07_v4'))
model27_speed.to(device).eval()


def classifier(dataloader, constants, weights):

    outputs1=[]
    count=0
    with torch.no_grad():
        for x in test_dataloader2:
            if torch.mean(x)>.5:
                x = 1-x
            #x=1-x
            #x1 = tv.transforms.functional.resize(x, 448)
            x2 = tv.transforms.functional.resize(x, 384)

            out5 =F.softmax((model5_cl((x).to(device)).detach().cpu()),1).numpy()
            out6 =F.softmax((model6_cl((x).to(device)).detach().cpu()),1).numpy() # class 4, close to 1
            out7 =F.softmax((model7_cl((1-x).to(device)).detach().cpu()),1).numpy()

            out10 =F.softmax((model10_cl((x2).to(device)).detach().cpu()),1).numpy()
            out11 =F.softmax((model11_cl((x2).to(device)).detach().cpu()),1).numpy()
            out13 =F.softmax((model13_cl((x2).to(device)).detach().cpu()),1).numpy()
            out13[:,3] = out13[:,3]*.1
            out14 =F.softmax((model15_cl((x2).to(device)).detach().cpu()),1).numpy() #class 2 close to cslaa 3
            out16 =F.softmax((model16_cl((x).to(device)).detach().cpu()),1).numpy() #class 2 .25
            out17 =F.softmax((model17_cl((x).to(device)).detach().cpu()),1).numpy() # class 2
            out18 =F.softmax((model18_cl((x).to(device)).detach().cpu()),1).numpy() # class 2
            out19 =F.softmax((model19_cl((x).to(device)).detach().cpu()),1).numpy() # class 3
            out20 =F.softmax((model20_cl((x).to(device)).detach().cpu()),1).numpy() # class 2, .22
            out21 =F.softmax((model21_cl((x).to(device)).detach().cpu()),1).numpy()
            out21z = np.zeros([len(out21), 5])
            out21z[:,2:5] = out21
            out22 =F.softmax((model22_cl((x).to(device)).detach().cpu()),1).numpy()
            out22z = np.zeros([len(out22), 5])
            out22z[:,2:5] = out22
            out23 =F.softmax((model23_cl((x).to(device)).detach().cpu()),1).numpy()
            out23z = np.zeros([len(out23), 5])
            out23z[:,2:5] = out23
            out24 =F.softmax((model24_cl((x).to(device)).detach().cpu()),1).numpy()
            out24z = np.zeros([len(out24), 5])
            out24z[:,2:5] = out24
            out25 =F.softmax((model25_cl((x2).to(device)).detach().cpu()),1).numpy()
            out25z = np.zeros([len(out25), 5])
            out25z[:,2:5] = out25
            out26 =F.softmax((model26_cl((x2).to(device)).detach().cpu()),1).numpy()
            out26z = np.zeros([len(out26), 5])
            out26z[:,2:5] = out26
            out27 =F.softmax((model27_cl((x2).to(device)).detach().cpu()),1).numpy()
            out27z = np.zeros([len(out27), 5])
            out27z[:,2:5] = out27
            
            #out=(out11*5+out14*.5+out16*28.5+out17*22.6+out19*3+out5*17.5+out10*3.4+out7*.6+out13*32.2+out6*2+out18*1.5)/constants[0]#(out16*4+out17*2+out14*.5)/6.5#(out0*2+out3*40+out5*2+out7*2+out10+out13)/48
            ensemble_expression = "+".join(f"out{weight[0]}*{weight[1]}" for weight in weights)
            out = eval(ensemble_expression) / constants[0]
            out[:,0] = out[:,0]+constants[1]
            out[:,1] = out[:,1]+constants[2]
            out[:,2] = out[:,2]+constants[3]
            out[:,3] = out[:,3]+constants[4]
            out[:,4] = out[:,4]+constants[5]
            out = out/1.25

            outputs1.append(out)
            count=count+1

    # print
    return outputs1


class TestDataset2(Dataset):
    def __init__(self, images):
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]

        return image.float()
        
def calculate_velocity( y_positions):
    # Calculate velocities
    y_velocities = (y_positions[1:] - y_positions[:-1]) 
            
    return  y_velocities
        
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
    angle_array = angle_array[angle_array>0]

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
    
def process_file(file_path, l):
    image = np.array(Image.open(file_path))
    resized_image = cv2.resize(image, [l, l])
    resized_image = (resized_image - np.min(resized_image)) / (np.max(resized_image) - np.min(resized_image))
    resized_image = np.nan_to_num(resized_image,1)
    return resized_image

def parallel_image_processing(full_files, l):
    num_files = len(full_files)
    images = np.zeros([num_files, l, l])
    results = Parallel(n_jobs=16)(delayed(process_file)(file_path, l) for file_path in full_files)
    for i, resized_image in enumerate(results):
        images[i] = resized_image
    return images
    
def calculate_angles_test(x_arr, y_arr):
    traj_stats=[]
    x_diff = np.diff(x_arr)
    y_diff = np.diff(y_arr)
    #print(x_diff.shape)
    angles=np.zeros((len(x_diff),1))
    for i in range(len(x_diff)-1):
        theta_1=np.arctan(y_diff[i]/x_diff[i])
        theta_2=np.arctan(y_diff[i+1]/x_diff[i+1])
        angles[i]=(theta_2-theta_1)*(180/np.pi)
        traj_stats.append([angles])
        time.sleep(0.0005)
    
    angles=np.abs(angles[~np.isnan(angles)])
    angles = interpolate_vectors(np.sort(angles), target_length)
    angles=angles[angles!=0]
    #angles[angles==0]=0.1
    return angles
    
# Custom sorting key function
def sort_key2(item):
    if 'het' in item:
        return (0, int(item.split('_')[0]))
    elif 'brown' in item:
        return (1, int(item[6:-10]))
    elif 'test' in item:
        return (3, int(item[7:-4])) # Extract the numeric part for files with '_brown'
    else:
        return (2, int(item[6:-4]))  # Extract the numeric part for files without '_brown'


# sim_base = 'C:/Users/marcb/OneDrive/Desktop/het_trajectories/sim_trajectories/sim/'
# tmate_base = 'C:/Users/marcb/OneDrive/Desktop/het_trajectories/sim_trajectories/tmate/'

sim_directory='ground_truth_trajectories/'
tmate_directory='tmate_trajectories/'
sim_filenames = os.listdir(sim_directory)
tmate_filenames = os.listdir(tmate_directory)

print(sim_filenames)
print(tmate_filenames)

gc.collect()
# Sort the list using the custom key
sorted_file_dir = sorted(sim_filenames, key=sort_key2)
# tmate_directory = sorted(tmate_filenames, key=sort_key2)
# Split the sorted list into 'yc_' and 'xc_' files
yc_files = [f for f in sorted_file_dir if 'yc' in f]
xc_files = [f for f in sorted_file_dir if 'xc' in f]

# Create a list of lists with corresponding 'yc_' and 'xc_' files
sorted_file_pairs = [[yc, xc] for yc, xc in zip(yc_files, xc_files)]

#sorted_file_pairs=sorted_file_pairs[1110:]
print(sorted_file_pairs)
# f_idx = 9
# print(filenames[f_idx])

directories = [
'simulated_imagery/1000part_16xspeed_heterogeneous/','simulated_imagery/1000part_32xspeed_heterogeneous/',
'simulated_imagery/1000part_4xspeed_heterogeneous/','simulated_imagery/1000part_heterogeneous/',
'simulated_imagery/2000part_16xspeed_heterogeneous/','simulated_imagery/2000part_32xspeed_heterogeneous/', ## Group 1
'simulated_imagery/2000part_4xspeed_heterogeneous/','simulated_imagery/2000part_heterogeneous/',
'simulated_imagery/500part_16xspeed_heterogeneous/','simulated_imagery/500part_32xspeed_heterogeneous/',
'simulated_imagery/500part_4xspeed_heterogeneous/','simulated_imagery/500part_heterogeneous/',
'simulated_imagery/sim111_brown/', 'simulated_imagery/sim112_brown/',
'simulated_imagery/sim113_brown/','simulated_imagery/sim114_brown/',  # Group 2 12,13,14,15,16,17,18
'simulated_imagery/sim115_brown/','simulated_imagery/sim119_brown/','simulated_imagery/sim120_brown/',
'simulated_imagery/sim2201/', 'simulated_imagery/sim2210/', 'simulated_imagery/sim2215/',
'simulated_imagery/sim2220/','simulated_imagery/sim2230/','simulated_imagery/sim2235/','simulated_imagery/sim2240/', # Group 3
'simulated_imagery/sim2241/','simulated_imagery/sim2242/','simulated_imagery/sim2243/','simulated_imagery/sim2244/', #Group 4 25,26,27,28
'simulated_imagery/sim2251/','simulated_imagery/sim2252/','simulated_imagery/sim2253/',
'simulated_imagery/sim2254/','simulated_imagery/sim2256/',
'simulated_imagery/sim2257/','simulated_imagery/sim2258/','simulated_imagery/sim2259/','simulated_imagery/sim2260/', # Group 5
'simulated_imagery/test1/','simulated_imagery/test2/','simulated_imagery/test5/','simulated_imagery/test8/' #Group 6
]
#Speed
file_error=[]
file_mse=[]
final_out=[]
final_true=[]
speed_w1=[]
speed_w2=[]
file_mse_pt=[]
final_pt=[]
file_error_pt=[]

#Angle
file_error_ang=[]
file_mse_ang=[]
final_out_ang=[]
final_true_ang=[]
ang_w1=[]
ang_w2=[]
file_mse_pt_ang=[]
file_error_pt_ang=[]
final_pt_ang=[]
#Vx
file_error_vx=[]
file_error_pt_vx=[]
file_mse_vx=[]
final_out_vx=[]
final_true_vx=[]
vx_w1=[]
vx_w2=[]
file_mse_pt_vx=[]
final_pt_vx=[]

#Vx
file_error_vy=[]
file_error_pt_vy=[]
file_mse_vy=[]
final_out_vy=[]
final_out_pt_vy=[]
final_true_vy=[]
vy_w1=[]
vy_w2=[]
file_mse_pt_vy=[]
final_pt_vy=[]


for f_idx in range(0,len(directories)):
    if (12<=f_idx<=18):
        brownian = 1
    else:
        brownian=0
    print('Brownian: ',brownian)
    print(f_idx)
    print(len(tmate_filenames), len(directories))
    base=directories[f_idx]
    
    print(base)
    files=os.listdir(base)
    print(files[0])

    sorted_frames=np.array(files)
    print(len(sorted_frames))
    for i in files:
        a=i.split('_')
        b=int(a[-1].split('.')[0])-1
        sorted_frames[b]=i


    full_files=[]
    for i in range(len(sorted_frames)):
        file = base+sorted_frames[i]
        full_files.append(file)
        
    l1=448
    l2 = 384
    l3 = 224


    images = parallel_image_processing(full_files,l1)
    images2 = parallel_image_processing(full_files,l1)

    plt.figure(figsize=(10,10),dpi=500)

    if (images.shape[0]-30)%40==0:
        images = images[30:,:,:].reshape([(images.shape[0]-30)//40,40,448,448])
    elif (images.shape[0]-20)%40==0:
        images = images[20:,:,:].reshape([(images.shape[0]-20)//40,40,448,448])
    elif (images.shape[0]-10)%40==0:
        images = images[10:,:,:].reshape([(images.shape[0]-10)//40,40,448,448])
    else:
        images = images.reshape([(images.shape[0])//40,40,448,448])
        
    images = torch.tensor(images)

    dataset_test = TestDataset2(images)#  xtorch_test, distr_torch_test, scales_test2)
    if len(images)<4:
        batch_size = 1
    else:
        batch_size = 4
    test_dataloader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=0)

    if (images2.shape[0]-20)%30==0:
        images2 = images2[20:,:,:].reshape([(images2.shape[0]-20)//30,30,448,448])
    elif (images2.shape[0]-10)%30==0:
        images2 = images2[10:,:,:].reshape([(images2.shape[0]-10)//30,30,448,448])
    else:
        images2 = images2.reshape([(images2.shape[0])//30,30,448,448])

    images2 = torch.tensor(images2)

    dataset_test = TestDataset2(images2)#  xtorch_test, distr_torch_test, scales_test2)
    #batch_size = 2
    test_dataloader2 = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=0)


    
        
    constants = [93,-.1,.1,.169,.126,-.066]
    weights = [['11',5],['14',.7],['16',28.25],['17',22.5],['19',2],['5',17.6],['10',3.2],['7',.4],['13',31.4],['6',1.75],['18',1.5]]
    outputs1 = classifier(test_dataloader2,constants,weights)
    mean_out = np.mean(np.vstack(outputs1),0)
    class_num = np.argmax(mean_out)
    print('Class Number: ',class_num)    

    outputs=[]
    outputs2=[]
    torch.cuda.empty_cache()
    gc.collect()
  

    max24 = 140.0
    max21 = 216.7189

    with torch.no_grad():
        for x in test_dataloader:
            if torch.mean(x)<.5:
                x = 1-x

            x3 = tv.transforms.functional.resize(x, 384)
            x4 = tv.transforms.functional.resize(x, 256)

            out11 = np.sort(model11_speed((x3).to(device)).detach().cpu().numpy())

            out8 = np.sort(model8_speed((x).to(device)).detach().cpu().numpy())
            # out7 =np.sort(model7((x).to(device)).detach().cpu().numpy())
            out6 = np.sort(model6_speed((1-x).to(device)).detach().cpu().numpy())

            out19 = np.sort(model19_speed((1-x3).to(device)).detach().cpu().numpy())

            out25 = np.sort(model25_speed((1-x[:,5:35,:,:]).to(device)).detach().cpu().numpy())
            out27 = np.sort(model27_speed((1-x[:,5:35,:,:]).to(device)).detach().cpu().numpy())


            if class_num==4:
                out = 0.95*(out25*35+out27*2+out19*6+out6*8+out8*4+out11)/59
            elif class_num==3:
                out = 1.3*(out25*25+out27*1+out19*4+out6*6+out8*11+out11*8)/55-1
            elif class_num==2:
                out = 1.75*(out25*25+out27*1+out19*3+out6*6+out8*8+out11*5)/48-1.5
            elif class_num==1:
                out = 1.2*(out25*40+out27*1+out19*3+out6*4+out11*2)/50-2
            else:
                out = 1.2*(out25*30+out27*1+out19*3+out6*4)/34-8
            
            outputs.append(np.sort(out))
   
    out = np.sort(np.vstack(outputs))

    print(out.shape)

    # directory='tmate_trajectories/'
    # filename = os.listdir(tmate_directory)[f_idx]
    # print('PT filename: ',filename)
    # df = pd.read_csv(os.path.join(tmate_directory,filename),header=0, skiprows=[1,2,3])
    # tracks = df.TRACK_ID.unique()
    # vels=1
    # for i in range(len(tracks)):
    #     idx = tracks[i]
    #     if df[df.TRACK_ID==idx].sort_values(by='FRAME').FRAME.iloc[-1] < (images.shape[0]*images.shape[1]):
            
    #         posx = df[df.TRACK_ID==idx].sort_values(by='FRAME').POSITION_X
    #         posy = df[df.TRACK_ID==idx].sort_values(by='FRAME').POSITION_Y

    #         vel = (np.sqrt(posx.diff()**2+posy.diff()**2)).dropna()
        
    #         vels = np.hstack([vels, vel])
    #         time.sleep(0.002)
            
    # vels=vels[~np.isnan(vels)]
    # vels = vels[vels<300]
    # vels = vels[vels>0]
    out = out[out<300]
    out = out[out>0]

    
    #vels_gt=1
    print('Ground Truth filenames: ', sorted_file_pairs[f_idx][1], sorted_file_pairs[f_idx][0])
    x_arr= np.array(pd.read_csv(sim_directory+sorted_file_pairs[f_idx][1], header=None))
    y_arr= np.array(pd.read_csv(sim_directory+sorted_file_pairs[f_idx][0], header=None))
    vx = np.diff(x_arr, axis=0)
    vy = np.diff(y_arr, axis=0)
    vels_gt = np.sqrt(vx**2+vy**2)
    vels_gt=vels_gt[~np.isnan(vels_gt)]
    vels_gt = vels_gt[vels_gt>0]
    vels_gt = vels_gt[vels_gt<300]

    mean_loss =  np.abs(np.mean(vels_gt)-np.mean(out))
    # mean_loss_pt =  np.abs(np.mean(vels_gt)-np.mean(vels))
    # print('Mean Speed from PT: ', np.mean(vels))
    print('Mean Speed from Ground Truth: ', np.mean(vels_gt))
    print('Mean Speed from DTS: ', np.mean(out))
    
    print('Absolute Error of Average Speed: ',mean_loss)
    q=np.sort(interpolate_vectors(np.sort(out), 500))
    # e = np.sort(interpolate_vectors(np.sort(vels), 500))
    w=np.sort(interpolate_vectors(np.sort(vels_gt), 500))
    
    speed_loss = RMSE(w, q)
    # speed_loss2 = RMSE(w, e)

    print('Speed Loss (model, PT): ', speed_loss)
    file_error.append(mean_loss)
    # file_error_pt.append(mean_loss_pt)
    file_mse.append(speed_loss)
    # file_mse_pt.append(speed_loss2)

    
    final_out.append(q)
    final_true.append(w)
    # final_pt.append(e)
    
    # plt.figure(figsize=(8, 6))
    a,b,c,d = betaprime.fit(q)

    xtorch1= np.linspace(betaprime.ppf(0.0001, a,b,c,d),
                    betaprime.ppf(0.9999, a,b,c,d), target_length)

    distr_torch1=betaprime.pdf(xtorch1, a,b,c,d)
    # plt.plot(xtorch1,distr_torch1,c='r')


    a,b,c,d = betaprime.fit(w)       
    xtorch= np.linspace(betaprime.ppf(0.0001, a,b,c,d),
                    betaprime.ppf(0.9999, a,b,c,d), target_length)
    distr_torch=betaprime.pdf(xtorch, a,b,c,d)
    # plt.plot(xtorch,distr_torch,c='g')
    
    # a,b,c,d = betaprime.fit(e)
    # xtorch2= np.linspace(betaprime.ppf(0.0001, a,b,c,d),
    #                betaprime.ppf(0.9999, a,b,c,d), target_length)
    # distr_torch2=betaprime.pdf(xtorch2, a,b,c,d)
    
    # plt.plot(xtorch2,distr_torch2,c='b')
    
    # plt.ylabel('Probability Density')
    # plt.xlabel('Speed (pixels/frame)')
    # plt.yscale('log')
    # plt.xscale('log')
    # plt.title('Speed Vectors for '+str(base), fontsize=12)
    # plt.legend([ 'Predictions', 'Ground Truth','Particle Tracking (TrackMate)'])
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig('Results/Speed_'+str(f_idx)+'.png')
    w1 = wasserstein_distance(distr_torch1, distr_torch)
    # w2 = wasserstein_distance(distr_torch2, distr_torch)
    print ('Earth Movers Distance (model, PT): ', w1)
    
    speed_w1.append(w1)
    # speed_w2.append(w2)
    

    
    
    #####################################
    #####   Turn Angle Prediction  ######
    #####################################

    # tracks = df.TRACK_ID.unique()
    # angles=1
    # for i in range(len(tracks)):
    #     idx = tracks[i]
    #     if df[df.TRACK_ID==idx].sort_values(by='FRAME').FRAME.iloc[-1] < (images.shape[0]*images.shape[1]):
    #         posx = df[df.TRACK_ID==idx].sort_values(by='FRAME').POSITION_X
    #         posy = df[df.TRACK_ID==idx].sort_values(by='FRAME').POSITION_Y

    #         angle_array = calculate_angles_test(np.array(posx),np.array(posy))
            
        
    #         angles = np.hstack([angles, angle_array])
            
    # print('Angles Calculated')
    # angles=np.abs(angles[~np.isnan(angles)])
    # angles = angles[angles>0]
    # angles = angles[angles<360]
    # angles = np.sort(angles)
    
    x_arr= np.array(pd.read_csv(sim_directory+sorted_file_pairs[f_idx][1], header=None))
    y_arr= np.array(pd.read_csv(sim_directory+sorted_file_pairs[f_idx][0], header=None))
    print('Calculating GT Angles ')
    print(x_arr.shape)

    traj_stats=[]
    for n in range(x_arr.shape[1]):
        # print(n)
        try:
            x_diff = np.diff(x_arr[:,n])
            y_diff = np.diff(y_arr[:,n])
            angles2=np.zeros((len(x_diff),1))
            for i in range(len(x_diff)-1):
                if x_diff[i]!=0 and x_diff[i+1]!=0:
                    theta_1=np.arctan(y_diff[i]/x_diff[i])
                    theta_2=np.arctan(y_diff[i+1]/x_diff[i+1])
                    angles2[i]=(theta_2-theta_1)*(180/np.pi)
                else:
                    angles2[i]=0
        except:
            angles2=np.array([0])
        time.sleep(0.001)
        traj_stats.append([n, angles2])
        
    angles_gt=[]
    for i in range(len(traj_stats)):
        section1=traj_stats[i][1]
        section1=np.reshape(section1,(len(section1),))
        angles_gt=np.hstack([section1,angles_gt])
    
    
    angles_gt=np.abs(angles_gt[~np.isnan(angles_gt)])
    angles_gt = angles_gt[angles_gt>0]
    
    angles_gt = angles_gt[angles_gt<360]


    # For videos where trajectories are observed to be primarily straight, we use a different class. 
    #Here, we use the vector "straight_indices" to indicate which videos contain straight trajectories
    straight_indices = [26,27,28,29]

    outputs_ang=[]
    with torch.no_grad():
        for x in test_dataloader:
            if torch.mean(x)>.5:
                x = 1-x
            #x1 = torchvision.transforms.functional.resize(x, 448)
            x2 = tv.transforms.functional.resize(x, 384)
            out0 =np.sort(model10_ang((x2).to(device)).detach().cpu().numpy())
            out1 =np.sort(model11_ang((x2).to(device)).detach().cpu().numpy())
            out2 =np.sort(model12_ang((x2).to(device)).detach().cpu().numpy())
            out3 =np.sort(model13_ang((x2).to(device)).detach().cpu().numpy())


            out = (out0*2+out1*3+out2*3+out3*4)/12
            out[:,0] = 0.0001

            outputs_ang.append(np.sort(out))
   
    
    
    if (np.min(straight_indices)<=f_idx <=np.max(straight_indices)):
        if np.mean(out)>10:
            out = out*.4

    out = np.sort(out)
    
    out = out[out>0]
    out = out[out<360]

    
    # mean_loss_pt =  np.abs(np.mean(angles)-np.mean(angles_gt))
    mean_loss_gt =  np.abs(np.mean(angles_gt)-np.mean(out))
    #print('Min, Max: ', np.min(angles), np.max(angles))
    print('Min out, Max out: ', np.min(out), np.max(out))
    # print('Mean Turn Angle from PT: ', np.mean(angles))
    print('Mean Turn Angle from DeepTrackStat: ', np.mean(out))
    print('Mean Turn Angle from Ground Truth: ', np.mean(angles_gt))
    
    print('Absolute Error of Average Turn Angle (DTS): ',mean_loss_gt)
    # print('Absolute Error of Average Turn Angle (PT): ',mean_loss_pt)
    q=np.sort(interpolate_vectors(np.sort(out), 500))
    # e = np.sort(interpolate_vectors(np.sort(angles), 500))
    w = np.sort(interpolate_vectors(np.sort(angles_gt), 500))
    speed_loss = RMSE(w, q)
    # speed_loss_pt = RMSE(w, e)

    print('Turn Angle Model Loss: ', speed_loss)  
    # print('Turn Angle PT Loss: ', speed_loss_pt) 
    file_error_ang.append(mean_loss_gt)
    # file_error_pt_ang.append(mean_loss_pt)
    file_mse_ang.append(speed_loss)
    # file_mse_pt_ang.append(speed_loss_pt)
    
    # plt.figure(figsize=(8, 6))
    a,b = expon.fit(q)
    xtorch1= np.linspace(expon.ppf(0.0001, a,b),
                    expon.ppf(0.9999, a,b), target_length)
    distr_torch1=expon.pdf(xtorch1, a,b)
    # plt.plot(xtorch1,distr_torch1,c='r', linewidth=3)
    
    # a,b = expon.fit(e)
    # xtorch2= np.linspace(expon.ppf(0.0001, a,b),
    #                 expon.ppf(0.9999, a,b), target_length)
    # distr_torch2=expon.pdf(xtorch2, a,b)
    # plt.plot(xtorch2,distr_torch2,c='g', linewidth=3)
    
    a,b = expon.fit(w)     
    xtorch= np.linspace(expon.ppf(0.0001, a,b),
                    expon.ppf(0.9999, a,b), target_length)
                    
    # plt.xlim([-5, 180])
    # plt.ylim([1e-4, 1e-1])
    distr_torch=expon.pdf(xtorch, a,b)
    # plt.plot(xtorch,distr_torch,c='b', linewidth=3)
    # plt.ylabel('Probability Density', fontsize=16)
    # plt.xlabel('Turn Angle (degrees)', fontsize=16)
    # plt.title('Turn Angle Vectors for '+str(base)+', Class Number = '+str(class_num) , fontsize=14)
    # plt.legend(['Predictions', 'TrackMate', 'Ground Truth'], fontsize=16)
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.tick_params(axis='both', which='major', labelsize=16)
    # plt.tick_params(axis='both', which='minor', labelsize=16)
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig('Results/Turn_angle_noStraightTrajFlag'+str(f_idx)+'.png')
    w1 = wasserstein_distance(distr_torch1, distr_torch)
    # w2 = wasserstein_distance(distr_torch2, distr_torch)
    print ('Earth Movers Distance (model, PT): ', w1)
    ang_w1.append(w1)
    final_out_ang.append(q)
    final_true_ang.append(w)
    # final_pt_ang.append(e)
    # ang_w2.append(w2)
    
    
    ##################
    #####   Vx  ######
    ##################
    
    print('Starting Vx')


    outputs=[]
    outputs2=[]
    torch.cuda.empty_cache()
    gc.collect()


    slope200, int200 = 182.7808, -16.0141
    slope_all, int_all = 216.6908, -49.9241
    slope_all2, int_all2 = 230, -45

    max1 = 151.1078
    min1 = -49.9241
    slope1 = max1-min1
    int1 = min1

    max2 = 219.541
    min2 = -49.9241
    slope2 = max2-min2
    int2 = min2

    with torch.no_grad():
        for x in test_dataloader:
            if torch.mean(x)<.5:
                x = 1-x
            #x1 = torchvision.transforms.functional.resize(x, 448)
            out0 = np.sort(model0_vx((x).to(device)).detach().cpu().numpy())*slope1+int1
           # # x=1-x
            x2 = torchvision.transforms.functional.resize(x, 384)
            
            out1 = np.sort(model1_vx((x2).to(device)).detach().cpu().numpy())
            out2 = np.sort(model2_vx((1-x2).to(device)).detach().cpu().numpy())*slope200+int200
            out4 = np.sort(model4_vx((x2).to(device)).detach().cpu().numpy())*slope_all+int_all
            out5 = np.sort(model5_vx((x).to(device)).detach().cpu().numpy())*slope_all2+int_all2
            out6 = np.sort(model6_vx((x).to(device)).detach().cpu().numpy())*50
            out7= np.sort(model7_vx((x2).to(device)).detach().cpu().numpy())*slope_all2+int_all2
            out7 = out7[:,:,0].squeeze()
            out12= 24*(np.sort(model7_vx((x2).to(device)).detach().cpu().numpy())*slope_all2+int_all2)
            out13= np.sort(-23*(np.sort(model7_vx((x2).to(device)).detach().cpu().numpy())*slope_all2+int_all2))
            out12 = out12[:,:,0].squeeze()
            out13 = out13[:,:,0].squeeze()
            out15 = (out12*3+out13)/4
            out16 = (out12*1+out13)/2
            out8 = out7*1.5-10
            out9 = 5*(out7*1.3-17)
            out10 = np.sort(model8_vx((x[:,5:35,:,:]).to(device)).detach().cpu().numpy())*slope2+int2
            out5_2 = out5*2

            out11 = (out0-1.0)*4
            out14 = (out6*6+out4*10+out7*4+out8*2+out0*2+out1*2+out5+out16*4+out10*5)/35.5
            out15 = (out6*2+out4*4+out10)/7
            
            # ##class1
            if class_num==0:
                out=8*(out15*1+out0*2+out14*13+out6*2+out4*4+out16*2)/24-1
            elif class_num==1:
                out=4.0*(out15*1+out0*2+out14*13+out6*2+out4*4)/20-1
            elif class_num==2:
                out=2.0*(out15*2+out0*2+out14*10+out6*2+out4*4+out16)/21-2
            elif class_num==3:
                out=1.2*(out15*2+out0*2+out14*9+out6*2+out4*4+out16*2)/20-1
            elif class_num==4:
                out=.4*(out15*1+out0*2+out14*9+out6*2+out4*4)/20-.5

            if np.mean(out)<0:
                out = out-np.mean(out)+.5
            if brownian:
                out = out - np.mean(out)
            outputs.append(out)

    newout = np.vstack(outputs)
    outputs = np.mean(newout,0)
    out3 = outputs
    

    vx=vx[~np.isnan(vx)]
    vx = vx[vx!=0]
    vx = vx[np.abs(vx)<=300]
    
    # vels=1
    # for i in range(len(tracks)):
    #     idx = tracks[i]
    #     if df[df.TRACK_ID==idx].sort_values(by='FRAME').FRAME.iloc[-1] < (images.shape[0]*images.shape[1]):
    #         posy = df[df.TRACK_ID==idx].sort_values(by='FRAME').POSITION_X
    #         yvel = calculate_velocity(np.array(posy))
        
    #         vels = np.hstack([vels, yvel])
    #         time.sleep(0.002)
            
    # vels=vels[~np.isnan(vels)]
    # vels = vels[vels!=0]
    # vels = vels[np.abs(vels)<=300]
    out3 = out3[out3!=0]
    out3 = out3[np.abs(out3)<=300]
    mean_loss =  np.abs(np.mean(vx)-np.mean(out3))
    # mean_loss_pt =  np.abs(np.mean(vels)-np.mean(vx))
    #print('Min, Max: ', np.min(vels), np.max(vels))

    
    print('Absolute Error of Average Vx (Model, PT): ',mean_loss)
    
    q=np.sort(interpolate_vectors(np.sort(out3), 500))
    # e=np.sort(interpolate_vectors(np.sort(vels), 500))
    w=np.sort(interpolate_vectors(np.sort(vx), 500))
    speed_loss = RMSE(q, w)
    # speed_loss2 = RMSE(e, w)
    print('Vx Loss (model, PT): ', speed_loss)
    
    file_error_vx.append(mean_loss)
    # file_error_pt_vx.append(mean_loss_pt)
    file_mse_vx.append(speed_loss)
    # file_mse_pt_vx.append(speed_loss2)
    final_pt_vx.append(vx)
    
    # plt.figure(figsize=(8, 6))
    a,b = norm.fit(q)

    xtorch1= np.linspace(norm.ppf(0.0001, a,b),
                    norm.ppf(0.9999, a,b), target_length)

    distr_torch1=norm.pdf(xtorch1, a,b)
    # plt.plot(xtorch1,distr_torch1,c='r')

    # a,b = norm.fit(e)
    # xtorch2= np.linspace(norm.ppf(0.0001, a,b),
    #                 norm.ppf(0.9999, a,b), target_length)
    # distr_torch2=norm.pdf(xtorch2, a,b)
    # plt.plot(xtorch2,distr_torch2,c='b')
    
    a,b = norm.fit(w)
        
    xtorch= np.linspace(norm.ppf(0.0001, a,b),
                    norm.ppf(0.9999, a,b), target_length)

    distr_torch=norm.pdf(xtorch, a,b)
    # plt.plot(xtorch,distr_torch,c='g')
    # plt.ylabel('Probability Density')
    # plt.xlabel('Speed (pixels/frame)')
    # plt.title('Vx Distribution for '+str(base), fontsize=12)
    # plt.legend(['Predictions', 'Particle Tracking (TrackMate)', 'Ground Truth'])
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig('Results/Vx_'+str(f_idx)+'.png')
    w1 = wasserstein_distance(distr_torch1*xtorch1, distr_torch*xtorch)
    # w2 = wasserstein_distance(distr_torch2, distr_torch)
    print ('Earth Movers Distance (model, PT): ', w1)
    
    vx_w1.append(w1)
    # vx_w2.append(w2)
    
    final_out_vx.append(np.sort(q))
    final_true_vx.append(w)
    
    # ##################
    # ######  Vy  ######
    # ##################
    
    print('Starting Vy')


    outputs=[]
    outputs2=[]
    torch.cuda.empty_cache()
    gc.collect()

    max1 = 69.2162
    min1 = -66.9786
    slope1 = max1-min1
    int1 = min1

    max2 = 64.4513
    min2 = -66.1406
    slope2 = max2-min2
    int2 = min2
    
    max3 =132.73
    min3 = -95.1744
    slope3 = max3-min3
    int3 = min3


    outputs=[]
    outputs2=[]
    torch.cuda.empty_cache()
    gc.collect()


    max1 = 69.2162
    min1 = -66.9786
    slope1 = max1-min1
    int1 = min1


    with torch.no_grad():
        for x in test_dataloader:
            if torch.mean(x)<.5:
                x = 1-x
            # x1 = torchvision.transforms.functional.resize(x, 448)
            
            x2 = torchvision.transforms.functional.resize(x, 384)
            x3 = torchvision.transforms.functional.resize(x, 224)
            out0 = np.sort(model0_vy((x2).to(device)).detach().cpu().numpy())*slope1+int1
            out1 = np.sort(model1_vy((x2).to(device)).detach().cpu().numpy())*slope1+int1
            out2 = np.sort(model2_vy((x2).to(device)).detach().cpu().numpy())*slope1+int1
            out3 = np.sort(model3_vy((x3).to(device)).detach().cpu().numpy())*slope1+int1
            out4 = np.sort(model4_vy((x).to(device)).detach().cpu().numpy())*slope2+int2
            out4_1 = out4*1.75+2.5
            out4_2 = out4*1.55-3.5
            out4_3 = out4*.7
            out5 = np.sort(model5_vy((x2).to(device)).detach().cpu().numpy())*slope2+int2
            out6 = np.sort(model6_vy((x[:,5:35,:,:]).to(device)).detach().cpu().numpy())*slope3+int3
            out5_2 = np.sort(model5_vy((x2).to(device)).detach().cpu().numpy())*slope1+int1
            out7 = (out0+out4*2+out6)/4

            
            ##class1
            if class_num==0:
                out=3.8*(out0+out1+out2+out3+out4*35+out5+out5_2+out4_2*20+out4_3*2+out7*10)/71+4
            elif class_num==1:
                out=3.0*(out0+out1+out2+out3+out4*30+out5+out5_2+out4_2*20+out4_3*2+out7*10)/66+2.6
            
            elif class_num==2:
                out=0.9*(out0+out1+out2+out3+out4*30+out5+out5_2+out4_2*22+out7*10)/66+1.6

            elif class_num==3:
                out=.7*(out0+out1+out2+out3+out4*30+out5+out5_2+out4_2*24+out4_1*6+out7*10)/72+1

            elif class_num==4:
                out=.25*(out4*25+out4_2*1+out4_3*17+out7*10)/53-.1

            outputs.append(out)

    max2 =132.73
    min2 = -95.1744
    slope2 = max2-min2
    int2 = min2
    outputs2=[]

    newout = np.vstack(outputs)
    outputs = np.mean(newout,0)


    out3=outputs#1.1*outputs-1#1.1*((outputs*15+outputs2*1)/16)-1
    

    # vels=1
    # for i in range(len(tracks)):
    #     idx = tracks[i]
    #     if df[df.TRACK_ID==idx].sort_values(by='FRAME').FRAME.iloc[-1] < (images.shape[0]*images.shape[1]):
    #         posy = df[df.TRACK_ID==idx].sort_values(by='FRAME').POSITION_Y
    #         yvel = calculate_velocity(np.array(posy))
        
    #         vels = np.hstack([vels, yvel])
    #         time.sleep(0.002)
    # vels=vels[~np.isnan(vels)]
    vy = vy[vy!=0]
    vy = vy[np.abs(vy)<=300]
    # vels = vels[vels!=0]
    # vels = vels[np.abs(vels)<=300]
    out3 = out3[out3!=0]
    out3 = out3[np.abs(out3)<=300]
    #print('Min out, Max out: ', np.min(out3), np.max(out3))

    mean_loss =  np.abs(np.mean(vy)-np.mean(out3))
    # mean_loss_pt =  np.abs(np.mean(vy)-np.mean(vels))
    print('Mean Vy from PT: ', np.mean(vy))
    
    # print('Absolute Error of Average Vy (model, PT): ',mean_loss)
    q=np.sort(interpolate_vectors(np.sort(out3), 500))
    # e=np.sort(interpolate_vectors(np.sort(vels), 500))
    w=np.sort(interpolate_vectors(np.sort(vy), 500))
    speed_loss = RMSE(q, w)
    # speed_loss2 = RMSE(e, w)
    print('Vy Loss (model, PT): ', speed_loss)
    
    file_error_vy.append(mean_loss)
    # file_error_pt_vy.append(mean_loss_pt)
    file_mse_vy.append(speed_loss)
    # file_mse_pt_vy.append(speed_loss2)

    #print('Min, Max: ', np.min(e), np.max(e))    
    # plt.figure(figsize=(8, 6))
    a,b = norm.fit(q)

    xtorch1= np.linspace(norm.ppf(0.0001, a,b),
                    norm.ppf(0.9999, a,b), target_length)

    distr_torch1=norm.pdf(xtorch1, a,b)
    # plt.plot(xtorch1,distr_torch1,c='r')
    
    # a,b = norm.fit(e)

    # xtorch2= np.linspace(norm.ppf(0.0001, a,b),
    #                 norm.ppf(0.9999, a,b), target_length)

    # distr_torch2=norm.pdf(xtorch2, a,b)
    # plt.plot(xtorch2,distr_torch2,c='b')


    a,b = norm.fit(w)
        
    xtorch= np.linspace(norm.ppf(0.0001, a,b),
                    norm.ppf(0.9999, a,b), target_length)

    distr_torch=norm.pdf(xtorch, a,b)
    # plt.plot(xtorch,distr_torch,c='g')

    # plt.ylabel('Probability Density')
    # plt.xlabel('Speed (pixels/frame)')
    # plt.title('Vy Distribution for '+str(base)+', class number = '+str(class_num), fontsize=12)
    # plt.legend(['Predictions', 'Particle Tracking (TrackMate)', 'Ground Truth'])
    # plt.grid(True)
    # plt.tight_layout()
    

    # plt.savefig('Results/Vy_'+str(f_idx)+'.png')
    w1 = wasserstein_distance(distr_torch, distr_torch1)
    # w2 = wasserstein_distance(distr_torch, distr_torch2)
    print ('Earth Movers Distance (model, PT): ', w1)
    # plt.show()
    
    final_out_vy.append(q)
    # final_out_pt_vy.append(e)
    final_true_vy.append(w)
    vy_w1.append(w1)
    # vy_w2.append(w2)
    
   
    
print('Average Model Scores Over All Testing Files: ')  
##Speeds  
print('Average Model Speed Error for all Files: ',np.nanmean(file_error))
print('Average Model Speed STD for all Files: ',np.nanstd(file_error))
print('10th/90th percentiles Speed Error: ', np.nanpercentile(file_error, [10, 90]))

print('Average Model MSE (speed) Error for all Files: ',np.nanmean(file_mse))
print('10th/90th percentiles MSE Speed: ', np.nanpercentile(file_mse, [10, 90]))

print('Average Model W1 (speed) Error for all Files: ',np.nanmean(speed_w1))
print('10th/90th percentiles W1 Speed: ', np.nanpercentile(speed_w1, [10, 90]))

##Angles
print('Average Model Angle Error for all Files: ',np.nanmean(file_error_ang))
print('Average Model Angle STD for all Files: ',np.nanstd(file_error_ang))
print('10th/90th percentiles Angle Error: ', np.nanpercentile(file_error_ang, [10, 90]))

print('Average Model MSE (angle) Error for all Files: ',np.nanmean(file_mse_ang))
print('10th/90th percentiles MSE Angle: ', np.nanpercentile(file_mse_ang, [10, 90]))

print('Average Model W1 (angle) Error for all Files: ',np.nanmean(ang_w1))
print('10th/90th percentiles W1 Angle: ', np.nanpercentile(ang_w1, [10, 90]))

#Vx
print('Average Model Vx Error for all Files: ',np.nanmean(file_error_vx))
print('Average Model Vx STD for all Files: ',np.nanstd(file_error_vx))
print('10th/90th percentiles Vx Error: ', np.nanpercentile(file_error_vx, [10, 90]))

print('Average Model MSE (Vx) Error for all Files: ',np.nanmean(file_mse_vx))
print('10th/90th percentiles MSE Vx: ', np.nanpercentile(file_mse_vx, [10, 90]))

print('Average W1 Model (Vx) Error for all Files: ',np.nanmean(vx_w1))
print('10th/90th percentiles W1 Vx: ', np.nanpercentile(vx_w1, [10, 90]))

#Vy
print('Average Model Vy Error for all Files: ',np.nanmean(file_error_vy))
print('Average Model Vy STD for all Files: ',np.nanstd(file_error_vy))
print('10th/90th percentiles Vy Error: ', np.nanpercentile(file_error_vy, [10, 90]))

print('Average Model MSE (Vy) Error for all Files: ',np.nanmean(file_mse_vy))
print('10th/90th percentiles MSE Vy: ', np.nanpercentile(file_mse_vy, [10, 90]))

print('Average W1 Model (Vy) Error for all Files: ',np.nanmean(vy_w1))
print('10th/90th percentiles W1 Vy: ', np.nanpercentile(vy_w1, [10, 90]))

print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

print('ERROR FOR HIGH SPEED SUBSET')
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')


k=28
j=-1
idx=[1,5,9,19,20,21,22,23,24,25,31,32]
file_error2 = [file_error[i] for i in idx]
# file_error_pt2 = [file_error_pt[i] for i in idx]
file_mse2 = [file_mse[i] for i in idx]
# file_mse_pt2 = [file_mse_pt[i] for i in idx]
speed_w12 = [speed_w1[i] for i in idx]
# speed_w22 = [speed_w2[i] for i in idx]
file_error_ang2 = [file_error_ang[i] for i in idx]
# file_error_pt_ang2 = [file_error_pt_ang[i] for i in idx]
file_mse_ang2 = [file_mse_ang[i] for i in idx]
# file_mse_pt_ang2 = [file_mse_pt_ang[i] for i in idx]
ang_w12 = [ang_w1[i] for i in idx]
# ang_w22 = [ang_w2[i] for i in idx]
file_error_vx2 = [file_error_vx[i] for i in idx]
# file_error_pt_vx2 = [file_error_pt_vx[i] for i in idx]
file_mse_vx2 = [file_mse_vx[i] for i in idx]
# file_mse_pt_vx2 = [file_mse_pt_vx[i] for i in idx]
vx_w12 = [vx_w1[i] for i in idx]
# vx_w22 = [vx_w2[i] for i in idx]
file_error_vy2 = [file_error_vy[i] for i in idx]
# file_error_pt_vy2 = [file_error_pt_vy[i] for i in idx]
file_mse_vy2 = [file_mse_vy[i] for i in idx]
# file_mse_pt_vy2 = [file_mse_pt_vy[i] for i in idx]
vy_w12 = [vy_w1[i] for i in idx]
# vy_w22 = [vy_w2[i] for i in idx]

print('Average Model Speed Error for all Files: ',np.nanmean(file_error2))
print('Average Model Speed STD for all Files: ',np.nanstd(file_error2))
print('10th/90th percentiles Speed Error: ', np.nanpercentile(file_error2, [10, 90]))
# print('Average PT Speed Error for all Files: ',np.nanmean(file_error_pt2))
# print('Average PT Speed STD for all Files: ',np.nanstd(file_error_pt2))

print('Average Model MSE (speed) Error for all Files: ',np.nanmean(file_mse2))
print('Average Model MSE (speed) STD for all Files: ',np.nanstd(file_mse2))
print('10th/90th percentiles MSE Speed: ', np.nanpercentile(file_mse2, [10, 90]))

# print('Average PT MSE (speed) for all Files: ',np.nanmean(file_mse_pt2))
# print('Average PT MSE (speed) STD for all Files: ',np.nanstd(file_mse_pt2))

print('Average Model W1 (speed) Error for all Files: ',np.nanmean(speed_w12))
print('Average Model W1 (speed) STD for all Files: ',np.nanstd(speed_w12))
print('10th/90th percentiles W1 Speed: ', np.nanpercentile(speed_w12, [10, 90]))
# print('Average PT W1 (speed) Error for all Files: ',np.nanmean(speed_w22))
# print('Average PT W1 (speed) STD for all Files: ',np.nanstd(speed_w22))


##Angles
print('Average Model Angle Error for all Files: ',np.nanmean(file_error_ang2))
print('Average Model Angle STD for all Files: ',np.nanstd(file_error_ang2))
print('10th/90th percentiles Angle Error: ', np.nanpercentile(file_error_ang2, [10, 90]))

# print('Average PT Angle Error for all Files: ',np.nanmean(file_error_pt_ang2))
# print('Average PT Angle STD for all Files: ',np.nanstd(file_error_pt_ang2))

print('Average Model MSE (angle) Error for all Files: ',np.nanmean(file_mse_ang2))
print('Average Model MSE (angle) STD for all Files: ',np.nanstd(file_mse_ang2))
print('10th/90th percentiles MSE Angle: ', np.nanpercentile(file_mse_ang2, [10, 90]))
# print('Average PT MSE (angle) Error for all Files: ',np.nanmean(file_mse_pt_ang2))
# print('Average PT MSE (angle) STD for all Files: ',np.nanstd(file_mse_pt_ang2))

print('Average Model W1 (angle) Error for all Files: ',np.nanmean(ang_w12))
print('Average Model W1 (angle) STD for all Files: ',np.nanstd(ang_w12))
print('10th/90th percentiles W1 Angle: ', np.nanpercentile(ang_w12, [10, 90]))
# print('Average PT W1 (angle) Error for all Files: ',np.nanmean(ang_w22))
# print('Average PT W1 (angle) STD for all Files: ',np.nanstd(ang_w22))

##Vx
print('Average Model Vx Error for all Files: ',np.nanmean(file_error_vx2))
print('Average Model Vx STD for all Files: ',np.nanstd(file_error_vx2))
print('10th/90th percentiles Vx Error: ', np.nanpercentile(file_error_vx2, [10, 90]))   
# print('Average PT Vx Error for all Files: ',np.nanmean(file_error_pt_vx2))
# print('Average PT Vx STD for all Files: ',np.nanstd(file_error_pt_vx2))

print('Average Model MSE (Vx) Error for all Files: ',np.nanmean(file_mse_vx2))
print('Average Model MSE (Vx) STD for all Files: ',np.nanstd(file_mse_vx2))
print('10th/90th percentiles MSE Vx: ', np.nanpercentile(file_mse_vx2, [10, 90]))
# print('Average PT MSE (Vx) Error for all Files: ',np.nanmean(file_mse_pt_vx2))
# print('Average PT MSE (Vx) STD for all Files: ',np.nanstd(file_mse_pt_vx2))

print('Average W1 Model (Vx) Error for all Files: ',np.nanmean(vx_w12))
print('Average W1 Model (Vx) STD for all Files: ',np.nanstd(vx_w12))
print('10th/90th percentiles W1 Vx: ', np.nanpercentile(vx_w12, [10, 90]))
# print('Average W1 PT (Vx) Error for all Files: ',np.nanmean(vx_w22))
# print('Average W1 PT (Vx) STD for all Files: ',np.nanstd(vx_w22))


###Vy
print('Average Model Vy Error for all Files: ',np.nanmean(file_error_vy2))
print('Average Model Vy STD for all Files: ',np.nanstd(file_error_vy2))
print('10th/90th percentiles Vy Error: ', np.nanpercentile(file_error_vy2, [10, 90]))
# print('Average PT Vy Error for all Files: ',np.nanmean(file_error_pt_vy2))
# print('Average PT Vy STD for all Files: ',np.nanstd(file_error_pt_vy2))

print('Average Model MSE (Vy) Error for all Files: ',np.nanmean(file_mse_vy2))
print('Average Model MSE (Vy) STD for all Files: ',np.nanstd(file_mse_vy2))
print('10th/90th percentiles MSE Vy: ', np.nanpercentile(file_mse_vy2, [10, 90]))
# print('Average PT MSE (Vy) Error for all Files: ',np.nanmean(file_mse_pt_vy2))
# print('Average PT MSE (Vy) STD for all Files: ',np.nanstd(file_mse_pt_vy2))

print('Average W1 Model (Vy) Error for all Files: ',np.nanmean(vy_w12))
print('Average W1 Model (Vy) STD for all Files: ',np.nanstd(vy_w12))
print('10th/90th percentiles W1 Vy: ', np.nanpercentile(vy_w12, [10, 90]))
# print('Average W1 PT (Vy) Error for all Files: ',np.nanmean(vy_w22))
# print('Average W1 PT (Vy) MSD for all Files: ',np.nanstd(vy_w22))

# print(len(final_out))
# print(final_out[0].shape)
# q = np.sort(np.hstack(final_out))
# e = np.sort(np.hstack(final_pt))
# w = np.sort(np.hstack(final_true))
# print(q.shape)
# plt.plot(q, c='b', linewidth=2)
# plt.plot(w, c='g', linewidth=2)
# plt.xlabel('Vector Index')
# plt.ylabel('Speed (pixels/frame)')
# plt.title('Total Speed Vectors for All Test Data', fontsize=12)
# plt.legend(['Predictions', 'Particle Tracking (TrackMate)'])
# plt.grid(True)
# plt.tight_layout()
# plt.savefig('Results/Speed_final.png')
# plt.show()

# a,b,c,d = betaprime.fit(q)
# xtorch= np.linspace(betaprime.ppf(0.0001, a,b,c,d),
#                 betaprime.ppf(0.9999, a,b,c,d), target_length)
# distr_torch=betaprime.pdf(xtorch, a,b,c,d)
# plt.plot(xtorch,distr_torch,c='r')

# a,b,c,d = betaprime.fit(e)
# xtorch= np.linspace(betaprime.ppf(0.0001, a,b,c,d),
#                 betaprime.ppf(0.9999, a,b,c,d), target_length)
# distr_torch=betaprime.pdf(xtorch, a,b,c,d)
# plt.plot(xtorch,distr_torch,c='b')

# a,b,c,d = betaprime.fit(w) 
# xtorch= np.linspace(betaprime.ppf(0.0001, a,b,c,d),
#                 betaprime.ppf(0.9999, a,b,c,d), target_length)
# distr_torch=betaprime.pdf(xtorch, a,b,c,d)
# plt.plot(xtorch,distr_torch,c='g')
# plt.title('Total Speed Distributions for All Test Data')
# plt.xlabel('Speed (pixels/frame)')
# plt.ylabel('Probability Density')
# plt.legend(['Prediction',  'Particle Tracking (TrackMate)','Ground Truth'])
# plt.xscale('log')
# plt.yscale('log')
# plt.savefig('Results/Speed_distribution_final.png')
# plt.show()


# ###Angles
# q = np.sort(np.hstack(final_out_ang))
# print(q.shape)
# e = np.sort(np.hstack(final_pt_ang))
# w = np.sort(np.hstack(final_true_ang))
# plt.plot(q, c='r', linewidth=2)
# plt.plot(e, c='b', linewidth=2)
# plt.plot(w, c='g', linewidth=2)
# plt.xlabel('Vector Index')
# plt.ylabel('Turn Angle (degrees)')
# plt.title('Total Turn Angle Vectors for All Test Data', fontsize=12)
# plt.legend(['Predictions', 'Particle Tracking (TrackMate)', 'Ground Truth'])
# plt.grid(True)
# plt.tight_layout()
# plt.savefig('Results/angles_final.png')
# plt.show()

# a,b,c,d = betaprime.fit(q)
# xtorch= np.linspace(betaprime.ppf(0.0001, a,b,c,d),
#                 betaprime.ppf(0.9999, a,b,c,d), target_length)
# distr_torch=betaprime.pdf(xtorch, a,b,c,d)
# plt.plot(xtorch,distr_torch,c='r')
# a,b,c,d = betaprime.fit(e)
# xtorch= np.linspace(betaprime.ppf(0.0001, a,b,c,d),
#                 betaprime.ppf(0.9999, a,b,c,d), target_length)
# distr_torch=betaprime.pdf(xtorch, a,b,c,d)
# plt.plot(xtorch,distr_torch,c='b')
# a,b,c,d = betaprime.fit(w) 
# xtorch= np.linspace(betaprime.ppf(0.0001, a,b,c,d),
#                 betaprime.ppf(0.9999, a,b,c,d), target_length)
# distr_torch=betaprime.pdf(xtorch, a,b,c,d)
# plt.plot(xtorch,distr_torch,c='g')
# plt.title('Total Turn Angle Distributions for All Test Data')
# plt.xlabel('Turn Angle (degrees)')
# plt.ylabel('Probability Density')
# plt.legend(['Predictions', 'Particle Tracking (TrackMate)', 'Ground Truth'])
# plt.xscale('log')
# plt.yscale('log')
# plt.savefig('Results/angle_distribution_final.png')
# plt.show()

# ###Vx
# q = np.sort(np.hstack(final_out_vx))
# print(q.shape)
# e = np.sort(np.hstack(final_pt_vx))

# w = np.sort(np.hstack(final_true_vx))
# plt.plot(q, c='b', linewidth=2)
# plt.plot(e, c='b', linewidth=2)

# plt.plot(w, c='g', linewidth=2)
# plt.xlabel('Vector Index')
# plt.ylabel('Vx (pixels/frame)')
# plt.title('Total Vx Vectors for All Test Data', fontsize=12)
# plt.legend(['Predictions', 'Particle Tracking (TrackMate)', 'Ground Truth'])
# plt.grid(True)
# plt.tight_layout()
# plt.savefig('Results/Vx_final.png')
# plt.show()

# a,b = norm.fit(q)
# xtorch= np.linspace(norm.ppf(0.0001, a,b),
#                 norm.ppf(0.9999, a,b), target_length)
# distr_torch=norm.pdf(xtorch, a,b)
# plt.plot(xtorch,distr_torch,c='r')
# a,b = norm.fit(e)
# xtorch= np.linspace(norm.ppf(0.0001, a,b),
#                 norm.ppf(0.9999, a,b), target_length)
# distr_torch=norm.pdf(xtorch, a,b)
# plt.plot(xtorch,distr_torch,c='b')
# a,b = norm.fit(w) 
# xtorch= np.linspace(norm.ppf(0.0001, a,b),
#                 norm.ppf(0.9999, a,b), target_length)
# distr_torch=norm.pdf(xtorch, a,b)
# plt.plot(xtorch,distr_torch,c='g')
# plt.title('Total Vx Distributions for All Test Data')
# plt.xlabel('Vx (pixels/frame)')
# plt.ylabel('Probability Density')
# plt.legend(['Predictions', 'Particle Tracking (TrackMate)', 'Ground Truth'])

# plt.savefig('Results/Vx_distribution_final.png')
# plt.show()

# ###Vy
# q = np.sort(np.hstack(final_out_vy))
# e = np.sort(np.hstack(final_out_pt_vy))
# print(q.shape)
# w = np.sort(np.hstack(final_true_vy))
# plt.plot(q, c='r', linewidth=2)
# plt.plot(e, c='b', linewidth=2)

# plt.plot(w, c='g', linewidth=2)
# plt.xlabel('Vector Index')
# plt.ylabel('Vy (pixels/frame)')
# plt.title('Total Vy Vectors for All Test Data', fontsize=12)
# plt.legend(['Predictions', 'Particle Tracking (TrackMate)', 'Ground Truth'])
# plt.grid(True)
# plt.tight_layout()
# plt.savefig('Results/Vy_final.png')
# plt.show()

# a,b = norm.fit(q)
# xtorch= np.linspace(norm.ppf(0.0001, a,b),
#                 norm.ppf(0.9999, a,b), target_length)
# distr_torch=norm.pdf(xtorch, a,b)
# plt.plot(xtorch,distr_torch,c='r')
# a,b = norm.fit(e)
# xtorch= np.linspace(norm.ppf(0.0001, a,b),
#                 norm.ppf(0.9999, a,b), target_length)
# distr_torch=norm.pdf(xtorch, a,b)
# plt.plot(xtorch,distr_torch,c='b')
# a,b = norm.fit(w) 
# xtorch= np.linspace(norm.ppf(0.0001, a,b),
#                 norm.ppf(0.9999, a,b), target_length)
# distr_torch=norm.pdf(xtorch, a,b)
# plt.plot(xtorch,distr_torch,c='g')
# plt.title('Total Vy Distributions for All Test Data')
# plt.xlabel('Vy (pixels/frame)')
# plt.ylabel('Probability Density')
# plt.legend(['Predictions', 'Particle Tracking (TrackMate)', 'Ground Truth'])

# plt.savefig('Results/Vy_distribution_final.png')
# plt.show()
