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
from scipy.stats import exponweib,lognorm, beta, betaprime, norm, expon, wasserstein_distance

#class_num=4
target_length=500
device='cuda'

def classifier(dataloader, constants, weights):
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
            
    # model3 = timm.create_model('volo_d3_448', in_chans=30,drop_path_rate=.0, num_classes=4,pretrained=False)
    # model3.load_state_dict(torch.load('models/speed_classifier_volod1_448px_disp_all_5_04'))
    # model3.to(device).eval()

    # model0 = timm.create_model('volo_d1_384', in_chans=30,drop_path_rate=.0, num_classes=4,pretrained=False)
    # model0.load_state_dict(torch.load('models/speed_classifier_volod1_384px_disp_all_5_04_v3'))
    # model0.to(device).eval()

    # model1 = timm.create_model('volo_d1_384', in_chans=30,drop_path_rate=.0, num_classes=4,pretrained=False)
    # model1.load_state_dict(torch.load('models/speed_classifier_volod1_384px_disp_all_5_04_v2'))
    # model1.to(device).eval()

    # model2 = timm.create_model('volo_d1_384', in_chans=30,drop_path_rate=.0, num_classes=4,pretrained=False)
    # model2.load_state_dict(torch.load('models/speed_classifier_volod1_384px_disp_all_5_04_v1'))
    # model2.to(device).eval()

    model4 = timm.create_model('volo_d4_448', in_chans=30,drop_path_rate=.0, num_classes=5,pretrained=False)
    model4.load_state_dict(torch.load('models/speed_classifier_5class_volod4_448px_disp_all_5_04'))
    model4.to(device).eval()

    model5 = timm.create_model('volo_d3_448', in_chans=30,drop_path_rate=.0, num_classes=5,pretrained=False)
    model5.load_state_dict(torch.load('models/speed_classifier_5class_volod3_448px_disp_all_5_05'))
    model5.to(device).eval()

    model6 = timm.create_model('volo_d3_448', in_chans=30,drop_path_rate=.0, num_classes=5,pretrained=False)
    model6.load_state_dict(torch.load('models/speed_classifier_5class_volod3_448px_disp_lbm_5_07'))
    model6.to(device).eval()

    model7 = timm.create_model('volo_d3_448', in_chans=30,drop_path_rate=.0, num_classes=5,pretrained=False)
    model7.load_state_dict(torch.load('models/speed_classifier_5class_volod3_448px_disp_homo_5_07'))
    model7.to(device).eval()

    #model8 = timm.create_model('volo_d3_448', in_chans=30,drop_path_rate=.0, num_classes=5,pretrained=False)
    #model8.load_state_dict(torch.load('models/speed_classifier_5class_volod3_448px_disp_lbm_5_07_v2'))
    #model8.to(device).eval()

    #model9 = timm.create_model('volo_d1_384', in_chans=30,drop_path_rate=.0, num_classes=5,pretrained=False)
    #model9.load_state_dict(torch.load('models/speed_classifier_5class_volod1_384px_disp_all_5_06'))
    #model9.to(device).eval()

    model10 = timm.create_model('volo_d1_384', in_chans=30,drop_path_rate=.0, num_classes=5,pretrained=False)
    model10.load_state_dict(torch.load('models/speed_classifier_5class_volod1_384px_disp_all_5_06_v2'))
    model10.to(device).eval()

    model11 = timm.create_model('volo_d1_384', in_chans=30,drop_path_rate=.0, num_classes=5,pretrained=False)
    model11.load_state_dict(torch.load('models/speed_classifier_5class_volod1_384px_disp_all_5_06_v3'))
    model11.to(device).eval()

    #model12 = timm.create_model('volo_d1_384', in_chans=30,drop_path_rate=.0, num_classes=5,pretrained=False)
    #model12.load_state_dict(torch.load('models/speed_classifier_5class_volod1_384px_disp_all_5_06_v4'))
    #model12.to(device).eval()

    model13 = timm.create_model('volo_d1_384', in_chans=30,drop_path_rate=.0, num_classes=5,pretrained=False)
    model13.load_state_dict(torch.load('models/speed_classifier_5class_volod1_384px_disp_all_5_06_v5'))
    model13.to(device).eval()

    model14 = timm.create_model('volo_d1_384', in_chans=30,drop_path_rate=.0, num_classes=5,pretrained=False)
    model14.load_state_dict(torch.load('models/speed_classifier_5class_volod1_384px_disp_all_5_06_v6'))
    model14.to(device).eval()
    
    model15 = timm.create_model('efficientvit_b1', in_chans=30, num_classes=5,pretrained=False)
    model15.load_state_dict(torch.load('models/speed_classifier_efficientvit_b1_mse_448px_all_5_08'))
    model15.to(device).eval()
    
    model16 = timm.create_model('pvt_v2_b1', in_chans=30,num_classes=5,pretrained=False)
    model16.load_state_dict(torch.load('models/speed_classifier_pvt_v2_b1_mse_448px_all_5_08'))
    model16.to(device).eval()
    
    model17 = timm.create_model('regnetx_032', in_chans=30,num_classes=5,pretrained=False)
    model17.load_state_dict(torch.load('models/speed_classifier_regnetx_032_mse_448px_all_5_08'))
    model17.to(device).eval()
    
    model18 = timm.create_model('twins_svt_small', in_chans=30, num_classes=5,pretrained=False)
    model18.load_state_dict(torch.load('models/speed_classifier_twins_svt_small_mse_448px_all_5_09'))
    model18.to(device).eval()
    
    model19 = timm.create_model('davit_base', in_chans=30, num_classes=5,pretrained=False)
    model19.load_state_dict(torch.load('models/speed_classifier_davit_base_small_mse_448px_all_5_09'))
    model19.to(device).eval()
    
    model20 = timm.create_model('davit_base', in_chans=30, num_classes=5,pretrained=False)
    model20.load_state_dict(torch.load('models/speed_classifier_davit_base_small_mse_448px_all_5_09_v2'))
    model20.to(device).eval()
    
    model21 = timm.create_model('regnetx_064', in_chans=30, num_classes=3,pretrained=False)
    model21.load_state_dict(torch.load('models/speed_classifier_regnetx064_mse_448px_all_5_12'))
    model21.to(device).eval()
    
    model22 = timm.create_model('regnetx_064', in_chans=30, num_classes=3,pretrained=False)
    model22.load_state_dict(torch.load('models/speed_classifier_regnetx064_mse_448px_all_5_12_v2'))
    model22.to(device).eval()
    
    model23 = timm.create_model('regnetx_064', in_chans=30, num_classes=3,pretrained=False)
    model23.load_state_dict(torch.load('models/speed_classifier_regnetx064_mse_448px_all_5_12_v3'))
    model23.to(device).eval()
    
    model24 = timm.create_model('regnetx_160', in_chans=30, num_classes=3,pretrained=False)
    model24.load_state_dict(torch.load('models/speed_classifier_regnetx160_mse_448px_all_5_12'))
    model24.to(device).eval()
    
    model25 = timm.create_model('pvt_v2_b0', in_chans=30, num_classes=3,pretrained=False)
    model25.load_state_dict(torch.load('models/speed_classifier_pvt_v2_b0_mse_384px_all_5_12'))
    model25.to(device).eval()
    
    model26 = timm.create_model('pvt_v2_b1', in_chans=30, num_classes=3,pretrained=False)
    model26.load_state_dict(torch.load('models/speed_classifier_pvt_v2_b1_mse_384px_all_5_12'))
    model26.to(device).eval()
    
    model27 = timm.create_model('pvt_v2_b2', in_chans=30, num_classes=3,pretrained=False)
    model27.load_state_dict(torch.load('models/speed_classifier_pvt_v2_b2_mse_384px_all_5_12'))
    model27.to(device).eval()
    
    
    #outputs3=[]
    
    outputs1=[]
    count=0
    with torch.no_grad():
        for x in test_dataloader2:
            if torch.mean(x)>.5:
                x = 1-x
            #x=1-x
            #x1 = tv.transforms.functional.resize(x, 448)
            x2 = tv.transforms.functional.resize(x, 384)
            #x3 = tv.transforms.functional.resize(x, 224)
            # out0 =F.softmax((model0((x2).to(device)).detach().cpu()),1).numpy()
            # out1 =F.softmax((model1((x2).to(device)).detach().cpu()),1).numpy()
            # out2 =F.softmax((model2((x2).to(device)).detach().cpu()),1).numpy()
            # out3 =F.softmax((model3((x).to(device)).detach().cpu()),1).numpy()


            #out4 =F.softmax((model4((x).to(device)).detach().cpu()),1).numpy()
            out5 =F.softmax((model5((x).to(device)).detach().cpu()),1).numpy()
            out6 =F.softmax((model6((x).to(device)).detach().cpu()),1).numpy() # class 4, close to 1
            out7 =F.softmax((model7((1-x).to(device)).detach().cpu()),1).numpy()
            #out8 =F.softmax((model8((x).to(device)).detach().cpu()),1).numpy()
            #out9 =F.softmax((model9((x2).to(device)).detach().cpu()),1).numpy()
            out10 =F.softmax((model10((x2).to(device)).detach().cpu()),1).numpy()
            out11 =F.softmax((model11((x2).to(device)).detach().cpu()),1).numpy()
            #out12 =F.softmax((model12((x2).to(device)).detach().cpu()),1).numpy()
            out13 =F.softmax((model13((x2).to(device)).detach().cpu()),1).numpy()
            out13[:,3] = out13[:,3]*.1
            out14 =F.softmax((model15((x2).to(device)).detach().cpu()),1).numpy() #class 2 close to cslaa 3
            #out15 =F.softmax((model15((x).to(device)).detach().cpu()),1).numpy() # class 4
            out16 =F.softmax((model16((x).to(device)).detach().cpu()),1).numpy() #class 2 .25
            out17 =F.softmax((model17((x).to(device)).detach().cpu()),1).numpy() # class 2
            out18 =F.softmax((model18((x).to(device)).detach().cpu()),1).numpy() # class 2
            out19 =F.softmax((model19((x).to(device)).detach().cpu()),1).numpy() # class 3
            out20 =F.softmax((model20((x).to(device)).detach().cpu()),1).numpy() # class 2, .22
            out21 =F.softmax((model21((x).to(device)).detach().cpu()),1).numpy()
            out21z = np.zeros([len(out21), 5])
            out21z[:,2:5] = out21
            out22 =F.softmax((model22((x).to(device)).detach().cpu()),1).numpy()
            out22z = np.zeros([len(out22), 5])
            out22z[:,2:5] = out22
            out23 =F.softmax((model23((x).to(device)).detach().cpu()),1).numpy()
            out23z = np.zeros([len(out23), 5])
            out23z[:,2:5] = out23
            out24 =F.softmax((model24((x).to(device)).detach().cpu()),1).numpy()
            out24z = np.zeros([len(out24), 5])
            out24z[:,2:5] = out24
            out25 =F.softmax((model25((x2).to(device)).detach().cpu()),1).numpy()
            out25z = np.zeros([len(out25), 5])
            out25z[:,2:5] = out25
            out26 =F.softmax((model26((x2).to(device)).detach().cpu()),1).numpy()
            out26z = np.zeros([len(out26), 5])
            out26z[:,2:5] = out26
            out27 =F.softmax((model27((x2).to(device)).detach().cpu()),1).numpy()
            out27z = np.zeros([len(out27), 5])
            out27z[:,2:5] = out27
            
            out=(out11*5+out14*.5+out16*28.5+out17*22.6+out19*3+out5*17.5+out10*3.4+out7*.6+out13*32.2+out6*2+out18*1.5)/constants[0]#(out16*4+out17*2+out14*.5)/6.5#(out0*2+out3*40+out5*2+out7*2+out10+out13)/48
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

    print('5: ', np.mean(np.vstack(out5),0))
    print('6: ',np.mean(np.vstack(out6),0))
    print('7: ',np.mean(np.vstack(out7),0))
    #print(np.mean(np.vstack(out8),0))
    #print(np.mean(np.vstack(out9),0))
    print('10: ',np.mean(np.vstack(out10),0))
    print('11: ',np.mean(np.vstack(out11),0))
    #print(np.mean(np.vstack(out12),0))
    print('13: ',np.mean(np.vstack(out13),0))
    print('14: ',np.mean(np.vstack(out14),0))
    #print(np.mean(np.vstack(out15),0))
    print('16: ',np.mean(np.vstack(out16),0))
    print('17: ',np.mean(np.vstack(out17),0))
    print('18: ',np.mean(np.vstack(out18),0))
    print('19: ',np.mean(np.vstack(out19),0))
    
    print('20: ',np.mean(np.vstack(out20),0))
    print('21: ',np.mean(np.vstack(out21),0))
    print('22: ',np.mean(np.vstack(out22),0))
    print('23: ',np.mean(np.vstack(out23),0))
    print('24: ',np.mean(np.vstack(out24),0))
    print('25: ',np.mean(np.vstack(out25),0))
    print('26: ',np.mean(np.vstack(out26),0))
    print('27: ',np.mean(np.vstack(out27),0))
            #print('next batch')
    return outputs1


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
    
    angles=np.abs(angles[~np.isnan(angles)])
    angles = interpolate_vectors(np.sort(angles), target_length)
    angles=angles[angles!=0]
    #angles[angles==0]=0.1
    return angles
    
directory='../experimental_trajectories/'
filenames = os.listdir(directory)
print(filenames)
# f_idx = 9
# print(filenames[f_idx])

directories = [
'../newbgs/Acid4010FR_1ulh_20X_25fps_1x1y_5min_001/',
'../newbgs/Acid4010FR_5ulh_20X_25fps_1x1y_5min_001/',
'../newbgs/Acid4020FR_1ulh_20X_25fps_1x1y_5min_001/',
'../newbgs/Acid4020FR_5ulh_20X_25fps_1x1y_5min_002/','Acid_1ulh_pre/', 
'acid_noflow/open_20x_20fps_001_frames/', 
'../newbgs/Geo8020FR_5ulh_20X_25fps_1x1y_5min_003/', 'Geo_1ulh_pre/',  
 '../new_traj/Acid01xx_20X_25fps_1min_001_frames_bgs/',
'../new_traj/Acid01xx_20X_25fps_1min_002_frames_bgs/','../new_traj/Acid01xx_20X_25fps_1min_003_frames_bgs/',
'../new_traj/Acid01xx_20X_25fps_1min_004_frames_bgs/','../new_traj/Geo01xx_20X_25fps_1min_001_frames_bgs/',
'../new_traj/Geo01xx_20X_25fps_1min_002_frames_bgs/','../new_traj/Geo01xx_20X_25fps_1min_003_frames_bgs/',
'../new_traj/01xx_20X_25fps_5min_003_frames_bgs/','../new_traj/01xx_20X_25fps_5min_003_frames_bgs/',
'../newbgs/Paen8020FR_5ulh_20X_25fps_1x1y_5min_003/',
'paen_noflow/paen_8040_20x_20fps_006_frames/', 'Paen_1ulh_pre/', 
'Paen_5ulh_pre/', 'shew_noflow/8040_20x_002_frames/', 'shew_noflow/808_20x_20fps_005_frames/']

#Speed
file_error=[]
file_mse=[]
final_out=[]
final_true=[]
speed_w1=[]
#Angle
file_error_ang=[]
file_mse_ang=[]
final_out_ang=[]
final_true_ang=[]
ang_w1=[]
#Vx
file_error_vx=[]
file_mse_vx=[]
final_out_vx=[]
final_true_vx=[]
vx_w1=[]
#Vx
file_error_vy=[]
file_mse_vy=[]
final_out_vy=[]
final_true_vy=[]
vy_w1=[]
#directionality
file_error_direction=[]
file_mse_direction=[]
final_out_direction=[]
final_true_direction=[]
direction_r2=[]

final_true_direction=[]
final_out_direction=[]
final_error_direction=[]

for f_idx in range(0,len(directories)):
    print(len(filenames), len(directories))
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
    elif 'test_sims_het' in base:
        sorted_frames = sorted(files, key=lambda x: int(x[4:-4]))
    elif 'frame' in files[0]:
        sorted_frames = sorted(files, key=lambda x: int(x[5:-4]))
    else:
        sorted_frames = sorted(files, key=lambda x: int(x[3:-4]))

    for i in range(len(sorted_frames)):
        file = base+sorted_frames[i]
        full_files.append(file)
        
    gc.collect()

    l1=448
    l2 = 384
    l3 = 224


    images = parallel_image_processing(full_files,l1)
    images2 = parallel_image_processing(full_files,l1)

    #plt.figure(figsize=(10,10),dpi=500)
    #plt.imshow(images1[-1])

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

    x = next(iter(test_dataloader))
    print(x.shape)
    #plt.imshow(x[0,29,:,:])

    
        
    constants = [93,0,0,.169,.126,-.066]
    weights = [['11',5],['14',.5],['16',28.5],['17',22.6],['19',3],['5',17.5],['10',3.4],['7',.6],['13',32.2],['6',2],['18',1.5]]
    outputs1 = classifier(test_dataloader2,constants,weights)
    mean_out = np.mean(np.vstack(outputs1),0)
    class_num = np.argmax(mean_out)
    print('Class Number: ',class_num)
    print(mean_out)
    # #class_num=1


    # class Patch_model2(nn.Module):
        # def __init__(self):
            # super(Patch_model2, self).__init__()
            # self.m0 = timm.create_model('volo_d3_448', in_chans=40, drop_path_rate=0.0,num_classes=2000,pretrained=True)
            # self.m1 = timm.create_model('volo_d1_224', in_chans=40, drop_path_rate=0.0,num_classes=500,pretrained=True)
            # self.m2 = timm.create_model('volo_d1_224', in_chans=40, drop_path_rate=0.0,num_classes=500,pretrained=True)
            # self.m3 = timm.create_model('volo_d1_224', in_chans=40, drop_path_rate=0.0,num_classes=500,pretrained=True)
            # self.m4 = timm.create_model('volo_d1_224', in_chans=40, drop_path_rate=0.0,num_classes=500,pretrained=True)

            # self.fc1=nn.Linear(2000, 1000)
            # self.fc2=nn.Linear(1000,500)


        # def forward(self, x):
            # l2=224
            # x1 = x[:,:,0:l2,0:l2]
            # x2 = x[:,:,0:l2,l2:]
            # x3 = x[:,:,l2:,0:l2]
            # x4 = x[:,:,l2:,l2:]
            # #torch.cuda.empty_cache()
            # #gc.collect()
            # x = F.gelu(self.m0(x))
            # x1 = F.gelu(self.m1(x1))
            # x2 = F.gelu(self.m2(x2))
            # x3 = F.gelu(self.m3(x3))
            # x4 = F.gelu(self.m4(x4))
            # x0 = torch.cat([x1,x2,x3,x4],dim=1)
            # del x1,x2, x3, x4
            # x = x*x0
            # x = F.gelu(self.fc1(x))
            # x = self.fc2(x)
            # return x#,x2,x3,x4

    # class Patch_model3(nn.Module):
        # def __init__(self):
            # super(Patch_model3, self).__init__()
            # self.m1 = timm.create_model('volo_d1_224', in_chans=40, drop_path_rate=0.0,num_classes=500,pretrained=True)
            # self.m2 = timm.create_model('volo_d1_224', in_chans=40, drop_path_rate=0.0,num_classes=500,pretrained=True)
            # self.m3 = timm.create_model('volo_d1_224', in_chans=40, drop_path_rate=0.0,num_classes=500,pretrained=True)
            # self.m4 = timm.create_model('volo_d1_224', in_chans=40, drop_path_rate=0.0,num_classes=500,pretrained=True)

            # self.fc1=nn.Linear(2000, 1000)
            # self.fc2=nn.Linear(1000,500)


        # def forward(self, x):
            # l2=224
            # x1 = x[:,:,0:l2,0:l2]
            # x2 = x[:,:,0:l2,l2:]
            # x3 = x[:,:,l2:,0:l2]
            # x4 = x[:,:,l2:,l2:]
            # #torch.cuda.empty_cache()
            # #gc.collect()
            # x1 = F.gelu(self.m1(x1))
            # x2 = F.gelu(self.m2(x2))
            # x3 = F.gelu(self.m3(x3))
            # x4 = F.gelu(self.m4(x4))
            # x = torch.cat([x1,x2,x3,x4],dim=1)
            # del x1,x2, x3, x4
            # x = F.gelu(self.fc1(x))
            # x = self.fc2(x)
            # return x#,x2,x3,x4

    # class Patch_model4(nn.Module):
        # def __init__(self):
            # super(Patch_model4, self).__init__()
            # self.m1 = timm.create_model('volo_d1_224', in_chans=40, drop_path_rate=0.0,num_classes=500,pretrained=True)
            # self.m2 = timm.create_model('volo_d1_224', in_chans=40, drop_path_rate=0.0,num_classes=500,pretrained=True)
            # self.m3 = timm.create_model('volo_d1_224', in_chans=40, drop_path_rate=0.0,num_classes=500,pretrained=True)
            # self.m4 = timm.create_model('volo_d1_224', in_chans=40, drop_path_rate=0.0,num_classes=500,pretrained=True)

            # self.fc0=nn.Linear(2000, 500)
            # #self.fc2=nn.Linear(1000,500)


        # def forward(self, x):
            # l2=224
            # x1 = x[:,:,0:l2,0:l2]
            # x2 = x[:,:,0:l2,l2:]
            # x3 = x[:,:,l2:,0:l2]
            # x4 = x[:,:,l2:,l2:]
            # #torch.cuda.empty_cache()
            # #gc.collect()
            # x1 = F.gelu(self.m1(x1))
            # x2 = F.gelu(self.m2(x2))
            # x3 = F.gelu(self.m3(x3))
            # x4 = F.gelu(self.m4(x4))
            # x = torch.cat([x1,x2,x3,x4],dim=1)
            # del x1,x2, x3, x4
            # #x = F.gelu(self.fc1(x))
            # x = self.fc0(x)
            # return x#,x2,x3,x4


    # class Patch_model(nn.Module):
        # def __init__(self):
            # super(Patch_model, self).__init__()
            # self.m0 = timm.create_model('volo_d3_448', in_chans=40, drop_path_rate=0.0,num_classes=2000,pretrained=True)
            # self.m1 = timm.create_model('volo_d1_224', in_chans=40, drop_path_rate=0.0,num_classes=500,pretrained=True)
            # self.m2 = timm.create_model('volo_d1_224', in_chans=40, drop_path_rate=0.0,num_classes=500,pretrained=True)
            # self.m3 = timm.create_model('volo_d1_224', in_chans=40, drop_path_rate=0.0,num_classes=500,pretrained=True)
            # self.m4 = timm.create_model('volo_d1_224', in_chans=40, drop_path_rate=0.0,num_classes=500,pretrained=True)

            # self.fc1=nn.Linear(4000, 1000)
            # self.fc2=nn.Linear(1000,500)


        # def forward(self, x):
            # l2=224
            # x1 = x[:,:,0:l2,0:l2]
            # x2 = x[:,:,0:l2,l2:]
            # x3 = x[:,:,l2:,0:l2]
            # x4 = x[:,:,l2:,l2:]
            # #torch.cuda.empty_cache()
            # #gc.collect()
            # x = F.gelu(self.m0(x))
            # x1 = F.gelu(self.m1(x1))
            # x2 = F.gelu(self.m2(x2))
            # x3 = F.gelu(self.m3(x3))
            # x4 = F.gelu(self.m4(x4))
            # x0 = torch.cat([x1,x2,x3,x4],dim=1)
            # del x1,x2, x3, x4
            # x = torch.cat([x,x0],dim=1)#x*x0
            # x = F.gelu(self.fc1(x))
            # x = self.fc2(x)
            # return x#,x2,x3,x4

    # class Net0(nn.Module):
        # def __init__(self):
            # super(Net0, self).__init__()
            # self.m1 = timm.create_model('volo_d1_384', in_chans=40, drop_path_rate=0,num_classes=500,pretrained=True)



        # def forward(self, x):
            # x = self.m1(x)


            # return x#,x2,x3,x4

    # class Net2(nn.Module):
        # def __init__(self):
            # super(Net2, self).__init__()
            # self.m1 = timm.create_model('volo_d3_448', in_chans=40, drop_path_rate=0,num_classes=1000,pretrained=True)



        # def forward(self, x):
            # x = self.m1(x)


            # return x#,x2,x3,x4

    # class Net3(nn.Module):
        # def __init__(self):
            # super(Net3, self).__init__()
            # self.m1 = timm.create_model('volo_d3_448', in_chans=40, drop_path_rate=0,num_classes=500,pretrained=True)



        # def forward(self, x):
            # x = self.m1(x)


            # return x#,x2,x3,x4
            
    # class Patch_model5(nn.Module):
        # def __init__(self):
            # super(Patch_model5, self).__init__()
            # self.m1 = timm.create_model('volo_d1_224', in_chans=30, drop_path_rate=0.4,num_classes=500,pretrained=True)
            # self.m2 = timm.create_model('volo_d1_224', in_chans=30, drop_path_rate=0.4,num_classes=500,pretrained=True)
            # self.m3 = timm.create_model('volo_d1_224', in_chans=30, drop_path_rate=0.4,num_classes=500,pretrained=True)
            # self.m4 = timm.create_model('volo_d1_224', in_chans=30, drop_path_rate=0.4,num_classes=500,pretrained=True)
            # self.fc0=nn.Linear(2000,500)

        # def forward(self, x):
            # l2=224
            # x1 = x[:,:,0:l2,0:l2]
            # x2 = x[:,:,0:l2,l2:]
            # x3 = x[:,:,l2:,0:l2]
            # x4 = x[:,:,l2:,l2:]

            # x1 = F.relu(self.m1(x1))
            # x2 = F.relu(self.m2(x2))
            # x3 = F.relu(self.m3(x3))
            # x4 = F.relu(self.m4(x4))

            # x = torch.cat([x1,x2,x3,x4],dim=1)
            # x = self.fc0(x)
            # return x#,x2,x3,x4



    # model6 = Patch_model3()
    # model6.load_state_dict(torch.load('models/speed_model_volod1_patch_4x224px_dispBrown_4_21'))
    # model6.to(device).eval()

    # model7 = Patch_model()
    # model7.load_state_dict(torch.load('models/speed_model_Volo224-448_dispBrown_patch_v3'))
    # model7.to(device).eval()

    # model8 =Patch_model2()
    # model8.load_state_dict(torch.load('models/speed_model_Volo224-448_dispBrown_patch_v1'))
    # model8.to(device).eval()

    # model17 =Patch_model4()
    # model17.load_state_dict(torch.load('models/speed_model_patch448px_disp300_4_23_v2'))
    # model17.to(device).eval()

    # model5 = timm.create_model('volo_d3_448', in_chans=40, drop_path_rate=.0,num_classes=1000,pretrained=False)
    # model5.load_state_dict(torch.load('models/speed_model_volod3_448px_dispBrown_4_22_v2'))
    # model5.to(device).eval()

    # model1 = timm.create_model('volo_d3_448', in_chans=40, drop_path_rate=.0,num_classes=1000,pretrained=False)
    # model1.load_state_dict(torch.load('models/speed_model_volod3_448px_dispBrown_4_22'))
    # model1.to(device).eval()

    # model2 = timm.create_model('twins_svt_small', in_chans=40, drop_path_rate=.0,num_classes=500,pretrained=False)
    # model2.load_state_dict(torch.load('models/speed_model_TwinsSvtSmall_500px_opposite_dispBrown_4_21'))
    # model2.to(device).eval()

    # model3 = timm.create_model('volo_d2_224', in_chans=40, drop_path_rate=.0,num_classes=500,pretrained=False)
    # model3.load_state_dict(torch.load('models/speed_model_volod1_224px_disp_4_21'))
    # model3.to(device).eval()

    # model4 = timm.create_model('volo_d1_224', in_chans=40, drop_path_rate=.0,num_classes=500,pretrained=False)
    # model4.load_state_dict(torch.load('models/speed_model_volod1_224px_dispBrown_4_21'))
    # model4.to(device).eval()

    # model18 = timm.create_model('volo_d1_384', in_chans=40, drop_path_rate=.0,num_classes=500,pretrained=False)
    # model18.load_state_dict(torch.load('models/speed_model_384px_disp300_4_23'))
    # model18.to(device).eval()

    # model19 = timm.create_model('volo_d2_384', in_chans=40, drop_path_rate=.0,num_classes=500,pretrained=False)
    # model19.load_state_dict(torch.load('models/speed_model_384px_disp300_4_23_v2'))
    # model19.to(device).eval()

    # model20 = timm.create_model('volo_d3_448', in_chans=40, drop_path_rate=.0,num_classes=500,pretrained=False)
    # model20.load_state_dict(torch.load('models/speed_model_volod3_448px_disp300_4_25'))
    # model20.to(device).eval()

    # model9 = Net0()
    # model9.load_state_dict(torch.load('models/speed_model_Volo384_dispBrown_full'))
    # model9.to(device).eval()

    # #model10 = Net0()
    # #model10.load_state_dict(torch.load('models/speed_model_Volo384_dispBrown_fullv2'))
    # #model10.to(device).eval()

    # model11 = Net0()
    # model11.load_state_dict(torch.load('models/speed_model_Volo384_dispBrown_fullv3'))
    # model11.to(device).eval()

    # model12 = Net0()
    # model12.load_state_dict(torch.load('models/speed_model_Volo384_dispBrown_full_new_v1'))
    # model12.to(device).eval()

    # model13 = Net2()
    # model13.load_state_dict(torch.load('models/speed_model_Volo448_dispBrown'))
    # model13.to(device).eval()

    # model14 = Net3()
    # model14.load_state_dict(torch.load('models/speed_model_Volo448_dispBrown_full'))
    # model14.to(device).eval()

    # model15 = timm.create_model('swinv2_small_window16_256', in_chans=40, drop_path_rate=.0,num_classes=500,pretrained=False)
    # model15.load_state_dict(torch.load('models/speed_model_patch_swin_256px_dispBrown_4_23'))
    # model15.to(device).eval()

    # # #model16 = timm.create_model('botnet26t_256', in_chans=40, drop_path_rate=.0,num_classes=500,pretrained=False)
    # # #model16.load_state_dict(torch.load('models/speed_model_ByobNet26_256px_dispBrown_4_23'))
    # # #model16.to(device).eval()

    # model21 = timm.create_model('volo_d1_384', in_chans=35, drop_path_rate=.0,num_classes=500,pretrained=False)
    # model21.load_state_dict(torch.load('models/speed_model_volod1_384px_disp_lbm_5_04_max216_7189'))
    # model21.to(device).eval()

    # model22 = timm.create_model('volo_d1_384', in_chans=30, drop_path_rate=.0,num_classes=500,pretrained=False)
    # model22.load_state_dict(torch.load('models/speed_model_volod1_384px_disp_lbm_5_04_v3'))
    # model22.to(device).eval()

    # model23 = timm.create_model('volo_d1_224', in_chans=30, drop_path_rate=.0,num_classes=500,pretrained=False)
    # model23.load_state_dict(torch.load('models/speed_model_volod1_384px_disp_all_5_04_v4'))
    # model23.to(device).eval()

    # model24 = timm.create_model('volo_d1_384', in_chans=40, drop_path_rate=.0,num_classes=500,pretrained=False)
    # model24.load_state_dict(torch.load('models/speed_model_volod2_384px_disp_lbm_5_04_max122_8786_v2'))
    # model24.to(device).eval()
    
    # model25 = timm.create_model('volo_d3_448', in_chans=30, drop_path_rate=.0,num_classes=500,pretrained=False)
    # model25.load_state_dict(torch.load('models/speed_model_volod3_448px_disp_lbm_5_07'))
    # model25.to(device).eval()

    # model26 = timm.create_model('volo_d3_448', in_chans=30, drop_path_rate=.0,num_classes=500,pretrained=False)
    # model26.load_state_dict(torch.load('models/speed_model_volod3_448px_disp_lbm_5_07_v2'))
    # model26.to(device).eval()
    
    # model27 = Patch_model5()
    # model27.load_state_dict(torch.load('models/speed_model_patch_448px_disp_lbm_5_07_v4'))
    # model27.to(device).eval()
    
    # # model28 = Patch_model5()
    # # model28.load_state_dict(torch.load('models/speed_model_patch_448px_disp_lbm_5_07_v3'))
    # # model28.to(device).eval()
    
    # # model29 = Patch_model5()
    # # model29.load_state_dict(torch.load('models/speed_model_patch_448px_disp_lbm_5_07_v2'))
    # # model29.to(device).eval()
    
    # # model30 = Patch_model5()
    # # model30.load_state_dict(torch.load('models/speed_model_patch_448px_disp_lbm_5_07'))
    # # model30.to(device).eval()
    

    # outputs=[]
    # outputs2=[]
    # torch.cuda.empty_cache()
    # gc.collect()

        

    # max24 = 140.0
    # max21 = 216.7189

    # with torch.no_grad():
        # for x in test_dataloader:
            # if torch.mean(x)<.5:
                # x = 1-x
            # # x1 = tv.transforms.functional.resize(x, 448)
            # x2 = tv.transforms.functional.resize(x, 224)
            # x3 = tv.transforms.functional.resize(x, 384)
            # x4 = tv.transforms.functional.resize(x, 256)
            # #out1_1000 = (model1((1-x1).to(device)).detach().cpu().numpy())
            # #out1 = np.zeros([batch_size,500])
            # #for i in range(len(out1)):
            # #    out1[i,:] = interpolate_vectors(out1_1000[i,:],500)
            # out2 = np.sort(model2((x).to(device)).detach().cpu().numpy())
            # out11 = np.sort(model11((x3).to(device)).detach().cpu().numpy())
            # #out10 =(model10((x3).to(device)).detach().cpu().numpy())
            # out9 = np.sort(model9((x3).to(device)).detach().cpu().numpy())
            # out8 = np.sort(model8((x).to(device)).detach().cpu().numpy())
            # out7 =np.sort(model7((x).to(device)).detach().cpu().numpy())
            # out6 = np.sort(model6((1-x).to(device)).detach().cpu().numpy())
            # #out3 = np.sort(model3((1-x2).to(device)).detach().cpu().numpy())
            # out4 = np.sort(model4((x2).to(device)).detach().cpu().numpy())
            # #out5 = np.sort(model5((x1).to(device)).detach().cpu().numpy())
            # #out12 = np.sort(model12((x3).to(device)).detach().cpu().numpy())
            # #out0 = (model((1-x1).to(device)).detach().cpu().numpy())
            # out13_1000 = np.sort(model13((x).to(device)).detach().cpu().numpy())
            # out13 = np.zeros([batch_size,500])
            # for i in range(len(out13)-1):
                 # out13[i,:] = interpolate_vectors(out13_1000[i,:],500)
            # out14 = np.sort(model14((x).to(device)).detach().cpu().numpy())
            # out15 = np.sort(model15((1-x4).to(device)).detach().cpu().numpy())
            # # #out16 = (model16((1-x4).to(device)).detach().cpu().numpy())
            # out17 = np.sort(model17((1-x).to(device)).detach().cpu().numpy())
            # out18 = np.sort(model18((x3).to(device)).detach().cpu().numpy())
            # out19 = np.sort(model19((1-x3).to(device)).detach().cpu().numpy())
            # out20 = np.sort(model20((1-x).to(device)).detach().cpu().numpy())
            # #out21 = np.sort(model21((x3[:,5:,:,:]).to(device)).detach().cpu().numpy())*max21
            # #out22 = np.sort(model22((1-x3[:,5:35,:,:]).to(device)).detach().cpu().numpy())
            
            # #out23 = np.sort(model23((1-x2[:,5:35,:,:]).to(device)).detach().cpu().numpy())
            # out25 = np.sort(model25((1-x[:,5:35,:,:]).to(device)).detach().cpu().numpy())
            # out26 = np.sort(model26((x[:,5:35,:,:]).to(device)).detach().cpu().numpy())
            # out26_2 = np.sort(model26((1-x[:,5:35,:,:]).to(device)).detach().cpu().numpy())
            # out27 = np.sort(model27((1-x[:,5:35,:,:]).to(device)).detach().cpu().numpy())
            # #out28 = np.sort(model28((1-x[:,5:35,:,:]).to(device)).detach().cpu().numpy())
            # # out29 = np.sort(model29((1-x[:,5:35,:,:]).to(device)).detach().cpu().numpy())
            # #out30 = np.sort(model30((1-x[:,5:35,:,:]).to(device)).detach().cpu().numpy())

            # out24 = np.sort(model24((1-x3).to(device)).detach().cpu().numpy())*max24
            

            # out14_2 = np.sort(model14((1-x).to(device)).detach().cpu().numpy())
            # #out15_2 = np.sort(model15((x4).to(device)).detach().cpu().numpy())
            # #out16 = (model16((1-x4).to(device)).detach().cpu().numpy())
            # out17_2 = np.sort(model17((x).to(device)).detach().cpu().numpy())
            # #out18_2 = np.sort(model18((1-x3).to(device)).detach().cpu().numpy())
            # out19_2 = np.sort(model19((x3).to(device)).detach().cpu().numpy())
            # #out20_2 = np.sort(model20((x).to(device)).detach().cpu().numpy())

            # out = (out6*8+out7*2+out15*4+out20+out25*4+out27)/19
            # if class_num==4:
                # #1ulh
                # #class 3 (x<3)
                # out=(out*160+out4*6+(out14)*15+out14_2*60+out17*50+out18*15+(out17_2)*20+(out19_2)*25+(out20)*10+(out8)*1.5+(out9)*.5+out26_2*2+out24)/365
                # out = out*.75
            # elif class_num==3:
                # #class 2 (3<x<6)
                # out=(out*400+out2*10+out6*4+out11*5+out14*4+out15*2+out17*10+out18*8+out19*26+out20*2+out24*9)/480
                # out = out*.95
            # elif class_num==2:
                # #5ulh
                # #class 1 (6<x<10)
                # out = (out*200+out9*90+out11*60+out27*10+out24*40+out26*80+out19*10+out18*10+out13*20)/520
                # #out=(out*50+out3*8+out2*5+out8*15+out11*33+out13*50+out14*5+out18*17+out19*17+out6*5+out2*5+out27*5+out9*40+out11*20)/270
                # out = out*1.65
            # elif class_num==1:
                # #5ulh
                # #class 1 (6<x<10)
                # out = (out*200+out9*120+out11*90+out27*10+out24*60+out26*120+out19*10+out18*10+out13*20)/640
                # #out=(out*50+out3*8+out2*5+out8*15+out11*33+out13*50+out14*5+out18*17+out19*17+out6*5+out2*5+out27*5+out9*40+out11*20)/270
                # out = out*1.8
            # else:
                # #class 0 (x>10)
                # out = (out*200+out9*120+out11*110+out27*10+out24*80+out26*160+out19*10+out18*10+out13*20)/720
                # out = out*2
            # # outputs.append(out)
            
            
            # #out = (out6*8+out7*2+out15*4+out25*4)/18
            # outputs.append(out)


    # stacked_out = np.vstack(outputs)
    # out = np.reshape(stacked_out,[stacked_out.shape[0]*stacked_out.shape[1], ])

    # directory='../tmate_trajectories/'
    filename = os.listdir(directory)[f_idx]
    print(filename)
    df = pd.read_csv(os.path.join(directory,filename),header=0, skiprows=[1,2,3])
    tracks = df.TRACK_ID.unique()
    # vels=1
    # for i in range(len(tracks)):
        # idx = tracks[i]
        # if df[df.TRACK_ID==idx].sort_values(by='FRAME').FRAME.iloc[-1] < (images.shape[0]*images.shape[1]):
            # posx = df[df.TRACK_ID==idx].sort_values(by='FRAME').POSITION_X
            # posy = df[df.TRACK_ID==idx].sort_values(by='FRAME').POSITION_Y

            # vel = (np.sqrt(posx.diff()**2+posy.diff()**2)).dropna()
        
            # vels = np.hstack([vels, vel])
            
    # vels=vels[~np.isnan(vels)]
    # vels = vels[vels<200]
    # vels = vels[vels>0]
    # out = out[out<200]
    # out = out[out>0]

    criterion = nn.MSELoss()
    # #speed_loss = criterion(torch.tensor(np.sort(interpolate_vectors(np.sort(out[out>0]), len(vels)))), torch.tensor(np.sort(vels)))
    # mean_loss =  np.abs(np.mean(vels)-np.mean(out))
    # print('Mean Speed from PT: ', np.mean(vels))
    
    # print('Absolute Error of Average Speed: ',mean_loss)
    # q=np.sort(interpolate_vectors(np.sort(out), len(vels)))
    # w=np.sort(vels)
    # speed_loss = criterion(torch.tensor(q), torch.tensor(w))
    # print('Speed Loss: ', speed_loss)
    # file_error.append(mean_loss)
    # file_mse.append(speed_loss)

    
    # final_out.append(q)
    # final_true.append(w)
    
    # plt.figure(figsize=(8, 6))
    # a,b,c,d = betaprime.fit(q)

    # xtorch= np.linspace(betaprime.ppf(0.0001, a,b,c,d),
                    # betaprime.ppf(0.9999, a,b,c,d), target_length)

    # distr_torch1=betaprime.pdf(xtorch, a,b,c,d)
    # plt.plot(xtorch,distr_torch1,c='r')


    # a,b,c,d = betaprime.fit(w)
        
    # xtorch= np.linspace(betaprime.ppf(0.0001, a,b,c,d),
                    # betaprime.ppf(0.9999, a,b,c,d), target_length)

    # distr_torch=betaprime.pdf(xtorch, a,b,c,d)

    # plt.plot(xtorch,distr_torch,c='g')
    # plt.ylabel('Probability Density')
    # plt.xlabel('Speed (pixels/frame)')
    # plt.title('Speed Vectors for '+str(base), fontsize=12)
    # plt.legend(['Predictions', 'Particle Tracking (TrackMate)'])
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig('Speed_'+str(f_idx)+'.png')
    # w1 = wasserstein_distance(distr_torch, distr_torch1)
    # print ('Earth Movers Distance: ', w1)
    
    # speed_w1.append(w1)
    

    
    
    ######################################
    ######   Turn Angle Prediction  ######
    ######################################

    tracks = df.TRACK_ID.unique()
    angles=1
    for i in range(len(tracks)):
        idx = tracks[i]
        if df[df.TRACK_ID==idx].sort_values(by='FRAME').FRAME.iloc[-1] < (images.shape[0]*images.shape[1]):
            posx = df[df.TRACK_ID==idx].sort_values(by='FRAME').POSITION_X
            posy = df[df.TRACK_ID==idx].sort_values(by='FRAME').POSITION_Y

            angle_array = calculate_angles_test(np.array(posx),np.array(posy))
            
        
            angles = np.hstack([angles, angle_array])
    angles=np.abs(angles[~np.isnan(angles)])
    angles = angles[angles>0]
    angles = angles[angles<360]
    angles = np.sort(angles)
    
    class Net0(nn.Module):
        def __init__(self):
            super(Net0, self).__init__()
            self.m1 = timm.create_model('volo_d1_384', in_chans=40, drop_path_rate=0,num_classes=500,pretrained=True)

        def forward(self, x):
            x = self.m1(x)

    model0 = timm.create_model('volo_d3_448', in_chans=40, drop_path_rate=0,num_classes=500,pretrained=False)
    model0.load_state_dict(torch.load('models/turn_angle_model_volod3_448px_disp_all_4_25'))
    model0.to(device).eval()

    model1 = timm.create_model('volo_d1_384', in_chans=40, drop_path_rate=0,num_classes=500,pretrained=False)
    model1.load_state_dict(torch.load('models/turn_angle_model_Volod1_384_dispBrownv4'))
    model1.to(device).eval()

    model2 = timm.create_model('volo_d1_384', in_chans=40, drop_path_rate=0,num_classes=500,pretrained=False)
    model2.load_state_dict(torch.load('models/turn_angle_model_Volod1_384_dispBrownv3'))
    model2.to(device).eval()

    model3 = timm.create_model('volo_d1_384', in_chans=40, drop_path_rate=0,num_classes=500,pretrained=False)
    model3.load_state_dict(torch.load('models/turn_angle_model_Volod1_384_dispBrownv2'))
    model3.to(device).eval()

    model4 = timm.create_model('volo_d1_384', in_chans=40, drop_path_rate=0,num_classes=500,pretrained=False)
    model4.load_state_dict(torch.load('models/turn_angle_model_Volod1_384_dispBrown'))
    model4.to(device).eval()

    model5 = timm.create_model('volo_d3_448', in_chans=30, drop_path_rate=0,num_classes=500,pretrained=False)
    model5.load_state_dict(torch.load('models/angle_model_volod3_448px_disp_all_5_5'))
    model5.to(device).eval()

    model6 = timm.create_model('volo_d1_384', in_chans=30, drop_path_rate=0,num_classes=500,pretrained=False)
    model6.load_state_dict(torch.load('models/angle_model_volod1_384px_disp_all_5_5'))
    model6.to(device).eval()

    model7 = timm.create_model('volo_d1_384', in_chans=30, drop_path_rate=0,num_classes=500,pretrained=False)
    model7.load_state_dict(torch.load('models/angle_model_volod1_384px_disp_all_5_5_v2'))
    model7.to(device).eval()

    model8 = timm.create_model('volo_d1_384', in_chans=30, drop_path_rate=0,num_classes=500,pretrained=False)
    model8.load_state_dict(torch.load('models/angle_model_volod1_384px_disp_all_5_5_v3'))
    model8.to(device).eval()
    
    model9 = timm.create_model('volo_d3_448', in_chans=30, drop_path_rate=0,num_classes=500,pretrained=False)
    model9.load_state_dict(torch.load('models/angle_model_volod3_448px_disp_lbm_5_7'))
    model9.to(device).eval()
    
    constants = [115,0,0.25,.289,.246,+.02]
    weights = [['11',5],
    #['14',.5],
    ['16',38.5],
    ['17',22.6],
    ['19',3],
    ['5',14.5],
    ['10',3.4],
    #['7',.6],
    ['13',32.2],
    ['6',2],
    ['18',1.5]]
    outputs_class = classifier(test_dataloader2,constants,weights)
    mean_out = np.mean(np.vstack(outputs_class),0)
    class_num_ang = np.argmax(mean_out)
    print('Class Number for Angles: ',class_num_ang)
    print(mean_out)
    outputs_ang=[]

    with torch.no_grad():
        for x in test_dataloader:
            #x1 = torchvision.transforms.functional.resize(x, 448)
            x2 = tv.transforms.functional.resize(x, 384)

            out0 =np.sort(model0((x).to(device)).detach().cpu().numpy())
            out1 =np.sort(model1((x2).to(device)).detach().cpu().numpy())*.55
            out2 =np.sort(model2((x2).to(device)).detach().cpu().numpy())*.5
            out3 =np.sort(model3((x2).to(device)).detach().cpu().numpy())*.45
            out4 =np.sort(model4((x2).to(device)).detach().cpu().numpy())*.4


            if class_num_ang==4:
                out=(out2*4+out1*3+out0*2)/8
            elif class_num_ang==3:
                out=(out0*10+out2*2+out1*4+out3*4+out4*5)/25
            elif class_num_ang==2:
                out=(out0+out1+out4+out3*3)/6
            elif class_num_ang==1:
                out=(out0*3+out4*1+out3*4)/8
            else:
                out=(out0*4+out4*1)/5

            outputs_ang.append(out)
          
    outputs2_ang=[]
    import torchvision
    with torch.no_grad():
        for x in test_dataloader2:
            #x1 = torchvision.transforms.functional.resize(x, 448)
            x2 = tv.transforms.functional.resize(x, 384)

            out5 =np.sort(model5((x).to(device)).detach().cpu().numpy())
            out6 =np.sort(model6((x2).to(device)).detach().cpu().numpy())
            out7 =np.sort(model7((x2).to(device)).detach().cpu().numpy())
            out8 =np.sort(model8((x2).to(device)).detach().cpu().numpy())
            out9 =np.sort(model9((x).to(device)).detach().cpu().numpy())*.25
            out10 =np.sort(model8((x2).to(device)).detach().cpu().numpy())*1.25

            if class_num_ang==4:
                out=(out5*1+out6*5+out7*10+out8*2+out10*8)/26
            elif class_num_ang==3:
                out=(out5*6+out6*3+out7*2+out8*2+out9*4+out10*2)/19
            elif class_num_ang==2:
                out=(out9*3+out5*5+out8+out10*1)/10
            elif class_num_ang==1:
                out=(out9*2+out5*2)/4
            #out=out0
            # elif class_num==2:
            #     out=out1
            #out = (out0+out1+out2)/2

            outputs2_ang.append(out)
    
    print('0: ',np.mean(out0), np.min(out0), np.max(out0))
    print('1: ',np.mean(out1), np.min(out1), np.max(out1))
    print('2: ',np.mean(out2), np.min(out2), np.max(out2))
    print('3: ',np.mean(out3), np.min(out3), np.max(out3))
    print('4: ',np.mean(out4), np.min(out4), np.max(out4))
    print('5: ',np.mean(out5), np.min(out5), np.max(out5))
    print('6: ',np.mean(out6), np.min(out6), np.max(out6))
    print('7: ',np.mean(out7), np.min(out7), np.max(out7))
    print('8: ',np.mean(out8), np.min(out8), np.max(out8))
    print('9: ',np.mean(out9), np.min(out9), np.max(out9))
    print('10: ',np.mean(out10), np.min(out10), np.max(out10))

    #print('next batch')  
    newout1 = np.vstack(outputs_ang)
    out1 = np.mean(np.abs(newout1),0)
    out1=out1[out1>0]

    newout2=np.vstack(outputs2_ang)
    out2 = np.mean(np.abs(newout2),0)
    out2=out2[out2>0]

    out1 = interpolate_vectors(out1,len(out2))
    print('final 1: ',np.mean(out1))
    print('final 2: ',np.mean(out2))


    if class_num_ang==4:
        out = 1.35*(out1*1+out2*1)/2
    elif class_num_ang==3:
        out = 1.2*(out1*5+out2*1)/6
    elif class_num_ang==2:
        out=1.05*(out1*7+out2*1)/8
    elif class_num_ang==1:
        out=(out1*9+out2*1)/10
    out.shape
    out = out[out>0]
    out = out[out<360]
    
    mean_loss =  np.abs(np.mean(angles)-np.mean(out))
    print('Min, Max: ', np.min(angles), np.max(angles))
    print('Min out, Max out: ', np.min(out), np.max(out))
    print('Mean Turn Angle from PT: ', np.mean(angles))
    print('Mean Turn Angle from DeepTrackStat: ', np.mean(out))
    
    print('Absolute Error of Average Turn Angle: ',mean_loss)
    q=np.sort(interpolate_vectors(np.sort(out), len(angles)))
    w=np.sort(angles)
    speed_loss = criterion(torch.tensor(q), torch.tensor(w))
    print('Turn Angle Loss: ', speed_loss)  
    file_error_ang.append(mean_loss)
    file_mse_ang.append(speed_loss)
    plt.figure(figsize=(8, 6))
    a,b = expon.fit(q)
    xtorch= np.linspace(expon.ppf(0.0001, a,b),
                    expon.ppf(0.9999, a,b), target_length)
    distr_torch1=expon.pdf(xtorch, a,b)
    plt.plot(xtorch,distr_torch1,c='r')
    a,b = expon.fit(w)     
    xtorch= np.linspace(expon.ppf(0.0001, a,b),
                    expon.ppf(0.9999, a,b), target_length)
    distr_torch=expon.pdf(xtorch, a,b)
    plt.plot(xtorch,distr_torch,c='g')
    plt.ylabel('Probability Density')
    plt.xlabel('Turn Angle (degrees)')
    plt.title('Turn Angle Vectors for '+str(base), fontsize=12)
    plt.legend(['Predictions', 'Particle Tracking (TrackMate)'])
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('Turn_angle_'+str(f_idx)+'.png')
    w1 = wasserstein_distance(distr_torch, distr_torch1)
    print ('Earth Movers Distance: ', w1)
    ang_w1.append(w1)
    final_out_ang.append(q)
    final_true_ang.append(w)
    
    
    ###################
    ######   Vx  ######
    ###################
    
    print('Starting Vx')

    
    from torch import nn
    import torch.nn.functional as F

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
            
    class Net4(nn.Module):
        def __init__(self):
            super(Net4, self).__init__()
            self.m1 = timm.create_model('volo_d1_384', in_chans=40, drop_path_rate=0.2,num_classes=500,pretrained=True) 
            self.m2 = timm.create_model('volo_d1_384', in_chans=40, drop_path_rate=0.2,num_classes=500,pretrained=True) 
            self.fc1=nn.Linear(2,2)
        def forward(self, x):
            vx = F.gelu(self.m1(x))
            vy = F.gelu(self.m2(x))
            vx = vx.unsqueeze(2)
            vy = vy.unsqueeze(2)
            x = torch.cat([vx,vy],dim=2)
            x = self.fc1(x)
            return x#,x2,x3,x4
            
    model0 = timm.create_model('volo_d4_448', in_chans=40, drop_path_rate=.0,num_classes=500,pretrained=False)
    model0.load_state_dict(torch.load('models/vx_model_volod4_448px_disp_all_max151_1078_min-49_9241'))
    model0.to(device).eval()

    model1 = timm.create_model('volo_d2_384', in_chans=40, drop_path_rate=.0,num_classes=500,pretrained=False)
    model1.load_state_dict(torch.load('models/vx_model_volod1_384px_noscale_disp_500_4_27'))
    model1.to(device).eval()

    model2 = timm.create_model('volo_d1_384', in_chans=40, drop_path_rate=.0,num_classes=500,pretrained=False)
    model2.load_state_dict(torch.load('models/vx_model_volod1_384px_opposite_disp_300_4_27'))
    model2.to(device).eval()

    model3 = timm.create_model('regnetx_032', in_chans=40, drop_path_rate=.0,num_classes=500,pretrained=False)
    model3.load_state_dict(torch.load('models/vx_model_regnetx32_384px_disp_all_4_27'))
    model3.to(device).eval()

    model4 = timm.create_model('volo_d1_384', in_chans=40, drop_path_rate=.0,num_classes=500,pretrained=False)
    model4.load_state_dict(torch.load('models/vx_model_volod1_384px_disp_all_4_27'))
    model4.to(device).eval()

    model5 = timm.create_model('volo_d3_448', in_chans=40, drop_path_rate=.0,num_classes=500,pretrained=False)
    model5.load_state_dict(torch.load('models/vx_model_volod3_448px_disp_all_4_27'))
    model5.to(device).eval()

    model6 = timm.create_model('regnetx_016', in_chans=40, drop_path_rate=.0,num_classes=500,pretrained=False)
    model6.load_state_dict(torch.load('models/vx_model_regnetx_384px_disp_all_4_27'))
    model6.to(device).eval()

    model7 = Net4()
    model7.load_state_dict(torch.load('models/vel_model_volod1_384px_disp_all_4_27'))
    model7.to(device).eval()

    model8 = timm.create_model('volo_d3_448', in_chans=30, drop_path_rate=.0,num_classes=500,pretrained=False)
    model8.load_state_dict(torch.load('models/vx_model_volod3_448px_disp_all_max219_541_min-49_9241'))
    model8.to(device).eval()
    
    def calculate_velocity( y_positions):
        # Calculate velocities
        y_velocities = (y_positions[1:] - y_positions[:-1]) 
                
        return  y_velocities
    
    outputs=[]
    outputs2=[]
    torch.cuda.empty_cache()
    gc.collect()


    # print (slope200, slope300, slope500, slope_all)
    # print (int200, int300, int500, int_all)
    # tensor(182.7808) tensor(182.7808) tensor(184.0055) tensor(216.6908)
    # tensor(-16.0141) tensor(-16.0141) tensor(-17.2388) tensor(-49.9241)
    slope200, int200 = 182.7808, -16.0141
    slope_all, int_all = 216.6908, -49.9241
    slope_all2, int_all2 = 230, -45
    #vx_model_volod4_448px_disp_all_max151_1078_min-49_9241
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
            x1 = torchvision.transforms.functional.resize(x, 448)
            out0 = np.sort(model0((x).to(device)).detach().cpu().numpy())*slope1+int1
           # x=1-x
            x2 = torchvision.transforms.functional.resize(x, 384)
            
            out1 = np.sort(model1((x2).to(device)).detach().cpu().numpy())
            out2 = np.sort(model2((1-x2).to(device)).detach().cpu().numpy())*slope200+int200
            out4 = np.sort(model4((x2).to(device)).detach().cpu().numpy())*slope_all+int_all
            out5 = np.sort(model5((x).to(device)).detach().cpu().numpy())*slope_all2+int_all2
            out6 = np.sort(model6((x).to(device)).detach().cpu().numpy())*50
            out7= np.sort(model7((x2).to(device)).detach().cpu().numpy())*slope_all2+int_all2
            out8 = out7[:,:,0]*1.5-10
            out9 = out7[:,:,0]*1.3-17
            out7 = out7[:,:,0]
            out11 = (out0-1.0)*4

            
            ##class1
            if class_num==2:
                out=(out11*10+out0*12+out1+out4*4+out5+out2*3+out7+out8*2)/34

            elif class_num==3:
                out=(out11*4+out0*7+out1*8+out2*10+out4*6+out8*3+out9*5)/43

            elif class_num==4:
                out=(out0*12+out1*18+out9*10)/47

            outputs.append(out)
            #outputs2.append(out2)
            
    print('0: ',np.mean(out0), np.min(out0), np.max(out0))
    print('1: ',np.mean(out1), np.min(out1), np.max(out1))
    print('2: ',np.mean(out2), np.min(out2), np.max(out2))
    print('4: ',np.mean(out4), np.min(out4), np.max(out4))
    print('5: ',np.mean(out5), np.min(out5), np.max(out5))
    print('6: ',np.mean(out6), np.min(out6), np.max(out6))
    print('7: ',np.mean(out7), np.min(out7), np.max(out7))
    print('8: ',np.mean(out8), np.min(out8), np.max(out8))
    print('9: ',np.mean(out9), np.min(out9), np.max(out9))
    print('11: ',np.mean(out11), np.min(out11), np.max(out11))

    newout = np.vstack(outputs)
    outputs = np.mean(newout,0)

    max2 = 219.541
    min2 = -49.9241
    slope2 = max2-min2
    int2 = min2

    outputs2=[]
    with torch.no_grad():
        for x in test_dataloader2:
            out10 = np.sort(model8((x).to(device)).detach().cpu().numpy())*slope2+int2

            outputs2.append(out8)
    newout = np.vstack(outputs2)
    outputs2 = np.mean(newout,0)
    
    print('10: ',np.mean(out10), np.min(out10), np.max(out10))
        
    if class_num==4:
        out3 = (outputs*16+outputs2)/17
    elif class_num==3:
        out3 = (outputs*16+ outputs2)/17
    elif class_num==2:
        out3 = (outputs*15+ outputs2)/16
    elif class_num==1:
        out3 = (outputs*14+ outputs2)/17

    vels=1
    for i in range(len(tracks)):
        idx = tracks[i]
        if df[df.TRACK_ID==idx].sort_values(by='FRAME').FRAME.iloc[-1] < (images.shape[0]*images.shape[1]):
            posy = df[df.TRACK_ID==idx].sort_values(by='FRAME').POSITION_X
            yvel = calculate_velocity(np.array(posy))
        
            vels = np.hstack([vels, yvel])
            
    vels=vels[~np.isnan(vels)]
    vels = vels[vels!=0]
    out3 = out3[out3!=0]
    mean_loss =  np.abs(np.mean(vels)-np.mean(out3))
    print('Min, Max: ', np.min(vels), np.max(vels))
    print('Min out, Max out: ', np.min(out3), np.max(out3))
    print('Mean Vx from PT: ', np.mean(vels))
    
    print('Absolute Error of Average Vx: ',mean_loss)
    q=np.sort(interpolate_vectors(np.sort(out3), len(vels)))
    w=np.sort(vels)
    speed_loss = criterion(torch.tensor(q), torch.tensor(w))
    print('Vx Loss: ', speed_loss)
    
    file_error_vx.append(mean_loss)
    file_mse_vx.append(speed_loss)
    
    plt.figure(figsize=(8, 6))
    a,b = norm.fit(q)

    xtorch= np.linspace(norm.ppf(0.0001, a,b),
                    norm.ppf(0.9999, a,b), target_length)

    distr_torch1=norm.pdf(xtorch, a,b)
    plt.plot(xtorch,distr_torch1,c='r')


    a,b = norm.fit(w)
        
    xtorch= np.linspace(norm.ppf(0.0001, a,b),
                    norm.ppf(0.9999, a,b), target_length)

    distr_torch=norm.pdf(xtorch, a,b)

    plt.plot(xtorch,distr_torch,c='g')
    plt.ylabel('Probability Density')
    plt.xlabel('Speed (pixels/frame)')
    plt.title('Vx Distribution for '+str(base), fontsize=12)
    plt.legend(['Predictions', 'Particle Tracking (TrackMate)'])
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('Vx_'+str(f_idx)+'.png')
    w1 = wasserstein_distance(distr_torch, distr_torch1)
    print ('Earth Movers Distance: ', w1)
    
    vx_w1.append(w1)
    
    final_out_vx.append(np.sort(q))
    final_true_vx.append(w)
    
    ##################
    ######  Vy  ######
    ##################
    
    print('Starting Vy')

    model0 = timm.create_model('volo_d1_384', in_chans=40, drop_path_rate=.0,num_classes=500,pretrained=False)
    model0.load_state_dict(torch.load('models/vy_model_volod1_384px_disp_all_max69_2162_min-66_9786_v3'))
    model0.to(device).eval()

    model1 = timm.create_model('volo_d1_384', in_chans=40, drop_path_rate=.0,num_classes=500,pretrained=False)
    model1.load_state_dict(torch.load('models/vy_model_volod1_384px_disp_all_max69_2162_min-66_9786_v2'))
    model1.to(device).eval()

    model2 = timm.create_model('volo_d1_384', in_chans=40, drop_path_rate=.0,num_classes=500,pretrained=False)
    model2.load_state_dict(torch.load('models/vy_model_volod1_384px_disp_all_max69_2162_min-66_9786'))
    model2.to(device).eval()

    model3 = timm.create_model('volo_d1_224', in_chans=40, drop_path_rate=.0,num_classes=500,pretrained=False)
    model3.load_state_dict(torch.load('models/vy_model_volod1_224px_disp_all_max69_2162_min-66_9786'))
    model3.to(device).eval()

    model4 = timm.create_model('volo_d4_448', in_chans=40, drop_path_rate=.0,num_classes=500,pretrained=False)
    model4.load_state_dict(torch.load('models/vy_model_volod4_448px_disp_all_max64_4513_min-66_1406'))
    model4.to(device).eval()

    model5 = timm.create_model('volo_d1_384', in_chans=40, drop_path_rate=.0,num_classes=500,pretrained=False)
    model5.load_state_dict(torch.load('models/vy_model_volod1_384px_disp_all_4_27'))
    model5.to(device).eval()

    model6 = timm.create_model('volo_d3_448', in_chans=30, drop_path_rate=.0,num_classes=500,pretrained=False)
    model6.load_state_dict(torch.load('models/vy_model_volod3_448px_disp_all_max132_73_min-95_1744'))
    model6.to(device).eval()
    
    outputs=[]
    outputs2=[]
    torch.cuda.empty_cache()
    gc.collect()

    # print (slope200, slope300, slope500, slope_all)
    # print (int200, int300, int500, int_all)
    # tensor(182.7808) tensor(182.7808) tensor(184.0055) tensor(216.6908)
    # tensor(-16.0141) tensor(-16.0141) tensor(-17.2388) tensor(-49.9241)
    outputs=[]
    outputs2=[]
    torch.cuda.empty_cache()
    gc.collect()


    # print (slope200, slope300, slope500, slope_all)
    # print (int200, int300, int500, int_all)
    # tensor(182.7808) tensor(182.7808) tensor(184.0055) tensor(216.6908)
    # tensor(-16.0141) tensor(-16.0141) tensor(-17.2388) tensor(-49.9241)
    max1 = 69.2162
    min1 = -66.9786
    slope1 = max1-min1
    int1 = min1

    max2 = 64.4513
    min2 = -66.1406
    slope2 = max2-min2
    int2 = min2

    with torch.no_grad():
        for x in test_dataloader:
            x1 = torchvision.transforms.functional.resize(x, 448)
            x2 = torchvision.transforms.functional.resize(x, 384)
            x3 = torchvision.transforms.functional.resize(x, 224)
            out0 = np.sort(model0((x2).to(device)).detach().cpu().numpy())*slope1+int1
            out1 = np.sort(model1((x2).to(device)).detach().cpu().numpy())*slope1+int1
            out2 = np.sort(model2((x2).to(device)).detach().cpu().numpy())*slope1+int1
            out3 = np.sort(model3((x3).to(device)).detach().cpu().numpy())*slope1+int1
            out4 = np.sort(model4((x).to(device)).detach().cpu().numpy())*slope2+int2
            out4_1 = out4*1.75+3
            out4_2 = out4*1.55-3
            out4_3 = out4*.7
            out5 = np.sort(model5((x2).to(device)).detach().cpu().numpy())*slope2+int2
            out5_2 = np.sort(model5((x2).to(device)).detach().cpu().numpy())*slope1+int1

            
            ##class1
            if class_num==2:
                out=1.08*(out0+out1+out2*3+out3*2+out4*30+out5+out5_2*5+out4_2*26+out4_1*16+out4_3*1)/84

            if class_num==3:
                out=1.02*(out0+out1+out2+out3*6+out4*30+out5+out5_2*3+out4_2*18+out4_1*16+out4_3*12)/90

            if class_num==4:
                out=.98*(out4*25+out4_2*12+out4_3*36+out4_1*10+out0+out1+out2+out3*10)/102

                    ##class3
            #out=(out0*18+out1*4+out4*1+out2*2)/31

            outputs.append(out)
            #outputs2.append(out2)
    max2 =132.73
    min2 = -95.1744
    slope2 = max2-min2
    int2 = min2
    outputs2=[]
    with torch.no_grad():
        for x in test_dataloader2:

            out6 = np.sort(model6((x).to(device)).detach().cpu().numpy())*slope2+int2


            outputs2.append(out6)
    print('0: ',np.mean(out0), np.min(out0), np.max(out0))
    print('1: ',np.mean(out1), np.min(out1), np.max(out1))
    print('2: ',np.mean(out2), np.min(out2), np.max(out2))
    print('4: ',np.mean(out4), np.min(out4), np.max(out4))
    print('5: ',np.mean(out5), np.min(out5), np.max(out5))
    print('5_2: ',np.mean(out5_2), np.min(out5_2), np.max(out5_2))
    print('6: ',np.mean(out6), np.min(out6), np.max(out6))
    print('3: ',np.mean(out3), np.min(out3), np.max(out3))
    print('4_1: ',np.mean(out4_1), np.min(out4_1), np.max(out4_1))
    print('4_2: ',np.mean(out4_2), np.min(out4_2), np.max(out4_2))
    print('4_3: ',np.mean(out4_3), np.min(out4_3), np.max(out4_3))        
    newout = np.vstack(outputs)
    outputs = np.mean(newout,0)

    newout = np.vstack(outputs2)
    outputs2 = np.mean(newout,0)
    out3=1.1*((outputs*15+outputs2*1)/16)-1
    

    vels=1
    for i in range(len(tracks)):
        idx = tracks[i]
        if df[df.TRACK_ID==idx].sort_values(by='FRAME').FRAME.iloc[-1] < (images.shape[0]*images.shape[1]):
            posy = df[df.TRACK_ID==idx].sort_values(by='FRAME').POSITION_Y
            yvel = calculate_velocity(np.array(posy))
        
            vels = np.hstack([vels, yvel])
    vels=vels[~np.isnan(vels)]
    vels = vels[vels!=0]
    out3 = out3[out3!=0]
    print('Min, Max: ', np.min(vels), np.max(vels))
    print('Min out, Max out: ', np.min(out3), np.max(out3))

    mean_loss =  np.abs(np.mean(vels)-np.mean(out3))
    print('Mean Vy from PT: ', np.mean(vels))
    
    print('Absolute Error of Average Vy: ',mean_loss)
    q=np.sort(interpolate_vectors(np.sort(out3), len(vels)))
    w=np.sort(vels)
    speed_loss = criterion(torch.tensor(q), torch.tensor(w))
    print('Vy Loss: ', speed_loss)
    
    file_error_vy.append(mean_loss)
    file_mse_vy.append(speed_loss)
    
    plt.figure(figsize=(8, 6))
    a,b = norm.fit(q)

    xtorch= np.linspace(norm.ppf(0.0001, a,b),
                    norm.ppf(0.9999, a,b), target_length)

    distr_torch1=norm.pdf(xtorch, a,b)
    plt.plot(xtorch,distr_torch1,c='b')


    a,b = norm.fit(w)
        
    xtorch= np.linspace(norm.ppf(0.0001, a,b),
                    norm.ppf(0.9999, a,b), target_length)

    distr_torch=norm.pdf(xtorch, a,b)

    plt.plot(xtorch,distr_torch,c='g')
    plt.ylabel('Probability Density')
    plt.xlabel('Speed (pixels/frame)')
    plt.title('Vy Distribution for '+str(base), fontsize=12)
    plt.legend(['Predictions', 'Particle Tracking (TrackMate)'])
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('Vy_'+str(f_idx)+'.png')
    w1 = wasserstein_distance(distr_torch, distr_torch1)
    print ('Earth Movers Distance: ', w1)
    
    final_out_vy.append(q)
    final_true_vy.append(w)
    vy_w1.append(w1)
    
    ########################
    ###  Directionality  ###
    ########################
    # vels=1
    model0 = timm.create_model('volo_d3_448', in_chans=30, drop_path_rate=.0,num_classes=1,pretrained=False)
    model0.load_state_dict(torch.load('models/directionality_volod3_448px_disp_lbm_5_04'))
    model0.to(device).eval()

    model1 = timm.create_model('volo_d1_384', in_chans=30, drop_path_rate=.0,num_classes=1,pretrained=False)
    model1.load_state_dict(torch.load('models/directionality_volod1_384px_disp_lbm_5_15'))
    model1.to(device).eval()


    model2 = timm.create_model('regnetx_064', in_chans=30, drop_path_rate=.0,num_classes=1,pretrained=False)
    model2.load_state_dict(torch.load('models/directionality_regnetx064_384px_disp_lbm_5_15_v2'))
    model2.to(device).eval()

    model3 = timm.create_model('regnetx_064', in_chans=30, drop_path_rate=.0,num_classes=1,pretrained=False)
    model3.load_state_dict(torch.load('models/directionality_regnetx064_384px_disp_lbm_5_15'))
    model3.to(device).eval()
    


    outputs=[]
    torch.cuda.empty_cache()
    gc.collect()

    with torch.no_grad():
        for x in test_dataloader2:
            x2 = torchvision.transforms.functional.resize(x, 384)
            out0 = (model0((x).to(device)).detach().cpu().numpy())
            out1 = (model1((x2).to(device)).detach().cpu().numpy())
            out2 = (model2((x2).to(device)).detach().cpu().numpy())
            out3 = (model3((x2).to(device)).detach().cpu().numpy())
            out = (out3*2+out2+out1*8+out0*2)/13

            outputs.append(out)
    print('directionalities: ', out0, out1, out2, out3)
    directionality = []
    for i in range(len(tracks)):
        idx = tracks[i]
        if df[df.TRACK_ID==idx].sort_values(by='FRAME').FRAME.iloc[-1] < (images.shape[0]*images.shape[1]):
            posx = np.array(df[df.TRACK_ID==idx].sort_values(by='FRAME').POSITION_X)
            posy = np.array(df[df.TRACK_ID==idx].sort_values(by='FRAME').POSITION_Y)

            xpart = posx[~np.isnan(posx)]
            ypart = posy[~np.isnan(posy)]
            try:
                xdist = xpart[-1] - xpart[0]
                ydist = ypart[-1] - ypart[0]
                speed = np.sum(np.sqrt(np.diff(xpart)**2+np.diff(ypart)**2))
                disp = np.sqrt(xdist**2+ydist**2)
                directionality.append(np.nanmean(disp)/np.nanmean(speed))
            except:
                directionality.append(0)
        
    
    direction =  np.nanmean(directionality)
    mean_loss =  np.abs(direction-np.mean(outputs))
    print('Directionality: ', direction)
    
    print('Absolute Error of Average direction: ',mean_loss)
    final_true_direction.append(direction)
    final_out_direction.append(np.mean(outputs))
    final_error_direction.append(mean_loss)
    
    
# print('Average Speed Error for all Files: ',np.nanmean(file_error))
# print('Average MSE (speed) Error for all Files: ',np.nanmean(file_mse))
# print('Average W1 (speed) Error for all Files: ',np.nanmean(speed_w1))

print('Average Angle Error for all Files: ',np.nanmean(file_error_ang))
print('Average MSE (angle) Error for all Files: ',np.nanmean(file_mse_ang))
print('Average W1 (angle) Error for all Files: ',np.nanmean(ang_w1))

print('Average Vx Error for all Files: ',np.nanmean(file_error_vx))
print('Average MSE (Vx) Error for all Files: ',np.nanmean(file_mse_vx))
print('Average W1 (Vx) Error for all Files: ',np.nanmean(vx_w1))

print('Average Vy Error for all Files: ',np.nanmean(file_error_vy))
print('Average MSE (Vy) Error for all Files: ',np.nanmean(file_mse_vy))
print('Average W1 (Vy) Error for all Files: ',np.nanmean(vy_w1))

print('Average Directionality Error for all Files: ',np.nanmean(final_error_direction))



# print(len(final_out))
# print(final_out[0].shape)
# q = np.sort(np.hstack(final_out))
# w = np.sort(np.hstack(final_true))
# print(q.shape)
# plt.plot(q, c='b', linewidth=2)
# plt.plot(w, c='g', linewidth=2)
# plt.xlabel('Vector Index')
# plt.ylabel('Speed (pixels/frame)')
# plt.title('Total Speed Vector for All Test Data', fontsize=12)
# plt.legend(['Predictions', 'Particle Tracking (TrackMate)'])
# plt.grid(True)
# plt.tight_layout()
# plt.savefig('Speed_final.png')
# plt.show()

# a,b,c,d = betaprime.fit(q)
# xtorch= np.linspace(betaprime.ppf(0.0001, a,b,c,d),
                # betaprime.ppf(0.9999, a,b,c,d), target_length)
# distr_torch=betaprime.pdf(xtorch, a,b,c,d)
# plt.plot(xtorch,distr_torch,c='r')
# a,b,c,d = betaprime.fit(w) 
# xtorch= np.linspace(betaprime.ppf(0.0001, a,b,c,d),
                # betaprime.ppf(0.9999, a,b,c,d), target_length)
# distr_torch=betaprime.pdf(xtorch, a,b,c,d)
# plt.plot(xtorch,distr_torch,c='g')
# plt.title('Total Speed Distribution for All Test Data')
# plt.xlabel('Speed (pixels/frame)')
# plt.ylabel('Probability Density')
# plt.legend(['Prediction', 'Ground Truth'])
# plt.xscale('log')
# plt.yscale('log')
# plt.savefig('Speed_distribution_final.png')
# plt.show()
plt.scatter(final_out_direction, final_true_direction)
plt.title('Directionality Predictions')
plt.xlabel('Predictions')
plt.ylabel('Particle Tracking')

# Angles
q = np.sort(np.hstack(final_out_ang)[::400])
print(q.shape)
w = np.sort(np.hstack(final_true_ang)[::400])
plt.plot(q, c='b', linewidth=2)
plt.plot(w, c='g', linewidth=2)
plt.xlabel('Vector Index')
plt.ylabel('Turn Angle (degrees)')
plt.title('Total Turn Angle Vector for All Test Data', fontsize=12)
plt.legend(['Predictions', 'Particle Tracking (TrackMate)'])
plt.grid(True)
plt.tight_layout()
plt.savefig('angles_final.png')
plt.show()

a,b,c,d = betaprime.fit(q)
xtorch= np.linspace(betaprime.ppf(0.0001, a,b,c,d),
                betaprime.ppf(0.9999, a,b,c,d), target_length)
distr_torch=betaprime.pdf(xtorch, a,b,c,d)
plt.plot(xtorch,distr_torch,c='b')
a,b,c,d = betaprime.fit(w) 
xtorch= np.linspace(betaprime.ppf(0.0001, a,b,c,d),
                betaprime.ppf(0.9999, a,b,c,d), target_length)
distr_torch=betaprime.pdf(xtorch, a,b,c,d)
plt.plot(xtorch,distr_torch,c='g')
plt.title('Total Turn Angle Distribution for All Test Data')
plt.xlabel('Turn Angle (degrees)')
plt.ylabel('Probability Density')
plt.legend(['Prediction', 'Ground Truth'])
plt.xscale('log')
plt.yscale('log')
plt.savefig('angle_distribution_final.png')
plt.show()

#Vx
q = np.sort(np.hstack(final_out_vx)[::400])
print(q.shape)
w = np.sort(np.hstack(final_true_vx)[::400])
plt.plot(q, c='b', linewidth=2)
plt.plot(w, c='g', linewidth=2)
plt.xlabel('Vector Index')
plt.ylabel('Vx (pixels/frame)')
plt.title('Total Vx Vector for All Test Data', fontsize=12)
plt.legend(['Predictions', 'Particle Tracking (TrackMate)'])
plt.grid(True)
plt.tight_layout()
plt.savefig('Vx_final.png')
plt.show()

a,b,c,d = betaprime.fit(q)
xtorch= np.linspace(betaprime.ppf(0.0001, a,b,c,d),
                betaprime.ppf(0.9999, a,b,c,d), target_length)
distr_torch=betaprime.pdf(xtorch, a,b,c,d)
plt.plot(xtorch,distr_torch,c='b')
a,b,c,d = betaprime.fit(w) 
xtorch= np.linspace(betaprime.ppf(0.0001, a,b,c,d),
                betaprime.ppf(0.9999, a,b,c,d), target_length)
distr_torch=betaprime.pdf(xtorch, a,b,c,d)
plt.plot(xtorch,distr_torch,c='g')
plt.title('Total Vx Distribution for All Test Data')
plt.xlabel('Vx (pixels/frame)')
plt.ylabel('Probability Density')
plt.legend(['Prediction', 'Ground Truth'])
plt.xscale('log')
plt.yscale('log')
plt.savefig('Vx_distribution_final.png')
plt.show()

#Vy
q = np.sort(np.hstack(final_out_vy)[::400])
print(q.shape)
w = np.sort(np.hstack(final_true_vy)[::400])
plt.plot(q, c='b', linewidth=2)
plt.plot(w, c='g', linewidth=2)
plt.xlabel('Vector Index')
plt.ylabel('Vy (pixels/frame)')
plt.title('Total Vy Vector for All Test Data', fontsize=12)
plt.legend(['Predictions', 'Particle Tracking (TrackMate)'])
plt.grid(True)
plt.tight_layout()
plt.savefig('Vy_final.png')
plt.show()

a,b,c,d = betaprime.fit(q)
xtorch= np.linspace(betaprime.ppf(0.0001, a,b,c,d),
                betaprime.ppf(0.9999, a,b,c,d), target_length)
distr_torch=betaprime.pdf(xtorch, a,b,c,d)
plt.plot(xtorch,distr_torch,c='b')
a,b,c,d = betaprime.fit(w) 
xtorch= np.linspace(betaprime.ppf(0.0001, a,b,c,d),
                betaprime.ppf(0.9999, a,b,c,d), target_length)
distr_torch=betaprime.pdf(xtorch, a,b,c,d)
plt.plot(xtorch,distr_torch,c='g')
plt.title('Total Vy Distribution for All Test Data')
plt.xlabel('Vy (pixels/frame)')
plt.ylabel('Probability Density')
plt.legend(['Prediction', 'Ground Truth'])
plt.xscale('log')
plt.yscale('log')
plt.savefig('Vy_distribution_final.png')
plt.show()