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
class_num=4

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
# f_idx = 9
# print(filenames[f_idx])

directories = ['test_sims_homo/1000part_16xspeed/','test_sims_homo/1000part_4xspeed/',
'test_sims_homo/1000part/','test_sims_homo/2000part_16xspeed/',
'test_sims_homo/2000part_4xspeed/','test_sims_homo/2000part/',
'test_sims_homo/500part_16xspeed/','test_sims_homo/500part_4xspeed/',
'test_sims_homo/500part/',
'../newbgs/Acid4010FR_1ulh_20X_25fps_1x1y_5min_001/',
'../newbgs/Acid4010FR_5ulh_20X_25fps_1x1y_5min_001/',
'../newbgs/Acid4020FR_1ulh_20X_25fps_1x1y_5min_001/',
'../newbgs/Acid4020FR_5ulh_20X_25fps_1x1y_5min_002/','Acid_1ulh_pre/', 
'acid_noflow/open_20x_20fps_001_frames/', '../newbgs/Geo8020FR_1ulh_20X_25fps_1x1y_5min_002/',
'../newbgs/Geo8020FR_5ulh_20X_25fps_1x1y_5min_003/', 'Geo_1ulh_pre/', 
'Geo_5ulh_pre/', 'test_sims_het/1000part_16xspeed_heterogeneous/','test_sims_het/1000part_32xspeed_heterogeneous/',
'test_sims_het/1000part_4xspeed_heterogeneous/',
'test_sims_het/1000part_heterogeneous/','test_sims_het/2000part_16xspeed_heterogeneous/','test_sims_het/2000part_32xspeed_heterogeneous/',
'test_sims_het/2000part_4xspeed_heterogeneous/','test_sims_het/2000part_heterogeneous/',
'test_sims_het/500part_16xspeed_heterogeneous/','test_sims_het/500part_32xspeed_heterogeneous/','test_sims_het/500part_4xspeed_heterogeneous/',
'test_sims_het/500part_heterogeneous/', '../new_traj/Acid01xx_20X_25fps_1min_001_frames_bgs/',
'../new_traj/Acid01xx_20X_25fps_1min_002_frames_bgs/','../new_traj/Acid01xx_20X_25fps_1min_003_frames_bgs/',
'../new_traj/Acid01xx_20X_25fps_1min_004_frames_bgs/','../new_traj/Geo01xx_20X_25fps_1min_001_frames_bgs/',
'../new_traj/Geo01xx_20X_25fps_1min_002_frames_bgs/','../new_traj/Geo01xx_20X_25fps_1min_003_frames_bgs/',
'../new_traj/01xx_20X_25fps_5min_003_frames_bgs/','../new_traj/01xx_20X_25fps_5min_003_frames_bgs/',
'../newbgs/Paen8020FR_5ulh_20X_25fps_1x1y_5min_003/',
'paen_noflow/paen_8040_20x_20fps_006_frames/', 'Paen_1ulh_pre/', 
'Paen_5ulh_pre/', 'shew_noflow/8040_20x_002_frames/', 'shew_noflow/808_20x_20fps_005_frames/']

file_error=[]
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
    batch_size = 1
    test_dataloader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=0)

    if (images2.shape[0]-20)%30==0:
        images2 = images2[20:,:,:].reshape([(images2.shape[0]-20)//30,30,448,448])
    elif (images2.shape[0]-10)%30==0:
        images2 = images2[10:,:,:].reshape([(images2.shape[0]-10)//30,30,448,448])
    else:
        images2 = images2.reshape([(images2.shape[0])//30,30,448,448])

    images2 = torch.tensor(images2)

    dataset_test = TestDataset2(images2)#  xtorch_test, distr_torch_test, scales_test2)
    batch_size = 1
    test_dataloader2 = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=0)

    x = next(iter(test_dataloader))
    print(x.shape)
    plt.imshow(x[0,29,:,:])


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

    model8 = timm.create_model('volo_d3_448', in_chans=30,drop_path_rate=.0, num_classes=5,pretrained=False)
    model8.load_state_dict(torch.load('models/speed_classifier_5class_volod3_448px_disp_lbm_5_07_v2'))
    model8.to(device).eval()

    model9 = timm.create_model('volo_d1_384', in_chans=30,drop_path_rate=.0, num_classes=5,pretrained=False)
    model9.load_state_dict(torch.load('models/speed_classifier_5class_volod1_384px_disp_all_5_06'))
    model9.to(device).eval()

    model10 = timm.create_model('volo_d1_384', in_chans=30,drop_path_rate=.0, num_classes=5,pretrained=False)
    model10.load_state_dict(torch.load('models/speed_classifier_5class_volod1_384px_disp_all_5_06_v2'))
    model10.to(device).eval()

    model11 = timm.create_model('volo_d1_384', in_chans=30,drop_path_rate=.0, num_classes=5,pretrained=False)
    model11.load_state_dict(torch.load('models/speed_classifier_5class_volod1_384px_disp_all_5_06_v3'))
    model11.to(device).eval()

    model12 = timm.create_model('volo_d1_384', in_chans=30,drop_path_rate=.0, num_classes=5,pretrained=False)
    model12.load_state_dict(torch.load('models/speed_classifier_5class_volod1_384px_disp_all_5_06_v4'))
    model12.to(device).eval()

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


            out4 =F.softmax((model4((x).to(device)).detach().cpu()),1).numpy()
            out5 =F.softmax((model5((x).to(device)).detach().cpu()),1).numpy()
            out6 =F.softmax((model6((x).to(device)).detach().cpu()),1).numpy() # class 4, close to 1
            out7 =F.softmax((model7((1-x).to(device)).detach().cpu()),1).numpy()
            out8 =F.softmax((model8((x).to(device)).detach().cpu()),1).numpy()
            out9 =F.softmax((model9((x2).to(device)).detach().cpu()),1).numpy()
            out10 =F.softmax((model10((x2).to(device)).detach().cpu()),1).numpy()
            out11 =F.softmax((model11((x2).to(device)).detach().cpu()),1).numpy()
            out12 =F.softmax((model12((x2).to(device)).detach().cpu()),1).numpy()
            out13 =F.softmax((model13((x2).to(device)).detach().cpu()),1).numpy()
            out14 =F.softmax((model15((x2).to(device)).detach().cpu()),1).numpy() #class 2 close to cslaa 3
            out15 =F.softmax((model15((x).to(device)).detach().cpu()),1).numpy() # class 4
            out16 =F.softmax((model16((x).to(device)).detach().cpu()),1).numpy() #class 2 .25
            out17 =F.softmax((model17((x).to(device)).detach().cpu()),1).numpy() # class 2
            out18 =F.softmax((model18((x).to(device)).detach().cpu()),1).numpy() # class 2
            out19 =F.softmax((model19((x).to(device)).detach().cpu()),1).numpy() # class 3
            out20 =F.softmax((model20((x).to(device)).detach().cpu()),1).numpy() # class 2, .22


            out=(out11*5+out14*.5+out16*15.5+out17*18.6+out19*3+out5*3.5+out10*2.4+out7*.6+out13*.3+out6*2+out18*.5)/51.9#(out16*4+out17*2+out14*.5)/6.5#(out0*2+out3*40+out5*2+out7*2+out10+out13)/48
            out[:,1] = out[:,1]+.035
            out[:,2] = out[:,2]+.11
            out[:,3] = out[:,3]+.035
            out[:,4] = out[:,4]-.025
            #out = (out4*10+out6*10+out7*1+out8*3+out9*15+out10*1+out11*2+out12*1+out13*1)/44

            # #x1 = torchvision.transforms.functional.resize(x, 448)
            # x2 = tv.transforms.functional.resize(x, 384)
            # #x3 = torchvision.transforms.functional.resize(x, 224)


            # out4 =F.softmax((model4((x).to(device)).detach().cpu()),1).numpy()
            # out5 =F.softmax((model5((x).to(device)).detach().cpu()),1).numpy()
            # out6 =F.softmax((model6((x2).to(device)).detach().cpu()),1).numpy()
            # out7 =F.softmax((model7((x2).to(device)).detach().cpu()),1).numpy()
            # out8 =F.softmax((model8((x2).to(device)).detach().cpu()),1).numpy()
            # out9 =F.softmax((model9((x2).to(device)).detach().cpu()),1).numpy()
            # out10 =F.softmax((model10((x2).to(device)).detach().cpu()),1).numpy()
            # out11 =F.softmax((model11((x2).to(device)).detach().cpu()),1).numpy()

            # #out=np.mean(np.vstack((out0+out1*3+out2+out3)/6),0)
            # out = (out4*1+out5*2+out6*2+out7*6+out8+out9)/13
            
            # Add a classifier/use Trackpy to reduce the weights of late time predictions (how many particles in the image?)
            outputs1.append(out)
            count=count+1

    # print(np.mean(np.vstack(out0),0))
    # print(np.mean(np.vstack(out1),0))
    # print(np.mean(np.vstack(out2),0))
    # print(np.mean(np.vstack(out3),0))
    print(np.mean(np.vstack(out4),0))
    print(np.mean(np.vstack(out5),0))
    print(np.mean(np.vstack(out6),0))
    print(np.mean(np.vstack(out7),0))
    print(np.mean(np.vstack(out8),0))
    print(np.mean(np.vstack(out9),0))
    print(np.mean(np.vstack(out10),0))
    print(np.mean(np.vstack(out11),0))
    print(np.mean(np.vstack(out12),0))
    print(np.mean(np.vstack(out13),0))
    print(np.mean(np.vstack(out14),0))
    print(np.mean(np.vstack(out15),0))
    print(np.mean(np.vstack(out16),0))
    print(np.mean(np.vstack(out17),0))
    print(np.mean(np.vstack(out18),0))
    print(np.mean(np.vstack(out19),0))
    print(np.mean(np.vstack(out20),0))


    mean_out = np.mean(np.vstack(outputs1),0)
    class_num = np.argmax(mean_out)
    print('Class Number: ',class_num)
    print(mean_out)
    # #class_num=1


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

    model5 = timm.create_model('volo_d3_448', in_chans=40, drop_path_rate=.0,num_classes=1000,pretrained=False)
    model5.load_state_dict(torch.load('models/speed_model_volod3_448px_dispBrown_4_22_v2'))
    model5.to(device).eval()

    model1 = timm.create_model('volo_d3_448', in_chans=40, drop_path_rate=.0,num_classes=1000,pretrained=False)
    model1.load_state_dict(torch.load('models/speed_model_volod3_448px_dispBrown_4_22'))
    model1.to(device).eval()

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

    # #model16 = timm.create_model('botnet26t_256', in_chans=40, drop_path_rate=.0,num_classes=500,pretrained=False)
    # #model16.load_state_dict(torch.load('models/speed_model_ByobNet26_256px_dispBrown_4_23'))
    # #model16.to(device).eval()

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
    
    model25 = timm.create_model('volo_d3_448', in_chans=30, drop_path_rate=.0,num_classes=500,pretrained=False)
    model25.load_state_dict(torch.load('models/speed_model_volod3_448px_disp_lbm_5_07'))
    model25.to(device).eval()

    model26 = timm.create_model('volo_d3_448', in_chans=30, drop_path_rate=.0,num_classes=500,pretrained=False)
    model26.load_state_dict(torch.load('models/speed_model_volod3_448px_disp_lbm_5_07_v2'))
    model26.to(device).eval()
    
    model27 = Patch_model5()
    model27.load_state_dict(torch.load('models/speed_model_patch_448px_disp_lbm_5_07_v4'))
    model27.to(device).eval()
    
    # model28 = Patch_model5()
    # model28.load_state_dict(torch.load('models/speed_model_patch_448px_disp_lbm_5_07_v3'))
    # model28.to(device).eval()
    
    # model29 = Patch_model5()
    # model29.load_state_dict(torch.load('models/speed_model_patch_448px_disp_lbm_5_07_v2'))
    # model29.to(device).eval()
    
    # model30 = Patch_model5()
    # model30.load_state_dict(torch.load('models/speed_model_patch_448px_disp_lbm_5_07'))
    # model30.to(device).eval()
    

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
        

    max24 = 130.0
    max21 = 216.7189

    with torch.no_grad():
        for x in test_dataloader:
            if torch.mean(x)<.5:
                x = 1-x
            # x1 = tv.transforms.functional.resize(x, 448)
            x2 = tv.transforms.functional.resize(x, 224)
            x3 = tv.transforms.functional.resize(x, 384)
            x4 = tv.transforms.functional.resize(x, 256)
            #out1_1000 = (model1((1-x1).to(device)).detach().cpu().numpy())
            #out1 = np.zeros([batch_size,500])
            #for i in range(len(out1)):
            #    out1[i,:] = interpolate_vectors(out1_1000[i,:],500)
            out2 = np.sort(model2((x).to(device)).detach().cpu().numpy())
            out11 = np.sort(model11((x3).to(device)).detach().cpu().numpy())
            #out10 =(model10((x3).to(device)).detach().cpu().numpy())
            out9 = np.sort(model9((x3).to(device)).detach().cpu().numpy())
            out8 = np.sort(model8((x).to(device)).detach().cpu().numpy())
            out7 =np.sort(model7((x).to(device)).detach().cpu().numpy())
            out6 = np.sort(model6((1-x).to(device)).detach().cpu().numpy())
            out3 = np.sort(model3((1-x2).to(device)).detach().cpu().numpy())
            out4 = np.sort(model4((x2).to(device)).detach().cpu().numpy())
            #out5 = np.sort(model5((x1).to(device)).detach().cpu().numpy())
            out12 = np.sort(model12((x3).to(device)).detach().cpu().numpy())
            #out0 = (model((1-x1).to(device)).detach().cpu().numpy())
            out13_1000 = np.sort(model13((x).to(device)).detach().cpu().numpy())
            out13 = np.zeros([batch_size,500])
            for i in range(len(out13)-1):
                 out13[i,:] = interpolate_vectors(out13_1000[i,:],500)
            out14 = np.sort(model14((x).to(device)).detach().cpu().numpy())
            out15 = np.sort(model15((1-x4).to(device)).detach().cpu().numpy())
            # #out16 = (model16((1-x4).to(device)).detach().cpu().numpy())
            out17 = np.sort(model17((1-x).to(device)).detach().cpu().numpy())
            out18 = np.sort(model18((x3).to(device)).detach().cpu().numpy())
            out19 = np.sort(model19((1-x3).to(device)).detach().cpu().numpy())
            out20 = np.sort(model20((1-x).to(device)).detach().cpu().numpy())
            out21 = np.sort(model21((x3[:,5:,:,:]).to(device)).detach().cpu().numpy())*max21
            out22 = np.sort(model22((1-x3[:,5:35,:,:]).to(device)).detach().cpu().numpy())
            out23 = np.sort(model23((1-x2[:,5:35,:,:]).to(device)).detach().cpu().numpy())
            out25 = np.sort(model25((1-x[:,5:35,:,:]).to(device)).detach().cpu().numpy())
            out26 = np.sort(model26((x[:,5:35,:,:]).to(device)).detach().cpu().numpy())
            out26_2 = np.sort(model26((1-x[:,5:35,:,:]).to(device)).detach().cpu().numpy())
            out27 = np.sort(model27((1-x[:,5:35,:,:]).to(device)).detach().cpu().numpy())
            #out28 = np.sort(model28((1-x[:,5:35,:,:]).to(device)).detach().cpu().numpy())
            # out29 = np.sort(model29((1-x[:,5:35,:,:]).to(device)).detach().cpu().numpy())
            #out30 = np.sort(model30((1-x[:,5:35,:,:]).to(device)).detach().cpu().numpy())

            out24 = np.sort(model24((1-x3).to(device)).detach().cpu().numpy())*max24
            

            out14_2 = np.sort(model14((1-x).to(device)).detach().cpu().numpy())
            out15_2 = np.sort(model15((x4).to(device)).detach().cpu().numpy())
            #out16 = (model16((1-x4).to(device)).detach().cpu().numpy())
            out17_2 = np.sort(model17((x).to(device)).detach().cpu().numpy())
            out18_2 = np.sort(model18((1-x3).to(device)).detach().cpu().numpy())
            out19_2 = np.sort(model19((x3).to(device)).detach().cpu().numpy())
            out20_2 = np.sort(model20((x).to(device)).detach().cpu().numpy())

            out = (out6*8+out7*2+out15*4+out20+out25*4+out27)/19
            if class_num==4:
                #1ulh
                #class 3 (x<3)
                out=(out*160+out4*6+(out14)*15+out14_2*60+out17*50+out18*15+(out17_2)*20+(out19_2)*25+(out20)*10+(out8)*1.5+(out9)*.5+out26_2*2)/365
                out = out*.6
            elif class_num==3:
                #class 2 (3<x<6)
                out=(out*200+out2*10+out6*4+out11*5+out14*4+out15*2+out17*10+out18*8+out19*26+out20*2)/271
            elif class_num==2:
                #5ulh
                #class 1 (6<x<10)
                out = (out*230+out9*70+out11*30+out27*10+out24*20+out26*80+out19*10+out18*10+out13*20)/480
                #out=(out*50+out3*8+out2*5+out8*15+out11*33+out13*50+out14*5+out18*17+out19*17+out6*5+out2*5+out27*5+out9*40+out11*20)/270
                out = out*1.55
            elif class_num==1:
                #5ulh
                #class 1 (6<x<10)
                out = (out*250+out9*120+out11*30+out27*10+out24*20+out26*120+out19*10+out18*10+out13*20)/590
                #out=(out*50+out3*8+out2*5+out8*15+out11*33+out13*50+out14*5+out18*17+out19*17+out6*5+out2*5+out27*5+out9*40+out11*20)/270
                out = out*1.8
            else:
                #class 0 (x>10)
                out = (out*200+out9*70+out11*30+out27*10+out24*20+out26*80)/410
                out = out*1.3
            # outputs.append(out)
            
            
            #out = (out6*8+out7*2+out15*4+out25*4)/18
            outputs.append(out)
            
            
            # x=1-x
            # x1 = torchvision.transforms.functional.resize(x, 448)
            # x2 = torchvision.transforms.functional.resize(x, 224)
            # x3 = torchvision.transforms.functional.resize(x, 384)
            # x4 = torchvision.transforms.functional.resize(x, 256)
            # #out1_1000 = (model1((1-x1).to(device)).detach().cpu().numpy())
            # #out1 = np.zeros([batch_size,500])
            # #for i in range(len(out1)):
            # #    out1[i,:] = interpolate_vectors(out1_1000[i,:],500)
            # out2 = np.sort(model2((x).to(device)).detach().cpu().numpy())
            # out11 = np.sort(model11((x3).to(device)).detach().cpu().numpy())
            # #out10 =(model10((x3).to(device)).detach().cpu().numpy())
            # out9 = np.sort(model9((x3).to(device)).detach().cpu().numpy())
            # out8 = np.sort(model8((x1).to(device)).detach().cpu().numpy())
            # out7 =np.sort(model7((x1).to(device)).detach().cpu().numpy())
            # out6 = np.sort(model6((1-x1).to(device)).detach().cpu().numpy())
            # out3 = np.sort(model3((1-x2).to(device)).detach().cpu().numpy())
            # out4 = np.sort(model4((x2).to(device)).detach().cpu().numpy())
            # #out5 = np.sort(model5((x1).to(device)).detach().cpu().numpy())
            # out12 = np.sort(model12((x3).to(device)).detach().cpu().numpy())
            # #out0 = (model((1-x1).to(device)).detach().cpu().numpy())
            # out13_1000 = np.sort(model13((x1).to(device)).detach().cpu().numpy())
            # out13 = np.zeros([batch_size,500])
            # for i in range(len(out13)):
                # out13[i,:] = interpolate_vectors(out13_1000[i,:],500)
            # out14 = np.sort(model14((x1).to(device)).detach().cpu().numpy())
            # out15 = np.sort(model15((1-x4).to(device)).detach().cpu().numpy())
            # #out16 = (model16((1-x4).to(device)).detach().cpu().numpy())
            # out17 = np.sort(model17((1-x1).to(device)).detach().cpu().numpy())
            # out18 = np.sort(model18((x3).to(device)).detach().cpu().numpy())
            # out19 = np.sort(model19((1-x3).to(device)).detach().cpu().numpy())
            # out20 = np.sort(model20((1-x1).to(device)).detach().cpu().numpy())
            # out21 = np.sort(model21((1-x3[:,5:,:,:]).to(device)).detach().cpu().numpy())*max21
            # out22 = np.sort(model22((1-x3[:,5:35,:,:]).to(device)).detach().cpu().numpy())
            # out23 = np.sort(model23((1-x2[:,5:35,:,:]).to(device)).detach().cpu().numpy())
            # out24 = np.sort(model24((1-x3).to(device)).detach().cpu().numpy())*max24

            # out14_2 = np.sort(model14((1-x1).to(device)).detach().cpu().numpy())
            # out15_2 = np.sort(model15((x4).to(device)).detach().cpu().numpy())
            # #out16 = (model16((1-x4).to(device)).detach().cpu().numpy())
            # out17_2 = np.sort(model17((x1).to(device)).detach().cpu().numpy())
            # out18_2 = np.sort(model18((1-x3).to(device)).detach().cpu().numpy())
            # out19_2 = np.sort(model19((x3).to(device)).detach().cpu().numpy())
            # out20_2 = np.sort(model20((x1).to(device)).detach().cpu().numpy())

            
            # if class_num==4:
                # #1ulh
                # #class 3 (x<3)
                # out=(out4*6+(out14)*15+out17*25+out18*15+(out17_2)*20+(out18_2)*2+(out19_2)*25+(out20)*10+(out8)*2.5+(out9)*13.5+out23*2+out22*1)/148
            # elif class_num==3:
                # #class 2 (3<x<6)
                # out=(out2*10+out3*4+out4*16+out6*4+out7*1+out11*10+out12*5+out13*4+out14*1+out15*2+out17*10+out18*8+out19*26+out20*2+out9*3+out21*3+out22*2+out23*4+out24*2)/106
            # elif class_num==2:
                # #5ulh
                # #class 1 (6<x<10)
                # out=(out3*8+out4*32+out8*15+out11*33+out13*50+out14*5+out18*17+out19*17+out6*5+out2*5+out21*3+out22*2+out23*4+out24*2)/166
                
            # else:
                # #class 0 (x>10)
                # # out=(out3*20+out4*128+out8*20+out11*50+out13*70+out14*5+out18*17+out19*17+out6*5+out2*20+out21*3+out22*2+out23*2+out24*2)/320

                # out=(out3*8+out4*64+out8*15+out11*33+out13*50+out14*5+out18*17+out19*17+out6*5+out2*10+out21*3+out22*2+out23*4+out24*2)/190

            # outputs.append(out)
            #outputs2.append(out2)

    stacked_out = np.vstack(outputs)
    out = np.reshape(stacked_out,[stacked_out.shape[0]*stacked_out.shape[1], ])

    directory='../tmate_trajectories/'
    filename = os.listdir(directory)[f_idx]
    print(filename)
    df = pd.read_csv(os.path.join(directory,filename),header=0, skiprows=[1,2,3])
    tracks = df.TRACK_ID.unique()
    vels=1
    for i in range(len(tracks)//2):
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
    speed_loss = criterion(torch.tensor(np.sort(interpolate_vectors(np.sort(out[out>0]), len(vels)))), torch.tensor(np.sort(vels)))
    mean_loss =  np.abs(np.mean(vels)-np.mean(out[out>0]))
    print('Mean Speed from PT: ', np.mean(vels))
    print('Speed Loss: ', speed_loss)
    print('Absolute Error of Average Speed: ',mean_loss)
    file_error.append(mean_loss)
print('Average Speed Error for all Files: ',np.mean(file_error))