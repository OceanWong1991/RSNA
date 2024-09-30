# -*- encoding: utf-8 -*-
'''
/* ***************************************************************************************************
*   NOTICE
*   This software is the property of Glint Co.,Ltd.. Any information contained in this
*   doc should not be reproduced, or used, or disclosed without the written authorization from
*   Glint Co.,Ltd..
***************************************************************************************************
*   File Name       : demo_v1.py
***************************************************************************************************
*    Module Name        : 
*    Prefix            : 
*    ECU Dependence    : None
*    MCU Dependence    : None
*    Mod Dependence    : None
***************************************************************************************************
*    Description        : 
*
***************************************************************************************************
*    Limitations        :
*
***************************************************************************************************
*
***************************************************************************************************
*    Revision History:
*
*    Version        Date            Initials        CR#                Descriptions
*    ---------    ----------        ------------    ----------        ---------------
*     1.0.0       2024-09-27            Neo                         
****************************************************************************************************/
'''
import natsort  
import torch
import numpy as np
import cv2
import numpy as np
import pydicom
import glob
from natsort import natsorted
import pandas as pd
from model_v1 import Net

class dotdict(dict):
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

#demo with dummy data
def run_check_net():
    D = [6, 7, 9, 11, 3, 4, 5] #input volumes with variable depth
    num_image  = len(D)
    
    image_size = 320
    mask_size  = image_size//4
    B = sum(D)
    num_point = 10

    batch = {
        'D': D,
        'image': torch.from_numpy( np.random.uniform(-1, 1, ( B, image_size, image_size))).byte(),
        'zxy_mask': torch.from_numpy(np.random.uniform(0,1,(B, num_point, mask_size, mask_size))).float(),
        'z': torch.from_numpy(np.random.choice(min(D), (num_image, num_point))).long(),
        'xy': torch.from_numpy(np.random.choice(image_size, (num_image, num_point, 2))).float(),
        'grade': torch.from_numpy(np.random.choice(3, (num_image, num_point))).long(), 
    }

    net = Net(pretrained=False, cfg=None)#.cuda()
    
    with torch.no_grad():
        with torch.amp.autocast('cuda', enabled=True):
            output = net(batch)
    # ---
    print('batch')
    for k, v in batch.items():
        if k == 'D':
            print(f'{k:>32} : {v} ')
        else:
            print(f'{k:>32} : {v.shape} ')

    print('output')
    for k, v in output.items():
        if 'loss' not in k:
            print(f'{k:>32} : {v.shape} ')
    print('loss')
    for k, v in output.items():
        if 'loss' in k:
            print(f'{k:>32} : {v.item()} ')

#demo with real kaggle data and visualisation
DATA_KAGGLE_DIR = '/home/ai/neo/data/rsna-2024-lumbar-spine-degenerative-classification'

def np_dot(a,b):
    return np.sum(a * b, 1)

def normalise_to_8bit(x, lower=0.1, upper=99.9): 
    lower, upper = np.percentile(x, (lower, upper))
    x = np.clip(x, lower, upper)
    x = x - np.min(x)
    x = x / np.max(x)
    return (x * 255).astype(np.uint8)

def read_series(study_id,series_id,series_description):
    error_code = ''
    
    data_kaggle_dir = DATA_KAGGLE_DIR
    dicom_dir = f'{data_kaggle_dir}/train_images/{study_id}/{series_id}'

    # read dicom file
    dicom_file = natsorted(glob.glob(f'{dicom_dir}/*.dcm'))
    instance_number = [int(f.split('/')[-1].split('.')[0]) for f in dicom_file]
    dicom = [pydicom.dcmread(f) for f in dicom_file]

    # make dicom header df
    dicom_df = []
    for i, d in zip(instance_number, dicom):  # d__.dict__
        dicom_df.append(
            dotdict(
                study_id=study_id,
                series_id=series_id,
                series_description=series_description,
                instance_number=i,
                # InstanceNumber = d.InstanceNumber,
                ImagePositionPatient=[float(v) for v in d.ImagePositionPatient],
                ImageOrientationPatient=[float(v) for v in d.ImageOrientationPatient],
                PixelSpacing=[float(v) for v in d.PixelSpacing],
                SpacingBetweenSlices=float(d.SpacingBetweenSlices),
                SliceThickness=float(d.SliceThickness),
                grouping=str([round(float(v), 3) for v in d.ImageOrientationPatient]),
                H=d.pixel_array.shape[0],
                W=d.pixel_array.shape[1],
            )
        )
    dicom_df = pd.DataFrame(dicom_df)
    # dicom_df.to_csv('dicom_df.csv',index=False)
    # exit(0)

    #----
    if ((dicom_df.W.nunique()!=1) or (dicom_df.H.nunique()!=1)):
        error_code = '[multi-shape]'
    Wmax = dicom_df.W.max()
    Hmax = dicom_df.H.max()

    # sort slices
    dicom_df = [d for _, d in dicom_df.groupby('grouping')]

    data = []
    sort_data_by_group = []
    for df in dicom_df:
        position = np.array(df['ImagePositionPatient'].values.tolist())
        orientation = np.array(df['ImageOrientationPatient'].values.tolist())
        normal = np.cross(orientation[:, :3], orientation[:, 3:])
        projection = np_dot(normal, position)
        df.loc[:, 'projection'] = projection
        df = df.sort_values('projection')


        # todo: assert all slices are continous ??
        # use  (position[-1]-position[0])/N = SpacingBetweenSlices ??
        assert len(df.SliceThickness.unique()) == 1
        #assert len(df.SpacingBetweenSlices.unique()) == 1

        volume = []
        for i in df.instance_number:
            v = dicom[instance_number.index(i)].pixel_array
            if error_code.find('multi-shape')!=-1:
                H,W = v.shape
                v=np.pad(v,[(0,Hmax-H),(0,Wmax-W)],'reflect')
            volume.append(v)

        volume = np.stack(volume)
        volume = normalise_to_8bit(volume)

        data.append(dotdict(
            df=df,
            volume=volume,
        ))

        if 'sagittal' in series_description.lower():
            sort_data_by_group.append(position[0, 0])  # x
        if 'axial' in series_description.lower():
            sort_data_by_group.append(position[0, 2])  # z

    data = [r for _, r in sorted(zip(sort_data_by_group, data))]
    for i, r in enumerate(data):
        r.df.loc[:, 'group'] = i

    df = pd.concat([r.df for r in data])
    df.loc[:, 'z'] = np.arange(len(df))
    volume = np.concatenate([r.volume for r in data])
    return volume, df, error_code

def do_resize_and_center(image, reference_size):
    H, W = image.shape[:2]
    if (W==reference_size) & (H==reference_size):
        return image, (1,0,0)

    s = reference_size / max(H, W)
    m = cv2.resize(image, dsize=None, fx=s, fy=s)
    h, w = m.shape[:2]
    padx0 = (reference_size-w)//2
    padx1 = reference_size-w-padx0
    pady0 = (reference_size-h)//2
    pady1 = reference_size-h-pady0

    m = np.pad(m, [[pady0, pady1], [padx0, padx1], [0, 0]], mode='constant', constant_values=0)
    #p = point * s +[[padx0,pady0]]
    scale_param = s,padx0,pady0
    return m, scale_param

if __name__ == "__main__":
    run_check_net()