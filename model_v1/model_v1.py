# -*- encoding: utf-8 -*-
'''
/* ***************************************************************************************************
*   NOTICE
*   This software is the property of Glint Co.,Ltd.. Any information contained in this
*   doc should not be reproduced, or used, or disclosed without the written authorization from
*   Glint Co.,Ltd..
***************************************************************************************************
*   File Name       : model_v1.py
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
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import timm
print('timm:',timm.__version__)

class MyDecoderBlock3d(nn.Module):
    def __init__(
            self,
            in_channel,
            skip_channel,
            out_channel,
    ):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channel + skip_channel, out_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channel),
            nn.ReLU(inplace=True),
        )
        self.attention1 = nn.Identity()
        self.conv2 = nn.Sequential(
            nn.Conv3d(out_channel, out_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channel),
            nn.ReLU(inplace=True),
        )
        self.attention2 = nn.Identity()

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=(1,2,2), mode='nearest')
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x
    
class MyUnetDecoder3d(nn.Module):
    def __init__(
            self,
            in_channel,
            skip_channel,
            out_channel,
    ):
        super().__init__()
        self.center = nn.Identity()

        i_channel = [in_channel, ] + out_channel[:-1]
        s_channel = skip_channel
        o_channel = out_channel
        block = [
            MyDecoderBlock3d(i, s, o)
            for i, s, o in zip(i_channel, s_channel, o_channel)
        ]
        self.block = nn.ModuleList(block)

    def forward(self, feature, skip):
        d = self.center(feature)
        decode = []
        for i, block in enumerate(self.block):  
            s = skip[i]
            d = block(d, s)
            decode.append(d)
        last = d
        return last, decode

    
# encoder helper
def pvtv2_encode(x, e):
    encode = []
    x = e.patch_embed(x)
    for stage in e.stages:
        x = stage(x); encode.append(x)
    return encode

# # 2.  prediction head

#modeling: prediction head

# magic.1.: convert local pixelwise prediction to global volume prediction

def heatmap_to_coord(heatmap):
    num_image = len(heatmap)
    device = heatmap[0].device
    _,_, H, W = heatmap[0].shape
    D = max([h.shape[1] for h in heatmap])

    # create coordinates grid.
    x = torch.linspace(0, W - 1, W, device=device)
    y = torch.linspace(0, H - 1, H, device=device)
    z = torch.linspace(0, D - 1, D, device=device)

    point_xy=[]
    point_z =[]
    for i in range(num_image):
        num_point, D, H, W = heatmap[i].shape
        pos_x = x.reshape(1,1,1,W)
        pos_y = y.reshape(1,1,H,1)
        pos_z = z[:D].reshape(1,D,1,1)

        py = torch.sum(pos_y * heatmap[i], dim=(1,2,3))
        px = torch.sum(pos_x * heatmap[i], dim=(1,2,3))
        pz = torch.sum(pos_z * heatmap[i], dim=(1,2,3))

        point_xy.append(torch.stack([px,py]).T)
        point_z.append(pz)

    xy = torch.stack(point_xy)
    z = torch.stack(point_z)
    return xy, z

def heatmap_to_grade(heatmap, grade_mask):
    num_image = len(heatmap)
    grade = []
    for i in range(num_image):
        num_point, D, H, W = heatmap[i].shape
        C, D, H, W = grade_mask[i].shape
        g = grade_mask[i].reshape(1,C,D,H,W)#.detach()
        h = heatmap[i].reshape(num_point,1,D,H,W)#.detach()
        g = (h*g).sum(dim=(2,3,4))
        grade.append(g)
    grade = torch.stack(grade)
    return grade

# # 3.  loss function
# - per pixel loss
# - per volume loss

#modeling: loss functions

# magic.2.: example shown here is for sagittal t1 neural foraminal narrowing points
# dynamic matching - becuase of ambiguous/confusion of ground truth labeling of L5 and S1,
# your predicted xy coordinates may be misaligned with ground truth xy. Hence you must 
# modified the grade target values as well in loss backpropagation

def do_dynamic_match_truth(xy, truth_xy, threshold=3):

    num_image, num_point, _2_ = xy.shape
    t = truth_xy[:, :5, 1].reshape(num_image, 5, 1)
    p = xy[:, :5, 1].reshape(num_image, 1, 5)
    diff = torch.abs(p - t)
    left, left_i = diff.min(-1)
    left_t = (left < threshold)
    
    t = truth_xy[:, 5:, 1].reshape(num_image, 5, 1)
    p = xy[:, 5:, 1].reshape(num_image, 1, 5)
    diff = torch.abs(p - t)
    right, right_i = diff.min(-1)
    right_t = (right < threshold)

    index = torch.cat([left_i,right_i+5],1).detach()
    valid = torch.cat([left_t,right_t],1).detach()
    return index, valid



def F_grade_loss(grade, truth):
    eps = 1e-5
    weight = torch.FloatTensor([1,2,4]).to(grade.device)

    t = truth.reshape(-1)
    g = grade.reshape(-1,3)

    #loss = F.nll_loss( torch.clamp(g, eps, 1-eps).log(), t,weight=weight, ignore_index=-1)
    loss = F.cross_entropy(g, t,weight=weight, ignore_index=-1)
    return loss

 
def F_zxy_loss(z, xy,  z_truth, xy_truth):
    m = z_truth!=-1
    z_truth = z_truth.float()
    loss = (
        F.mse_loss(z[m], z_truth[m]) + F.mse_loss(xy[m], xy_truth[m])
    )
    return loss


#https://discuss.pytorch.org/t/jensen-shannon-divergence/2626/11
#Jensen-Shannon divergence
def F_xyz_mask_loss(heatmap, truth, D):
    heatmap =  torch.split_with_sizes(heatmap, D, 0)
    truth =  torch.split_with_sizes(truth, D, 0)
    num_image = len(heatmap)

    loss =0
    for i in range(num_image):
        p,q = truth[i], heatmap[i]
        D,num_point,H,W = p.shape

        eps = 1e-8
        p = torch.clamp(p.transpose(1,0).flatten(1),eps,1-eps)
        q = torch.clamp(q.transpose(1,0).flatten(1),eps,1-eps)
        m = (0.5 * (p + q)).log()

        kl = lambda x,t: F.kl_div(x,t, reduction='batchmean', log_target=True)
        loss += 0.5 * (kl(m, p.log()) + kl(m, q.log()))
    loss = loss/num_image
    return loss


class Net(nn.Module):
    def __init__(self, pretrained=False, cfg=None):
        super(Net, self).__init__()
        self.output_type = ['infer', 'loss']
        self.register_buffer('D', torch.tensor(0))
        self.register_buffer('mean', torch.tensor(0.5))
        self.register_buffer('std', torch.tensor(0.5))

        arch = 'pvt_v2_b4'

        encoder_dim = {
            'pvt_v2_b2': [64, 128, 320, 512],
            'pvt_v2_b4': [64, 128, 320, 512],
        }.get(arch, [768])

        decoder_dim = \
              [384, 192, 96]
              #[256, 128, 64]

        self.encoder = timm.create_model(
            model_name=arch, pretrained=pretrained, in_chans=3, num_classes=0, global_pool=''
        )
        self.decoder = MyUnetDecoder3d(
            in_channel=encoder_dim[-1],
            skip_channel=encoder_dim[:-1][::-1],
            out_channel=decoder_dim,
        )


        self.zxy_mask = nn.Conv3d(decoder_dim[-1], 10, kernel_size=1)
        self.grade_mask = nn.Conv3d(decoder_dim[-1], 128, kernel_size=1)
        self.grade = nn.Sequential(
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 3),
        )


    def forward(self, batch):
        device = self.D.device
        image = batch['image'].to(device)
        D = batch['D']
        num_image = len(D)

        B, H, W = image.shape
        image = image.reshape(B, 1, H, W)

        x = image.float() / 255
        x = (x - self.mean) / self.std
        x = x.expand(-1, 3, -1, -1)

        #---
        encode = pvtv2_encode(x, self.encoder)
        ##[print(f'encode_{i}', e.shape) for i,e in enumerate(encode)]
        encode = [ torch.split_with_sizes(e, D, 0) for e in encode ]

        grade_mask = []  #feature map
        zxy_mask   = []  #prob heatmap
        for i in range(num_image):
            e = [ encode[s][i].transpose(1,0).unsqueeze(0) for s in range(4) ]
            l, _ = self.decoder(
                feature=e[-1], skip=e[:-1][::-1]
            )

            g = self.grade_mask(l).squeeze(0)
            grade_mask.append(g)

            zxy = self.zxy_mask(l).squeeze(0)
            _,d,h,w = zxy.shape
            zxy = zxy.flatten(1).softmax(-1).reshape(-1,d,h,w)
            zxy_mask.append(zxy)

        ##print(D)
        ##[print(f'zxy_logit_{i}', x.shape) for i, x in enumerate(zxy_mask_prob)]

        xy, z = heatmap_to_coord(zxy_mask)
        ##print('xy', xy.shape, 'z', z.shape)

        #---
        num_point = xy.shape[1]
        grade = heatmap_to_grade(zxy_mask, grade_mask)
        #print('grade', grade.shape)
        grade = grade.reshape(num_image*num_point,-1)
        grade = self.grade(grade)
        grade = grade.reshape(num_image,num_point,3)
        ##print('grade', grade.shape)

        #---
        zxy_mask = torch.cat(zxy_mask, 1).transpose(1, 0)

        output = {}
        if 'loss' in self.output_type:
            output['zxy_mask_loss'] = F_xyz_mask_loss(zxy_mask, batch['zxy_mask'].to(device), D)
            output['zxy_loss'] = F_zxy_loss(z, xy, batch['z'].to(device), batch['xy'].to(device))
            #output['grade_loss'] = F_grade_loss(grade,  batch['grade'].to(device))
            if 1:
                index, valid = do_dynamic_match_truth(xy, batch['xy'].to(device))
                truth = batch['grade'].to(device)
                truth_matched = []
                for i in range(num_image):
                    truth_matched.append(truth[i][index[i]])
                truth_matched = torch.stack(truth_matched)
                output['grade_loss'] = F_grade_loss(grade[valid],  truth_matched[valid])

        if 'infer' in self.output_type:
            output['zxy_mask'] = zxy_mask
            output['xy'] = xy
            output['z'] = z
            output['grade'] = F.softmax(grade,-1)

        return output


