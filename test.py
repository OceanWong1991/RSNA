# -*- encoding: utf-8 -*-
'''
/* ***************************************************************************************************
*   NOTICE
*   This software is the property of Glint Co.,Ltd.. Any information contained in this
*   doc should not be reproduced, or used, or disclosed without the written authorization from
*   Glint Co.,Ltd..
***************************************************************************************************
*   File Name       : test.py
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
*     1.0.0       2024-09-23            Neo                         
****************************************************************************************************/
'''
import os


#hello.py
def sayHello():
    # List out all of the Studies we have on patients.
    part_1 = os.listdir('/home/ai/neo/data/rsna-2024-lumbar-spine-degenerative-classification/train_images')
    
    tmp = filter(lambda x: x.find('.DS') == -1, part_1)
    part_1 = list(filter(lambda x: x.find('.DS') == -1, part_1))
    print(part_1)

if __name__ == "__main__":
    sayHello()