import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import cv2
import torch
import numpy as np
from options import opt

class AlignedDataset(data.Dataset):
    def __init__(self, file_list="",input_nc = 3,output_nc=1,isTrain = True): #/train or /test

        # input_nc is input image channle (A)
        super(AlignedDataset, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.isTrain = isTrain
        self.data_balance = opt.data_balance
        if self.data_balance:
            print("using data_balance")
        self.AB_file_list = file_list
        self.AB_paths = self.get_file_list()

        if self.isTrain:
            self.A_transform = transforms.Compose([
                transforms.Resize((256, 256), Image.BICUBIC),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.59416118, 0.51189164, 0.45280306],
                                     std=[0.25687563, 0.26251543, 0.26231294])]
            )

            self.B_transform = transforms.Compose([
                transforms.Resize((32, 32), Image.BICUBIC),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
        else:
            self.A_transform = transforms.Compose([
                transforms.Resize((256, 256), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.59416118, 0.51189164, 0.45280306],
                                     std=[0.25687563, 0.26251543, 0.26231294])]
            )
            self.B_transform = transforms.Compose([
                transforms.Resize((32, 32), Image.BICUBIC),
                transforms.ToTensor(),
            ])


    def get_file_list(self):
        A_path = []
        B_path = []
        label = []
        for x in open(self.AB_file_list):
            A_path.append(x.strip().split(' ')[0])
            B_path.append(x.strip().split(' ')[1])
            label.append(int(x.strip().split(' ')[2]))
            if self.isTrain and int(x.strip().split(' ')[2])==1:
                A_path.append(x.strip().split(' ')[0])
                B_path.append(x.strip().split(' ')[1])
                label.append(int(x.strip().split(' ')[2]))
                if self.data_balance:
                    A_path.append(x.strip().split(' ')[0])
                    B_path.append(x.strip().split(' ')[1])
                    label.append(int(x.strip().split(' ')[2]))

        return (A_path,B_path,label)



    def __getitem__(self, index):
        A_path = self.AB_paths[0][index]
        B_path = self.AB_paths[1][index]
        label = self.AB_paths[2][index]
        A = Image.open(A_path).convert('RGB')
        A_hsv = Image.open(A_path).convert('HSV')
        B = Image.open(B_path).convert('L')
        A_32 = transforms.Compose([
                transforms.Resize((32, 32), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.59416118, 0.51189164, 0.45280306],
                                        std=[0.25687563, 0.26251543, 0.26231294])]
            )(A)

        A = self.A_transform(A)
        B = self.B_transform(B)
        return {'A': A,'A_32':A_32,'B': B,'label':label,'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths[0])
