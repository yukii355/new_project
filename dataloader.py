import torch
from torch.utils.data import DataLoader
import cv2 as cv
import os
import numpy as np
from torchvision import transforms
from PIL import Image
def image_list(path="~/home/moriyama/", type="real_images/"):
    data_list = []
    for image in sorted(os.listdir(path + type)):
        data_list.append(os.path.join(path + type, image ))

    return data_list




transform = transforms.Compose([transforms.ToPILImage(),
                                transforms.Resize(256),
                                transforms.RandomCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


'''
def video_list(path="~/home/moriyama/", type="real_videos/"):
    data_list =[]
    for video in sorted(os.listdir(path + type)):
        data_list.append(os.path.join(path + type, video))

    return data_list
'''




class image_dataset(DataLoader):

    def __init__(self, path, transforms=transform, type = "real_images/"):

        self.real_image_list = image_list(path=path, type=type)
        self.transforms = transforms
        print("# of training real images samples:", len(self.real_image_list))

    def __getitem__(self, index):

        img_path_list = self.real_image_list[index]
        img = cv.imread(img_path_list)[:,:,(2,1,0)]
        return self.transforms(img)

    def __len__(self):
        return len(self.real_image_list)



# data = image_dataset(path="/home/yachao-li/Downloads/")
# dataloader = torch.utils.data.DataLoader(data,128)
# for i in dataloader:
#     print(i.size())




# cartoonGANでは上記の画像dataloaderを使ったが、video-to-videoでは下のコードを使う

'''
class video_dataset(DataLoader):

    def __init__(self,path, transforms=transform, type="real_videos/"):

        self.real_video_list = video_list(path)
        self.transforms = transform
'''



'''
class video_dataset(DataLoader):

    def __init__(self):

        video_path = "vtest.avi"

        cap = cv.VideoCapture(video_path)

        ret, frame1 = cap.read()
        prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
        hsv = np.zeros_like(frame1)  # hsv = (576, 768, 3)

        # set saturation
        hsv[..., 1] = 255  # hsv[...,1] = (576, 768)

        return frame1

'''


def optical_flow(index):

    real_image_list = image_list(path="~/home/moriyama/", type="real_images/")
    img_path_list = real_image_list[index]
    img = cv.imread(img_path_list)[:, :, (2, 1, 0)]
    for i in range(len(real_image_list) - 1):
        prvs = cv.cvtColor(real_image_list[i], cv.COLOR_BGR2GRAY)
        next = cv.cvtColor(real_image_list[i + 1], cv.COLOR_BGR2GRAY)

        optical = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        return optical



#　videoをloadしたいが、下のコードでいいのか考え中.

def video_dataset():

    cap = cv.VideoCapture('vtest.avi')

    while(cap.isOpened()):
        ret, frame = cap.read()

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        cv.imshow('frame', gray)

        k = cv.waitKey(30) & 0xff
        if k == 27:
            break

    cv.destroyAllWindows()
    cap.release()