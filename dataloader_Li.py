import torch
import torch.utils.data as data
import numpy as np
import cv2
import glob
import random

def realTrainList(path):

    # Your real image path
    imageList = sorted(glob.glob(path + '*.jpg'))

    trainList = []
    for i in range(0, len(imageList), 12):
        tmp = imageList[i:i + 12]
        if len(tmp) == 12:
            trainList.append(imageList[i:i + 12])


    return trainList


def randomCropOnList(image_list, output_size):
    cropped_img_list = []

    h, w = output_size
    height, width, _ = image_list[0].shape

    # print(h,w,height,width)

    i = random.randint(0, height - h)
    j = random.randint(0, width - w)

    st_y = 0
    ed_y = w
    st_x = 0
    ed_x = h

    or_st_y = i
    or_ed_y = i + w
    or_st_x = j
    or_ed_x = j + h

    # print(st_x, ed_x, st_y, ed_y)
    # print(or_st_x, or_ed_x, or_st_y, or_ed_y)

    for img in image_list:
        new_img = np.empty((h, w, 3), dtype=np.float32)
        new_img.fill(128)
        new_img[st_y: ed_y, st_x: ed_x, :] = img[or_st_y: or_ed_y, or_st_x: or_ed_x, :].copy()
        cropped_img_list.append(np.ascontiguousarray(new_img))

    return cropped_img_list


class realDataLoader(data.DataLoader):

    def __init__(self, folderPath):
        self.trainList = realTrainList(folderPath)

        print("# of training samples:", len(self.trainList))


    def __getitem__(self, index):
        img_path_list = self.trainList[index]
        h, w, c = cv2.imread(img_path_list[0]).shape



        # print(h,w,c)

        if h > w:
            scaleX = int(360 * (h / w))
            scaleY = 360
        elif h <= w:
            scaleX = 360 # scaleX:360
            scaleY = int(360 * (w / h)) # scaleY:539

        img_list = []
        opticalflow_list = []
        for image_path in img_path_list:
            tmp = cv2.resize(cv2.imread(image_path), (scaleX, scaleY))[:, :, (2, 1, 0)] # tmp = (539,360,3)
            img_list.append(np.array(tmp, dtype=np.float32))

        for i in range(len(img_list)):
            # print(img_list[i].shape)
            # brak
            img_list[i] /= 255
            img_list[i][:, :, 0] -= 0.485  # (img_list[i]/127.5) - 1
            img_list[i][:, :, 1] -= 0.456
            img_list[i][:, :, 2] -= 0.406

            img_list[i][:, :, 0] /= 0.229
            img_list[i][:, :, 1] /= 0.224
            img_list[i][:, :, 2] /= 0.225

            cropped_img_list = randomCropOnList(img_list, (352, 352))
        for i in range(len(img_list) - 1):
            optical = cv2.calcOpticalFlowFarneback(img_list[i],img_list[i+1], None, 0.5, 3, 15, 3, 5, 1.2, 0)
            opticalflow_list.append(optical)

            for i in range(len(cropped_img_list)):
                cropped_img_list[i] = torch.from_numpy(cropped_img_list[i].transpose((2, 0, 1)))
            for i in range(len(opticalflow_list)):
                opticalflow_list[i] = torch.from_numpy(cropped_img_list[i])

            return cropped_img_list, opticalflow_list

    def __len__(self):
        return len(self.trainList)




