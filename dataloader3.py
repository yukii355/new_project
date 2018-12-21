import torch
import torch.utils.data as data
import numpy as np
import cv2 as cv
from PIL import Image
import glob
import os
import random


# 画像取得のdataloaderテスト

'''
directory = os.listdir('/home/moriyama/real_images/')

print(directory)
'''


'''
path = '/home/moriyama/real_images/*.jpg'
files_list = []

files = sorted(glob.glob(path))

p = len(files)

print(files)
'''

'''
folderList = []

def video_dataset():

	for folder in folderList:
			path = '/home/moriyama/real_images/*.jpg'
			imageList = sorted(glob.glob(path))
			for i in range(0, len(imageList), 12):
					tmp = imageList[i:i+12]
					if len(tmp) == 12:
						imageList.append(imageList[i:i+12])
	return imageList

'''

path = '/home/moriyama/real_images/*.jpg'

files = sorted(glob.glob(path))

# filesの中に、取得した画像ファイルが入っている
folderList = []
files_list = []
trainList = []

print('len(files) >> ', len(files)) # 6593 files


'''
以下のコードで、6593枚の画像をリサイズ
'''

for f in files:

	img = Image.open(f)
	# img = np.array(Image.open(f))

	img_resize = img.resize((768,576))

	# img_resize = cv.resize(img, dsize=(768,576)) # (576,768)にリサイズする
	# print((img_resize).shape) # (576,768,3)が出力

	img_numpy = np.array(img_resize)
	# print(img_resize)
	## print((img_numpy).shape)
	# print('len(img_numpy) >> ', len(img_numpy)) # 出力576?


'''
ここから下で、前画像フレーム3チャンネルとoptical flow2チャンネル、
次画像フレーム2チャンネルを取り出す.
(つまり、8チャンネルを1まとめとしてGeneratorに入れることを考える)
以下、例えば12枚を1セットとして、ファイルを抽出してくる
'''


for i in range(0, len(img_numpy), 12):
		tmp = img_numpy[i:i+12]
		if len(tmp) == 12:
			trainList.append(img_numpy[i:i+12]) # appendはリスト内にリストを追加
		print(tmp)
		print((tmp).shape)
		print(len(tmp))
		for j in range(12):
			flow = cv.calcOpticalFlowFarneback(j,j+1,None, 0.5, 3, 15, 3, 5, 1.2, 0)

			print(flow)
			print((flow).shape)







'''
def calc_flow():
# for文で画像を順番に取り出し(基本は12枚ずつ、残りは残りだけ)
	for i in range(0, len(img_numpy), 12):
		tmp = img_numpy[i:i+12]
		if len(tmp) == 12:
			trainList.append(img_numpy[i:i+12]) # appendはリスト内にリストを追加
		print(tmp)
		for j in range(12):
			flow = cv.calcOpticalFlowFarneback(j,j+1,None, 0.5, 3, 15, 3, 5, 1.2, 0)

			print(flow)
			print((flow).shape)
	return flow



calc_flow()
'''




'''
# for文で画像を順番に取り出し(基本は12枚ずつ、残りは残りだけ)
for i in range(0, len(img_resize), 12):
	tmp = img_resize[i:i+12]
	if len(tmp) == 12:
		trainList.append(img[i:i+12]) # appendはリスト内にリストを追加
	print(tmp)
# return tmp
'''

