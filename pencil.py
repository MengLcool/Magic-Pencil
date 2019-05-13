import torch
import torch.nn as nn 
import torchvision.transforms as transforms
import torch.nn.functional as F 
import cv2 
from PIL import Image
import matplotlib.pyplot as plt
import math
import numpy as np 

DIRECTIONNUM = 8
LINEDIVISOR = 25



class PencilDraw(nn.Module):
	def __init__(self):
		super().__init__()
		self.transform = transforms.Compose([transforms.ToTensor()])
		self.unloader = transforms.ToPILImage()
		
	def show(self, x ):
		show = self.unloader(x)
		print(type(show))
		show.show()

	# data is C*H*W tensor 
	# TODO: add cuda compute
	def get_line(self,input_data , gammaS = 5 ):

		print(input_data.dtype)
		print('max min ' ,input_data.max(), input_data.min())
		# get kernel 
		C ,H ,W = input_data.size()
		kernel_size = min(input_data.size()[-2:]) //LINEDIVISOR
		kernel_size += kernel_size%2==0
		half_size = kernel_size /2 

		print(kernel_size)
		
		# calculate kernel 
		kernel = torch.zeros(DIRECTIONNUM,kernel_size,kernel_size)
		for i in range(DIRECTIONNUM):
			if 	i != (DIRECTIONNUM // 2) and abs(math.tan(math.pi/DIRECTIONNUM * i))<=1:
				d = math.tan(math.pi/DIRECTIONNUM * i)
				for x in range (kernel_size):
					# if(i==2):	
					# 	print(x , int(math.tan(math.pi/DIRECTIONNUM*i)*(x-half_size)+half_size))
					y = int(d*(x-half_size)+half_size)
					if y < kernel_size and y>=0 :
						kernel[i,y,x] = 1
			else :
				d = 1 / math.tan(math.pi/DIRECTIONNUM * i)
				for y in range (kernel_size):
					x = int(d*(y-half_size)+half_size)
					if x < kernel_size and x >=0:
						kernel[i,y,x] = 1
		

		#compute dx , dy  forward differnece
		dx = torch.cat([input_data[:,:,:-1]- input_data[:,:,1:],torch.zeros(1,H,1)],2)
		dy = torch.cat ([input_data[:,:-1,:]-input_data[:,1:,:],torch.zeros(1,1,W)],1)
		d_image = torch.sqrt(dx*dx+dy*dy)
		self.show((1-d_image)**gammaS)

		# improve 
		d_image[d_image<0.01] = 0
		#d_image[d_image>0] +=0.01

		G = torch.zeros(1,DIRECTIONNUM,H,W)
		for n in range(DIRECTIONNUM):
			data = d_image[0]
			output = F.conv2d(data.expand(1,1,*data.size()),kernel[n].expand(1,1,*kernel[n].size()),padding=kernel[n].size()[0]//2 )     
			G[0,n] = output 

		# below has a error , ask why 
		# G = F.conv2d(input_data.unsqueeze(0),kernel.unsqueeze(1), padding = int(half_size))
		##############################################
		
		print('test g size ' , G.size())
		_ , g_index = G.max(1) 
		g_index = g_index.squeeze()
		C = torch.zeros(1,DIRECTIONNUM,H,W)
	
		# classify 
		for i in range(DIRECTIONNUM):
			C[0,i] =   d_image[0]*(g_index==i).float()

		print(d_image.sum() , C.sum())
	
		for i in range(DIRECTIONNUM):
			c_tmp = C[0,i]
			kernel_tmp = kernel[i]
			C[0,i] = F.conv2d(c_tmp.expand(1,1,*c_tmp.size()),kernel_tmp.expand(1,1,*kernel_tmp.size()),padding=int(half_size))[0,0]
			print(C[0,i].size())
		
		Spn = C
		Sp = Spn.sum(1)
		print('sp size ' , Sp.size())
		Sp = (Sp - Sp.min()) / (Sp.max() - Sp.min())
		Sp = (1- Sp) **gammaS
		self.show(Sp)
	
	
	def forward(self,filename):
		img = Image.open(filename)
				
		counter = 1 
		if min(img.size[0] , img.size[1])// counter >1000:
			counter *=2 
		img = img.resize((img.size[0]//counter,img.size[1]//counter))

		img_gray = img.convert('L') 

		# C*H*W tensor
		img = self.transform(img)
		img_gray = self.transform(img_gray)

		print('test img gray size ' , img_gray.size() )
		self.get_line(img_gray)

if __name__ == "__main__":
	pc = PencilDraw()
	pc('test2.jpg')

