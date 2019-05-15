import torch
import torch.nn as nn 
import torchvision.transforms as transforms
import torch.nn.functional as F 
import cv2 
from PIL import Image , ImageFilter

import matplotlib.pyplot as plt
import math
import numpy as np 
import time 
from modify_histo import match_histo
import sys
import os

DIRECTIONNUM = 8
LINEDIVISOR = 25


def np_show(Imag):
	img = Image.fromarray(Imag*255)
	img.show()

def show_hsv(img , filename = None):
	img = Image.fromarray(img , mode = 'HSV')
	img = img.convert('RGB')
	img.show()

	
	if filename:
		img.save(os.path.join('output', filename))


def tensor_save_output(img_tensor, name):
    img = img_tensor.squeeze().numpy()
    img = Image.fromarray(img * 255)

    if img.mode != 'RGB':
        img = img.convert('RGB')

    img.save(name)
	# if img.mode != 'RGB':
	# 	img = img.convert('RGB')
	# img.save(name)


class TextureLearn(nn.Module):
	def __init__(self , textname,target ,device = None):
		super().__init__()
		if device:
			self.device = device
		else :
			self.device  = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

		self.theta = 0.5
		self.avoid_add = 0.001
		self.C ,self.H , self.W = target.size()
		print(type(self.H),type(self.W))
		texture = Image.open(textname).resize((self.W , self.H))
		texture = texture.convert('L') 	
		#texture.show()
		self.H = transforms.ToTensor()(texture).to(self.device)


		self.logH = torch.log(self.H) 
		target = target.to(self.device) + self.avoid_add
		self.logTar = torch.log(target) 
		# print(self.logTar.size()  ,self.logTar)
		#self.weight = torch.zeros_like( self.logTar , requires_grad =True)
		#self.weight = nn.Parameter(torch.Tensor(self.logTar.size()) , requires_grad = True)
		
		# self.weight.data.uniform_(0.0, 0.02)
		self.weight = nn.Parameter(torch.abs(torch.randn_like(self.logTar)))
	
		print( 'weight , target , H ' ,self.weight.size(), self.logTar.size(), self.H.size())
		print('okkkkkkay')

	def forward(self):
		return  self.H  **self.weight - self.avoid_add
	
	def loss (self):
		#print (((self.weight * self.logH - self.logTar)**2).mean())
		#print(((self.weight[:-1,:]-self.weight[1:,:])**2).mean() )
		content_loss = ((self.weight * self.logH - self.logTar)**2).sum()
		dx_loss = self.theta * ((self.weight[:,:-1,:]-self.weight[:,1:,:])**2).sum() 
		dy_loss =   self.theta * ((self.weight[:,:,:-1]-self.weight[:,:,1:])**2).sum()
		
		loss = content_loss  + dy_loss + dx_loss 
		#print ('content ,dy ,dx' , content_loss , dy_loss , dx_loss)
		#print('loss ', type(loss.item()),loss.item())
		return loss 

	def train(self):
		self.to(self.device)
		optimizer = torch.optim.Adam(self.parameters(), lr=0.8)
		for i in range (150):
			loss = self.loss()
			optimizer.zero_grad()
			loss.backward()
			if i %10 == 9:
				print('render loss :',loss.item())
			optimizer.step()

		return 
		

class PencilDraw(nn.Module):
	def __init__(self , device = 'cpu'):
		super().__init__()
		self.transform = transforms.Compose([transforms.ToTensor()])
		self.unloader = transforms.ToPILImage()
		self.device = device
	
	def show(self, x , filename =None , mode = None):
		x = x.to('cpu')
		show = self.unloader(x)
		#print(type(show))
		if mode :
			show.mode = mode 
		show.show()
		if filename:
			show.save(os.path.join('output', filename))
		#(show , filename)

	# data is C*H*W tensor 

	def get_line(self,input_data , gammaS = 1 ):

		input_data = input_data.to(self.device)
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
		
		kernel = nn.Parameter(kernel,requires_grad = False).to(self.device).to(self.device)

		#compute dx , dFy  forward differnece
		dx = torch.cat([input_data[:,:,:-1]- input_data[:,:,1:],torch.zeros(1,H,1).to(self.device)],2)
		dy = torch.cat ([input_data[:,:-1,:]-input_data[:,1:,:],torch.zeros(1,1,W).to(self.device)],1)
		d_image = torch.sqrt(dx*dx+dy*dy)
		self.show((1-d_image)**gammaS ,'test_gd.jpg')

		# improve 
		d_image[d_image<0.015] = 0
		#d_image[d_image>0] +=0.01

		G = torch.zeros(1,DIRECTIONNUM,H,W).to(self.device)
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
		C = torch.zeros(1,DIRECTIONNUM,H,W).to(self.device)
	
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
		self.show(Sp , 'test_line.jpg')
		return Sp
	
	def get_tone (self, img_data):
		
		mod_data = match_histo(img_data)
		mod_data = self.transform(mod_data)
		self.show(mod_data ,'test_tone.jpg')
		return mod_data

	def render_texture(self, img_data , texturename):
		
		print('start render !!!!')
		
		texture = TextureLearn(texturename , img_data ,self.device)
		texture.train()
		render_result = texture()

		self.show(render_result, 'test_texture.jpg')

		return render_result



	def forward(self,filename , mode = 'gray'):
		img = Image.open(filename)
				
		counter = 1 
		print(min(img.size[0] , img.size[1])// counter)
		while (min(img.size[0] , img.size[1])/ counter >1000):
			counter *=2 
			print('test counter' , counter)
		
		img = img.resize((img.size[0]//counter,img.size[1]//counter))
		img = img.filter(ImageFilter.SMOOTH)
		print(img.size)
		if mode == 'gray':
			img_gray = img.convert('L') 
			img_gray = np.array(img_gray)	
		else :
			img_color = np.array(img.convert('HSV'))
			print ('test pre img color size ' , img_color.shape)
			img_gray = img_color[:,:,2]
			
		
		clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
		img_gray = clahe.apply(img_gray)

		tone = self.get_tone(img_gray).to(self.device)
		tone = self.render_texture(tone , 'texture.jpg')
		# C*H*W tensor
		img = self.transform(img)
		img_gray = self.transform(img_gray)

		_ , self.H , self.W = img_gray.size()
		print ( 'img size ' ,self.H , self.W)
		
		line = self.get_line(img_gray)

		# tensor_save_output(tone.to('cpu'), 'tone.jpg')
		# tensor_save_output(line.to('cpu') ,'line.jpg')
		S = tone * line 

		if mode == 'gray':
			self.show(S,'test_result.jpg')
		else :
			img_color[:,:,2] = S.detach().cpu().numpy()*255 
			show_hsv(img_color,'test_result_color.jpg')
			#self.show(img_color , 'test_result_color.jpg')

if __name__ == "__main__":


	start = time.time()
	pc = PencilDraw(device = 'cuda')
	
	argv = sys.argv
	if len(argv)>1:
		filename = argv[1]
	else :
		filename = 'test.jpg'
	if len(argv)>2:
		mode = argv[2]
	else :
		mode = 'gray'

	pc(filename , mode )

	print('spend time ',time.time() - start)
