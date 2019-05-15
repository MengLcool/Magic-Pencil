import torch
import cv2 
import numpy as np 
import math
from PIL import Image ,ImageFilter


DATARANGE = 256



def np_p1 (theta_b = 9):
    '''
    p1 : Laplacian distribution ->brightest value

    '''
    x = np.array(range(DATARANGE),dtype=np.float32)
    distr = np.exp((x-DATARANGE)/theta_b)/theta_b
    distr /= distr.sum()
    
    # distr_sum = np.zeros(DATARANGE , dtype = np.float32)
    # for i in range(1,256):
    # 	distr_sum[i] = distr_sum[i-1] + distr[i] 

    # #return distr , distr_sum
    return distr

def np_p2 (u_a = 105 , u_b = 225 ):
    '''
    p2 : uniform distribution ->mid tone 

    '''

    distr = np.zeros(DATARANGE , dtype = np.float32)
    distr[u_a :u_b] = 1/(u_b - u_a)

    # distr_sum = np.zeros(DATARANGE , dtype = np.float32 )
    # for i in range(u_a , u_b):
    # 	distr_sum[i]  = distr_sum[i-1] + distr[i]
    # distr_sum[u_b:] = 1 
    #return distr , distr_sum
    return distr

def np_p3(ud = 90 , theta = 11 ):
    
    x = np.array(range(DATARANGE) , dtype = np.float32)
    distr = 1 / math.sqrt(2*math.pi*theta) * np.exp((x-ud)**2/(-2*theta**2))
    distr /= distr.sum()

    # distr_sum = np.zeros(DATARANGE , dtype = np.float32)
    # for i in range(1, DATARANGE):
    # 	distr_sum[i] = distr_sum[i-1] + distr[i]
    #return distr , distr_sum
    return distr

def np_p( weight_s =[ 86, 22 , 2] ):

    w1 , w2 , w3 = weight_s
    p1 = np_p1()
    p2 = np_p2()
    p3 = np_p3()

    p = w1*p1 + w2*p2 + w3*p3 
    p /= p.sum()
    p_total = np.zeros(DATARANGE)

    for i in range(1 , DATARANGE):
        p_total[i] = p_total[i-1] + p[i]
    
    return p , p_total


def get_histo ( input_image ):

    print('image type ' , input_image.dtype)
    p_image = np.zeros(DATARANGE , dtype = np.float32)
    p_image_total = np.zeros(DATARANGE, dtype = np.float32)
    for i in range(DATARANGE):
        p_image[i] = (input_image==i).sum()
    p_image  /= p_image.sum()
    for i in range(1 , DATARANGE):
        p_image_total[i] = p_image_total[i-1] + p_image[i]

    return p_image , p_image_total



def match_histo(input_image ):
    '''
    input_image : [0,255] h*w  np array

    '''

    p_image , p_image_total = get_histo(input_image)
    p_pencil , p_pencil_total = np_p()

    num_map = []
    color_pencil = 0
    #draw_pic(p_pencil)

    modified_image = np.zeros_like(input_image)
    for color_input in range (DATARANGE):
        
        while color_pencil <DATARANGE-1 and  abs(p_image_total[color_input]- p_pencil_total[color_pencil]) > abs(p_image_total[color_input]- p_pencil_total[color_pencil+1]):
            color_pencil +=1 

        modified_image[input_image == color_input] = color_pencil
    
    p_image , p_image_total = get_histo(modified_image)
    #draw_pic(p_image)

    return modified_image


def show (x ):
    img = Image.fromarray(x)
    img.show()
    

def draw_pic ( y , title = None ):
    import matplotlib.pyplot as plt
    
    if title :
        plt.title(title) 
    x = np.linspace(0,255,256)
    plt.plot(x,y)
    plt.show()


if __name__ =='__main__':

    imr = Image.open('test2.jpg')
    im = imr.convert("L")
    J = np.array(im)
    match_histo(J)
    
    