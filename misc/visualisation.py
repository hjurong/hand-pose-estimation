# -*- coding: utf-8 -*-
"""
Created on Sat May 09 12:23:46 2015

@author: JURONG
"""

import glob
import numpy as np
import matplotlib.pyplot as plt

from visualiser_Vr2 import read_joints_txt
from handmodel import handmodel

from matplotlib.widgets import Button
from mpl_toolkits.mplot3d import axes3d
import mpl_toolkits.mplot3d.axes3d as p3

home = "C:/Users/juron/Documents/iWork/U-4/ENGN4200/"
cdir = home + "handModelling/Release_2014_5_28/Subject1/"
fdir = home + "handposeEstimation/misc/data/"

filename = fdir + "out_theta.dat"    


data_array = np.loadtxt(filename,dtype=float)
data_array[:,:3] = np.deg2rad(data_array[:,:3])
data_array[:,6:] = np.deg2rad(data_array[:,6:])

axoff = True
current_frame = 0
num_files, theta_dim = data_array.shape

#http://bastibe.de/2013-05-30-speeding-up-matplotlib.html
#http://matplotlib.org/examples/animation/simple_3danim.html
fig = plt.figure()
ax = p3.Axes3D(fig)


hand = handmodel()
hand_spheres = hand.build_hand_model(data_array[current_frame,:]).T
hand_jointsp = read_joints_txt(line=current_frame)

cens, = ax.plot(hand_spheres[0,:], 
                hand_spheres[1,:], 
                hand_spheres[2,:], "b.", markersize=12, label="optimised model") 
ptns, = ax.plot(hand_jointsp[0,:], 
                hand_jointsp[1,:], 
                hand_jointsp[2,:], "g.", markersize=16, label="ground truth")
ax.legend()
ax.axis("off")

def draw_new_data(new_spheres, new_joints):
    
    ptns.set_data(new_joints[:2,:])
    ptns.set_3d_properties(new_joints[2,:])
    
    cens.set_data(new_spheres[:2,:])
    cens.set_3d_properties(new_spheres[2,:])
    
    plt.draw()


axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
axprev = plt.axes([0.70, 0.05, 0.1, 0.075])
axauto = plt.axes([0.59, 0.05, 0.1, 0.075])
axsoff = plt.axes([0.48, 0.05, 0.1, 0.075])
bnext = Button(axnext, 'Next')
bprev = Button(axprev, 'Previous')
bauto = Button(axauto, 'AUTO')
bxoff = Button(axsoff, 'ax-off/on')


def btn_next(event):
    
    global current_frame 
    global data_array
    global hand
    
    current_frame += 1
    current_frame %= num_files
    
    new_spheres = hand.build_hand_model(data_array[current_frame,:]).T
    new_hjoints = read_joints_txt(line=current_frame)
    
    draw_new_data(new_spheres, new_hjoints)
    
    plt.draw()
    

def btn_prev(event):
    
    global current_frame 
    global data_array
    global hand
    
    current_frame -= 1
    current_frame %= num_files
    
    new_spheres = hand.build_hand_model(data_array[current_frame,:]).T
    new_hjoints = read_joints_txt(line=current_frame)
    
    draw_new_data(new_spheres, new_hjoints)
    
    plt.draw()

def btn_ax_off(event):
    
    global axoff
    
    if axoff:
        ax.axis("on")
        axoff = False
    else:
        ax.axis("off")
        axoff = True
    plt.draw()
    
def btn_auto(event):
    
    global data_array
    global hand
    
    for i in range(num_files):
        new_spheres = hand.build_hand_model(data_array[i,:]).T
        new_hjoints = read_joints_txt(line=i)
        draw_new_data(new_spheres, new_hjoints)
        plt.draw()
        

bnext.on_clicked(btn_next)
bprev.on_clicked(btn_prev)
bxoff.on_clicked(btn_ax_off)
bauto.on_clicked(btn_auto)
    

        

            
        
        
        
        
        
