# -*- coding: utf-8 -*-
"""
Created on Sat May 09 12:23:46 2015

@author: JURONG
"""

import glob
import numpy as np
import matplotlib.pyplot as plt


from matplotlib.widgets import Button
from mpl_toolkits.mplot3d import axes3d
import mpl_toolkits.mplot3d.axes3d as p3


filenames = glob.glob("*.dat")   

num_files = len(filenames)
current_file = 0

hand_spheres = np.loadtxt(filenames[current_file],dtype=float)
hand_spheres = hand_spheres.T


axoff = True

#http://bastibe.de/2013-05-30-speeding-up-matplotlib.html
#http://matplotlib.org/examples/animation/simple_3danim.html
fig = plt.figure()
ax = p3.Axes3D(fig)


cens, = ax.plot(hand_spheres[0,:], 
                hand_spheres[1,:], 
                hand_spheres[2,:], "b.", markersize=12, label="point cloud")
ax.legend()
ax.axis("off")

def draw_new_data(new_spheres):
        
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
    
    global current_file
    global data_array
    global hand
    
    current_file += 1
    current_file %= num_files
    
    new_spheres = np.loadtxt(filenames[current_file],dtype=float).T

    
    draw_new_data(new_spheres)
    
    plt.title(filenames[current_file])
    plt.draw()
    

def btn_prev(event):
    
    global current_file
    global data_array
    global hand
    
    current_file -= 1
    current_file %= num_files
    
    new_spheres = np.loadtxt(filenames[current_file],dtype=float).T
    
    draw_new_data(new_spheres)
    
    plt.title(filenames[current_file])
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
        new_spheres = np.loadtxt(filenames[i],dtype=float)
        draw_new_data(new_spheres)
        plt.title(filenames[i])
        plt.draw()
        

bnext.on_clicked(btn_next)
bprev.on_clicked(btn_prev)
bxoff.on_clicked(btn_ax_off)
bauto.on_clicked(btn_auto)
    

        

            
        
        
        
        
        
