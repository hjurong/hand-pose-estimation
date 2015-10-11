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

class visualiser(object):
    def __init__(self, path=""):
        
        if path:
            self.path = path
        else:
            cdir = "C:/Users/RYAN/Documents/iWork/U-4/ENGN4200/" + \
                    "handModelling/Release_2014_5_28/Subject1/"
            self.path = cdir
            
        self.current_frame = 0
        self.sph_plt = None
        
        dataloc = "C:/Users/RYAN/Documents/iWork/U-4/" + \
                    "ENGN4200/handposeEstimation/data/"
                    
        filename = "*.dat"
        files = glob.glob(dataloc+filename)
        
        self.files = files
        self.num_files = len(files)
        
        
    def read_spheres_txt(self, line=0, dplot=False):
        sphere_txt = self.path + "sphere.txt"
        data = np.loadtxt(sphere_txt, skiprows=1)
        spos = data[line].reshape(48, 5)
        
        spheres = spos[:,1:4]
        spheres /= 10. # convert from mm to cm
        
        
        if dplot:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            ax.plot(spheres[:,0],spheres[:,1],spheres[:,2],"b.")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            plt.show()
            
        return spheres.T

    def get_depthmap(self, filenum=1, plot=False, to_cm=True):
        ## read a flattened depth map from binfiles
        ## unit of mm
        ## reshape to 240x320
    
        mm_to_cm = 10. ## mm to cm conversion
        imh, imw = 320, 240
        binfiles = glob.glob(self.path+"/*.bin")
        data = np.fromfile(binfiles[filenum], dtype = np.float32)
        depth_map = data.reshape(imw, imh).T / mm_to_cm
    
        center = [imh/2., imw/2.]
        constant = 241.42
    
        # ptncloud = np.zeros((imh,imw,3))
        xgrid = np.ones((imh,1))*np.arange(imw) - center[0]
        ygrid = np.arange(imh).reshape(-1,1) * np.ones((1,imw)) - center[1]
    
        X = (xgrid*depth_map/constant).flatten()
        Y = (ygrid*depth_map/constant).flatten()
        Z = (depth_map).flatten()
        
        nonzeros, = np.nonzero(Z)
        
        nX = X[nonzeros]
        nY = Y[nonzeros]
        nZ = Z[nonzeros]
        
        XYZ = np.vstack((nX, nY, nZ))
        
        if plot:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            ax.plot(nX,nY,nZ,"b.")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            plt.show()
        
        return XYZ
        
    
    def cmp_models(self):
        spheres_pos = -1*self.read_spheres_txt()
        XYZ_pos = self.get_depthmap();
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(spheres_pos[0,:], spheres_pos[1,:], spheres_pos[2,:],"b.")
        ax.plot(XYZ_pos[0,:], XYZ_pos[1,:], XYZ_pos[2,:], "g.")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.show()
        
    
    def vis_optfunc_data(self):
        
        dataloc = "C:/Users/RYAN/Documents/iWork/U-4/" + \
                    "ENGN4200/handposeEstimation/data/"
                    
        filename = "*.dat"
        files = glob.glob(dataloc+filename)
        num_files = len(files)
        current_frame = 0
        data_array = np.loadtxt(files[0],dtype=np.float64)
        
        #http://bastibe.de/2013-05-30-speeding-up-matplotlib.html
        #http://matplotlib.org/examples/animation/simple_3danim.html
        fig = plt.figure()
        ax = p3.Axes3D(fig)
        
        
        depthmap = self.get_depthmap()
        
        dmap, = ax.plot(depthmap[0,:], 
                        depthmap[1,:], depthmap[2,:], "b.") 
        ptns, = ax.plot(data_array[:,0], 
                        data_array[:,1], data_array[:,2], "g.", markersize=16)
        
        self.sph_plt = ptns
        
        def draw_new_data(data_array, ptns):
            
            print "0-"
            print data_array2.shape
            ptns.set_data(data_array[:2,:])
            ptns.set_3d_properties(data_array[2,:])
            
            plt.pause(0.01)
        
        data_array2 = np.loadtxt(files[-1],dtype=np.float64)
        draw_new_data(data_array2, ptns)        
        
        axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
        axprev = plt.axes([0.70, 0.05, 0.1, 0.075])
        bnext = Button(axnext, 'Next')
        bprev = Button(axprev, 'Previous')
        
        
        
        def btn_next(event):
            
            current_frame = 1
            
            if current_frame < 0 or current_frame > num_files:
                current_frame = 0
            
            data_array = np.loadtxt(files[current_frame], dtype=float)
            
            draw_new_data(data_array)
            
            plt.draw()


#        bnext.on_clicked(btn_next)
#        bprev.on_clicked(self.btn_update_next)
        
        
        
        
#        for i in range(1, self.num_files):
#            data_array = np.loadtxt(self.files[i], dtype=np.float64).T
#            ptns.set_data(data_array[:2,:])
#            ptns.set_3d_properties(data_array[2,:])
#            
#            plt.pause(0.5)
            
            
    def btn_update_next(self, event):
        
        self.current_frame += 1
        
        
        
        print self.current_frame
        
        if self.current_frame < 0 or self.current_frame > self.num_files:
            self.current_frame = 0
        
        data_array = np.loadtxt(self.files[self.current_frame], dtype=float)
#        
#        self.sph_plt.set_data(data_array[:2,:])
#        self.sph_plt.set_3d_properties(data_array[2,:])
        
        
        
        
        plt.draw()
        
    
        
        
            
            
        
        
        
        
        
        

def main():
    plt.close("all")
    vis = visualiser()
#    vis.read_spheres_txt()
#    vis.cmp_models()
    vis.vis_optfunc_data()

main()