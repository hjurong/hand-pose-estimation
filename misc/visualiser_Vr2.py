# -*- coding: utf-8 -*-
"""
Created on Fri May 15 15:12:59 2015

@author: JURONG
"""

import glob
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.widgets import Button
from mpl_toolkits.mplot3d import axes3d
import mpl_toolkits.mplot3d.axes3d as p3
from mpl_toolkits.mplot3d import proj3d

cdir = "C:/Users/juron/Documents/iWork/U-4/ENGN4200/" + \
        "handModelling/Release_2014_5_28/Subject1/"

def load_ptncloud():
    
    ptncloud = np.loadtxt("ptncloud.dat", dtype=float)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(ptncloud[:,0], ptncloud[:,1], ptncloud[:,2],"b.")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()

def read_spheres_txt(line=1, dplot=False):
    sphere_txt = cdir + "sphere.txt"
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

def read_joints_txt(line=1, dplot=False):
    joints_txt = cdir + "joint.txt"
    data = np.loadtxt(joints_txt, skiprows=1)
    
    joints = data[line].reshape(-1, 3)
    joints /= 10.
    
    if dplot:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(joints[:,0],joints[:,1],joints[:,2],"b.")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.show()
        
    return joints.T
    

def get_depthmap(filenum=1, plot=False, to_cm=True):
    ## read a flattened depth map from binfiles
    ## unit of mm
    ## reshape to 240x320

    mm_to_cm = 10. ## mm to cm conversion
    imh, imw = 320, 240
    binfiles = glob.glob(cdir+"/*.bin")
    binfiles.sort()
    
    data = np.fromfile(binfiles[filenum], dtype = np.float32)
      
    
    depth_map = data.reshape(imw, imh) / mm_to_cm

    center = [imw/2., imh/2.]
    constant = 241.42

    # ptncloud = np.zeros((imh,imw,3))
    xgrid = np.ones((imw,1))*np.arange(imh) - center[1]
    ygrid = np.arange(imw).reshape(-1,1) * np.ones((1,imh)) - center[0]

    X = (xgrid*depth_map/constant).flatten()
    Y = (ygrid*depth_map/constant).flatten()
    Z = (depth_map).flatten()
    
    nonzeros, = np.nonzero(Z)
    
    nX = X[nonzeros]
    nY = Y[nonzeros]
    nZ = Z[nonzeros]
    
    XYZ = np.vstack((nX, nY, nZ))
    
    randid = np.random.choice(len(nX), 256)
    
    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(nX[randid],nY[randid],nZ[randid],"b.")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.axis("off")
        plt.show()
    
    return XYZ
    

def cmp_models(line=1, spheres_pos=None, XYZ_file="", figtitle=""):
    
    if spheres_pos is None:
        spheres_pos = read_spheres_txt(line=line)
    
    
    try:
        XYZ_pos = np.loadtxt(XYZ_file, dtype=float)
        XYZ_pos = XYZ_pos.T
    
    except:
        XYZ_pos = get_depthmap(filenum=line)
        np.random.shuffle(XYZ_pos.T)
        XYZ_pos = XYZ_pos[:,:256]
        XYZ_pos[1,:] *= -1
        XYZ_pos[2,:] *= -1
        print "in except"
        
    
    joints_pos = read_joints_txt(line=line)
    
    
    fig = plt.figure()   
    
    ax = fig.add_subplot(111, projection="3d")
    colors = ('b', 'r', 'c', 'm', 'y')
    names = ["thumb", "index", "middle", "ring", "little"]
    start = [0, 8, 18, 28, 38]
    end = [8, 18, 28, 38, 48]
    
#    for i in xrange(5):
#        s_idx = start[i]
#        e_idx = end[i]
#        ax.plot(spheres_pos[0,s_idx:e_idx], 
#                spheres_pos[1,s_idx:e_idx], spheres_pos[2,s_idx:e_idx],
#                marker=".", markersize=18, color=colors[i], label=names[i])
    ax.plot(joints_pos[0,:], joints_pos[1,:], 
            joints_pos[2,:], "g.", markersize=14, label="joints\nground truth")
    ax.plot(XYZ_pos[0,:], 
            XYZ_pos[1,:], 
            XYZ_pos[2,:], "k.", markersize=6, label="ptncloud\n(down sampled)")
    
    ax.set_title(figtitle)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.legend(loc='upper center', 
               bbox_to_anchor=(0.5, 0.23),
               ncol=3, fancybox=True, shadow=True)
    plt.axis("off")
#    ax.set_xticklabels([])
#    ax.set_yticklabels([])
#    ax.set_zticklabels([])
#    ax.grid(False)
    fig.tight_layout()
    plt.show()


    
def plotcost():
    
#    lowerbound = [-np.pi]*3 + [-100]*3 + [np.deg2rad(-15), 0, 0, 0]*5
#    upperbound = [np.pi]*3 + [100]*3 + [np.deg2rad(15), np.pi/2, 
#                                        np.deg2rad(110), np.pi/2]*5
    
    labels = ["Rz", "Ry", "Rx", "Dx", "Dy", "Dz"] + \
             ["MCP1", "MCP2","PIP","DIP"]*5
    xlabel = "No. incr from lb"
    
    data = np.load("data/frame_0001/costfunc_poorVr2.npy")
    ndim, nptns = data.shape
    
    xrng = np.arange(nptns)
    lgsize = 9
    
    fig = plt.figure()
    
    ax = fig.add_subplot(231)    
    for i in xrange(6):
        ax.semilogy(xrng, data[i,:], label=labels[i])
    ax.set_xlabel(xlabel)
    ax.set_ylabel("cost")
    ax.set_title("g pos&trans")
    plt.legend(prop={"size":7}, loc='upper center', 
               bbox_to_anchor=(0.5, 1.05),
               ncol=3, fancybox=True, shadow=True)
    
    ax = fig.add_subplot(232)
    for i in xrange(4):
        idx = 6+i
        ax.semilogy(xrng, data[idx,:], label=labels[idx])
    ax.set_xlabel(xlabel)
    ax.set_ylabel("cost")
    ax.set_title("thumb")
    plt.legend(prop={'size':lgsize})
    
    ax = fig.add_subplot(233)
    for i in xrange(4):
        idx = 10+i
        ax.semilogy(xrng, data[idx,:], label=labels[idx])
    ax.set_xlabel(xlabel)
    ax.set_ylabel("cost")
    ax.set_title("index")
    plt.legend(prop={'size':lgsize})
    
    ax = fig.add_subplot(234)
    for i in xrange(4):
        idx = 14+i
        ax.semilogy(xrng, data[idx,:], label=labels[idx])
    ax.set_xlabel(xlabel)
    ax.set_ylabel("cost")
    ax.set_title("middle")
    plt.legend(prop={'size':lgsize})
    
    ax = fig.add_subplot(235)
    for i in xrange(4):
        idx = 18+i
        ax.semilogy(xrng, data[idx,:], label=labels[idx])
    ax.set_xlabel(xlabel)
    ax.set_ylabel("cost")
    ax.set_title("ring")
    plt.legend(prop={'size':lgsize})
        
    ax = fig.add_subplot(236)
    for i in xrange(4):
        idx = 22+i
        ax.semilogy(xrng, data[idx,:], label=labels[idx])
    ax.set_xlabel(xlabel)
    ax.set_ylabel("cost")
    ax.set_title("little")
    plt.legend(prop={'size':lgsize})
    
    plt.tight_layout()    
    plt.show()





def plot_PSO():
#    root = "data/frame_0001/icppso_wolfe_32p_20g.dat"
#    root = "data/frame_0001/line_search_cmp.dat"
    root = "data/frame_0004/frame0004_cost_cmp1.dat"
    data = np.loadtxt(root, dtype=float)
    
    colors = ('b', 'g', 'r', 'c', 'm', 'y', 'k')
    linestyles = ("-", "--")
    
    try:
        pnum, ngen = data.shape
    except:
        ngen, = data.shape
        pnum = 1
        data = data.reshape(1,-1)
    
    nrows = 1
    ncols = 1
    perplot = pnum / nrows / ncols
    xlabel = "No. PSO generations/ICP iterations"
    lgsize = 10
    
#    figtitle = "ICP; 20gen; 10iter/gen; 20 search/iter\n4threads; 32p; runtime=29.46s"    
    figtitle = "Comparison of optimisation methods on test frame0004\n80 particles for PSO; 32 particles for ICP"      
#    figtitle = "4-thread icp+gbest_pso; 32p; 20gen\n10icp/gen; 10wolfe/icp; runtime=47.4s" 
    
    xrng = np.arange(ngen)
    
    fig = plt.figure()
    axisNum = 0
    gridON = True
    
#    names = ["p"+str(i) for i in xrange(pnum)]
    names = ["Wolfe-ICP+gbest-PSO;\nPSO cost", 
             "Wolfe-ICP+gbest-PSO;\nICP cost", "ICP", "gbest-PSO", "dynamic-PSO"]
    
    
    for i in xrange(nrows):
        for j in xrange(ncols):
            axisNum += 1
            ax = fig.add_subplot(nrows, ncols, axisNum)
            ax.grid(gridON)
            if axisNum < nrows*ncols:
                d_row = (axisNum-1)*perplot-1
                for k in xrange(perplot):
                    
                    if k < perplot / 2:
                        d_row += 1
#                        name = "p"+str(d_row)
                        name = names[d_row]
                        c = colors[d_row % len(colors)]
                        ax.semilogy(xrng, data[d_row,:], 
                                    linestyles[0], label=name, color=c)
                    else:
                        d_row += 1
#                        name = "p"+str(d_row)
                        name = names[d_row]
                        c = colors[d_row % len(colors)]
                        ax.semilogy(xrng, data[d_row,:],
                                    linestyles[1], label=name, color=c)

            
            elif axisNum == nrows*ncols:
                remaining = pnum - (axisNum-1)*perplot
                d_row = (axisNum-1)*perplot-1
                for k in xrange(remaining):
                    
                    if k < perplot / 2:
                        d_row += 1
#                        name = "p"+str(d_row)
                        name = names[d_row]
                        c = colors[d_row % len(colors)]
                        ax.semilogy(xrng, data[d_row,:], 
                                    linestyles[0], label=name, color=c)
                    else:
                        d_row += 1
#                        name = "p"+str(d_row)
                        name = names[d_row]
                        c = colors[d_row % len(colors)]
                        ax.semilogy(xrng, data[d_row,:],
                                    linestyles[1], label=name, color=c)
                                    
            ax.set_title(figtitle) #, fontsize=12
            ax.set_xlabel(xlabel)
            ax.set_ylabel("cost")
            plt.legend(prop={"size":lgsize}, loc='upper center', 
                       bbox_to_anchor=(0.5, 1.0),
                       ncol=2, fancybox=True, shadow=True)
            
    plt.tight_layout(pad=1.01)
    plt.show()


def cal_misalignment(fname, frame=1):

    try:
        opt_joints = np.loadtxt(fname)
    except:
        return "ERROR IN LOADING FILE"
    
    gnd_joints = read_joints_txt(line=frame).T

#    fig = plt.figure()
#    ax = fig.add_subplot(111, projection="3d")
    
#    for i in range(len(gnd_joints)):        
#        ax.plot([opt_joints[i,0]], 
#                [opt_joints[i,1]], [opt_joints[i,2]], 'b.', markersize=18)
#        ax.plot([gnd_joints[i,0]], 
#                [gnd_joints[i,1]], [gnd_joints[i,2]], 'k.', markersize=18)
#        plt.pause(3)
    
    pos = [0, 4, 8, 12, 16, -1]
    
    dis = np.linalg.norm(opt_joints - gnd_joints, axis=1)
    
    return np.sum(dis[pos])
    
    
def plot_joints(fname):
    try:
        opt_joints = np.loadtxt(fname)
    except:
        return "ERROR IN LOADING FILE"
    
    pos = [1, 5, 9, 13, 17, 22]         
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    geo = np.array([
                    68.96344533,  30.80009179,  23.10057695,  20.90010475,
                    74.27700757,  36.29982272,  24.20042803,  21.99987377,
                    77.44081579,  31.9002325 ,  20.90000473,  19.7995486 ,
                    80.15924911,  23.10022144,  14.29931098,  15.39998521,
                    31.1074037 ,  27.2800217 ,  22.00017175,  21.99995952])
                
    
    for i in xrange(5):
        start = pos[i]
        end = pos[i+1]
        X = np.hstack((opt_joints[0,0], opt_joints[start:end,0]))
        Y = np.hstack((opt_joints[0,1], opt_joints[start:end,1]))
        
        ax.plot(X,Y, "-o", linewidth=3, markersize=15)
        
        for j in xrange(1, len(X)):
            midx = (X[j-1]+ X[j]) * 0.5
            midy = (Y[j-1]+ Y[j]) * 0.5
            
            l = np.round(geo[i*4+j-1], decimals=2)
            ax.annotate(str(l), 
                        xy = (midx, midy), 
                        xytext = (-20, 20),
                        textcoords = 'offset points',
                        ha = 'right', va = 'bottom',
                        bbox = dict(boxstyle = 'round,pad=0.5',
                                    fc = 'yellow', alpha = 0.5),
                        arrowprops = dict(arrowstyle = '->',
                                          connectionstyle = 'arc3,rad=0'))
                                          
    ax.set_title("Hand Skeleton geometry in mm")                                    
    plt.show()
    

def plot_spheresR(fname):
    try:
        spheres = np.loadtxt(fname)
    except:
        return "ERROR IN LOADING FILE"
    
    rii =  np.array([
                18.26  ,  15.18  ,  16.61  ,  14.96  ,  13.53  ,  12.1   ,
                10.45  ,   8.8   ,  12.1   ,  11.66  ,  11.8067,  11.9533,
                11.11  ,  10.56  ,  10.12  ,   9.68  ,   9.46  ,   9.24  ,
                16.5   ,  15.0333,  13.5667,  12.1   ,  11.55  ,  11.    ,
                10.78  ,  10.56  ,  10.01  ,   9.46  ,  16.5   ,  10.56  ,
                12.54  ,  14.52  ,  10.12  ,   9.68  ,   9.46  ,   9.24  ,
                 8.8   ,   8.36  ,  12.1   ,  11.2933,  10.4867,   9.68  ,
                 9.46  ,   9.24  ,   8.8   ,   8.36  ,   7.81  ,   7.26  ])
    
    
    fig = plt.figure()
    ax = fig.add_subplot(111)    
    for i in range(len(spheres)):
        x = spheres[i,0]
        y = spheres[i,1]
        ax.scatter(x, y, s=100*rii[i], c=np.random.rand(1,3),  
                   marker=r"$ {} $".format(rii[i]), edgecolors='none' )
    
    plt.show()