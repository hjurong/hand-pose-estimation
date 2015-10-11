# -*- coding: utf-8 -*-
"""
Created on Thu Apr 09 19:30:13 2015

@author: JURONG
"""

import os
import cv2
import glob
import time
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import axes3d


def project_spheres(spheres, depthmap, scale, plot=False):
    """
        project a spheres model onto the depth map; this is done using the 
        transformation:
                        const*[u, v, 1] = K[X, Y, Z]; 
        where:
        const = constant --> required to get the homogeneous 2d coordinate
        [u, v, 1] = homogeneous 2d coordinate of depth map
        [X, Y, Z] = 3d coordinate of sphere center
        K = camera calibration matrix of the form:
        [[f, 0, cx], [0, f, cy], [0, 0, 1]]
    """
    f = 241.42
    imH, imW = 320, 240
    cx, cy = imH/2, imW/2
    K = np.array([f, 0, cx, 0, f, cy, 0, 0, 1]).reshape(3,3)
    
    assert spheres.shape[0] == 3
    
    projection = np.dot(K, spheres) # shpheres.shape = (3, nspheres)
    projected = projection[:2,:] / projection[2,:] # convert to homogeneous 
    projected = np.around(projected.T) # shape == (-1, 2)

    
    def invert_depthmap(depth, imh=320, imw=240, dplot=False):
        depth = depth.flatten()
        zeros = np.where(depth==0)
        nonzeros = np.where(depth!=0)
        depth[zeros] = 1.
        depth[nonzeros] = 0.
        depth = depth.reshape(imh, imw)
        
        if dplot:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.imshow(depth)
            plt.show()
        
        return np.int8(depth)
    
    inverted_depth = invert_depthmap(depthmap, imh=imH, imw=imW, dplot=plot)
    dist_transform = cv2.distanceTransform(inverted_depth,cv2.cv.CV_DIST_L2,5)        
    
    total = 0
    rgn = spheres.shape[1]
    for i in range(rgn):        
        dx, dy = projected[i]
        xbounded = dx >= 0 and dx < depthmap.shape[0]
        ybounded = dy >= 0 and dy < depthmap.shape[1]
        
        if xbounded and ybounded:
            D_jc = depthmap[dx][dy]
            if D_jc != 0:
                D_cz = spheres[2,i]
                total += np.square(max(0, D_jc-D_cz))
            else:
                total += np.square(dist_transform[dx][dy] * scale)
        else:
            total += np.square(dist_transform.max() * scale)
            
    
    
    if plot:
        plt.figure()
        plt.imshow(dist_transform)
        plt.figure()
        plt.plot(projected[0], projected[1], "b.")
        plt.show()

    return total


def display_ptncloud():
    cpath = os.getcwd()
    loadp = cpath + "/handposeEstimation/"
    datfile = loadp + "ptncloud.mat"

    data = sio.loadmat(datfile)
    ptncloud = data["ptncloud"]
    
    X = ptncloud[:,0]
    Y = ptncloud[:,1]
    Z = ptncloud[:,2]
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(X,Y,Z,"k.")
    plt.show()

    return 0


def load_depthmap(filenum=1):
    imh, imw = 320, 240
    cdir = "C:/Users/juron/Documents/iWork/U-4/ENGN4200"
    path = cdir + "/handModelling/Release_2014_5_28/Subject1"
    binfiles = glob.glob(path+"/*.bin")
    data = np.fromfile(binfiles[filenum], dtype = np.float32)
    depth_map = data.reshape(imw, imh)
    return depth_map


def get_depthmap(filenum=1, plot=False, to_cm=True):
    ## read a flattened depth map from binfiles
    ## unit of mm
    ## reshape to 240x320

    mm_to_cm = 10. ## mm to cm conversion
    imh, imw = 320, 240
    cdir = "C:/Users/juron/Documents/iWork/U-4/ENGN4200"
    path = cdir + "/handModelling/Release_2014_5_28/Subject1"
    binfiles = glob.glob(path+"/*.bin")
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
    
#    print XYZ[:,:5].T, nonzeros[:5]
    
    sumdist = XYZ[:,:-1] - XYZ[:,1:]
    avg_scale = np.average(np.linalg.norm(sumdist, axis=0))
    
    if plot:
        fig0 = plt.figure()
        ax0 = fig0.add_subplot(111)
        ax0.imshow(depth_map.T)
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(nX,nY,nZ,"b.")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.show()
    
    return depth_map, XYZ, avg_scale

def test_cv():
    start = time.time()
    img1 = cv2.imread('home.jpg',0)          # queryImage
    img2 = cv2.imread('home.jpg',0) # trainImage
    
    # Initiate SIFT detector
    sift = cv2.SIFT()
    
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    np.random.shuffle(des2)
    print "feature extraction time: ", time.time()-start    
    
    start = time.time()
    matcher = "FLANN"
    if matcher == "BFM":
        # BFMatcher with default params
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1,des2, k=1)
    elif matcher == "FLANN":
        # FLANN parameters
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)   # or pass empty dictionary
        
        flann = cv2.FlannBasedMatcher(index_params,search_params)
        
        matches = flann.knnMatch(des1,des2,k=1)
    
    print "\nexecution time: ", time.time()-start
    return 0


def load_joints_txt(plot=False):
    cdir = "C:/Users/juron/Documents/iWork/U-4/ENGN4200"
    path = cdir + "/handModelling/Release_2014_5_28/Subject1/"
    joint_txt = path + "joint.txt"
    
    out = np.loadtxt(joint_txt, skiprows=1)
    
    jpos = out.reshape(400,21,3);
    frame = jpos[0]

    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(frame[0],frame[1],frame[2],"b.")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.show()
        
    
    gpos = frame[0] # global position

    frame1 = frame[1:]
    length = []
    CMC = []
    for i in range(5):
        start = i*4
        end = 4+start
        digit1 = np.vstack((gpos, frame1[start:end]))
        digit2 = np.vstack((frame1[start:end], gpos))
        dist = np.linalg.norm(digit1-digit2, axis=1)
        length.append(dist)
        
        if i < 4:
            if i < 3:
                u, v = frame1[start] - gpos, frame1[end+1] - gpos
            else:
                u, v = frame1[start] - gpos, frame1[0] - gpos
            cosT = np.dot(u, v) / np.linalg.norm(u) / np.linalg.norm(v)
            theta = np.rad2deg(np.arccos(cosT))
            CMC.append(theta)
    
    
    handgeo = np.vstack(length)
    handgeo = handgeo[:,:-1]
    handgeo = handgeo[np.array([4,0,1,2,3])]
    
    return gpos, handgeo, CMC


def load_sphere_txt(line=0, draw=False):
    cdir = "C:/Users/juron/Documents/iWork/U-4/ENGN4200"
    path = cdir + "/handModelling/Release_2014_5_28/Subject1/"
    sphere_txt = path + "sphere.txt"
    
    data = np.loadtxt(sphere_txt, skiprows=1)
    spos = data[line].reshape(48, 5)
    
    spheres = spos[:,1:4]
    radius = spos[:,-1]
    
    if draw:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(spheres[:,0],spheres[:,1],spheres[:,2],"b.")
        ax.plot([spheres[6][0]], [spheres[6][1]], [spheres[6][2]], "r.")
        ax.plot([spheres[7][0]], [spheres[7][1]], [spheres[7][2]], "y.")
        ax.plot(spheres[-10:-6,0], spheres[-10:-6,1], spheres[-10:-6,2], "g.")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.show()
    
    prd = np.array([18.26,  15.18  ,  
                    12.1   ,  11.66  ,  11.8067,  11.9533,  
                    16.5   ,  15.0333,  13.5667,  12.1   ,
                    16.5   ,  10.56  ,  12.54  ,  14.52  ,   
                    12.1   ,  11.2933,  10.4867,  9.68   ])
    
    
    rd_idx = np.array([i + (1 if i%2==0 else -1) for i in range(48-18)])
    frd = radius[rd_idx]
    
    tb = np.hstack((prd[:2], frd[:6]))
    ix = np.hstack((prd[2:6], frd[6:12]))
    md = np.hstack((prd[6:10], frd[12:18]))
    rg = np.hstack((prd[10:14], frd[18:24]))
    lt = np.hstack((prd[14:18], frd[24:30]))
    
    radi = [tb, ix, md, rg, lt]
    radi = np.hstack(radi)
     
    
    return radi


def project_ptns(spheres, depthmap, distT, scale):
    
    f = 241.42
    imH, imW = 320, 240
    cx, cy = imH/2, imW/2
    K = np.array([f, 0, cx, 0, f, cy, 0, 0, 1]).reshape(3,3)
    
    spheres[:,1:] *= -1
    
    projection = np.dot(K, spheres.T) # shpheres.shape = (3, nspheres)
    projected = projection[:2,:] / projection[2,:] # convert to homogeneous 
    projected = np.floor(projected.T) # shape == (-1, 2)
    
#    plt.figure()
#    plt.imshow(depthmap)
#    plt.plot(projected[:,0], projected[:,1], "r.")
    

    total = 0
    rgn = spheres.shape[0]
    for i in range(rgn):        
        dx, dy = projected[i]

        xbounded = dx >= 0 and dx < depthmap.shape[0]
        ybounded = dy >= 0 and dy < depthmap.shape[1]
        
        if xbounded and ybounded:
            D_jc = depthmap[dy][dx]
            
#            print D_jc, "Djc\n"
            if D_jc != 0:
                D_cz = spheres[i,2]
                total += np.square(max(0, D_jc-D_cz))
                
                print D_cz, max(0, D_jc-D_cz), D_jc
            else:
                total += np.square(distT[dy][dx] * scale)
#                print distT[dx][dy], "\n"
        else:
            total += np.square(distT.max() * scale)


def depth_contour(filenum=1):
    depthmap, _XYZ, _scale = get_depthmap(filenum=filenum)
    depthmap = depthmap / depthmap.max() * 255.
    depthmap = np.uint8(depthmap)
    ret,thresh = cv2.threshold(depthmap,127,255,0)
    contours, hierarchy = cv2.findContours(thresh,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
    
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(len(contours)):
        contour = contours[i].reshape(-1, 2)
        ax.plot(contour[:,0], contour[:,1], "b.")
        print contour.shape
    
    plt.show()
    

def plot3d(XYZ):
    assert XYZ.shape[0] == 3, "input shape must be: (3, nptns)"
    fig = plt.figure()   
    
    ax = fig.add_subplot(111, projection="3d")
    
    ax.plot(XYZ[0,:], 
            XYZ[1,:], 
            XYZ[2,:], "k.", markersize=6, label="3d ptns")
    
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.legend(loc='upper center', 
               bbox_to_anchor=(0.5, 0.23),
               ncol=3, fancybox=True, shadow=True)
    plt.axis("off")
    fig.tight_layout()
    plt.show()


def plot_cost_full():
    
    file_dir = "data/test_full/"
    filenames = ["fitness_contour_only.dat", "fitness_contour_ptns.dat",
                 "fitness_ptns_only.dat", "fitness_uniform_sample.dat"]
    
    filenames = ["fitness_error.dat"]
    title = "Discrepency between ground truth and \noptimised model in cm"
    markers_type = ["--", ":", "-.", "-"]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    for i, name in enumerate(filenames):
        ydata = np.loadtxt(file_dir+name) / 10
        xdata = np.arange(len(ydata))
        ax.plot(xdata, ydata, label=name, linestyle=markers_type[i])
    
    plt.legend()
    ax.set_title(title)
    ax.set_xlabel("frame No.")
    ax.set_ylabel("error (cm)")
    plt.show()


#def plot_cost_func():
#    
#    nptns = 100
#    
#    data = np.loadtxt("gnd_truth_theta.txt", dtype=float)
#    data[:3] = np.deg2rad(data[:3])
#    data[6:] = np.deg2rad(data[6:])
#    
#    lowerbound = [-np.pi]*3 + [-100]*3 + [np.deg2rad(-15), 0, 0, 0]*5
#    upperbound = [np.pi]*3 + [100]*3 + [np.deg2rad(15), np.pi/2, 
#                                        np.deg2rad(110), np.pi/2]*5
#                                        
#    increment = (np.array(upperbound) - np.array(lowerbound)) / nptns
#    
#    hand = handmodel()
#    ptncloud = observation()
#    
#    radii = hand.Sradius
#    observed = ptncloud.get_depthmap(filenum=1, downsample=True)
#    disttran = ptncloud.invert_depthmap()
#    depthmap = ptncloud.depthmap
#    mapscale = ptncloud.scale
#
#    xytheta = np.zeros((26, nptns))
#    
#    for i in xrange(26):
#        
#        temp = data.copy()
#        temp[i] = lowerbound[i]
#        
#        for j in xrange(nptns):
#            
#            temp[i] += increment[i]
#            
#            centres = hand.build_hand_model(temp)
#            pcost = calcost(centres, radii, observed, 
#                            disttran, depthmap, mapscale)
#            xytheta[i, j] = pcost
#    
#    np.save("costfunc_poor", xytheta)
    



if __name__ == '__main__':
    ##############################################################
#    depth, ptncloud, scale = get_depthmap(filenum=1, plot=False)
#    spheres = np.arange(9).reshape(3,3)    
#    BcD = project_spheres(spheres, depth, scale, plot=False)
#    dpmp = load_depthmap()

#    gpos, hgeo, cmc = load_joints_txt();
#    rad = load_sphere_txt();
#    depth_contour(1)
    plot_cost_full()
    
    
    
#    depthmp = load_depthmap()
#    display_ptncloud()
    
    #######################################################
    ######################################################
    # np.set_printoptions(precision=2)
    # textfile = path+"/sphere.txt"
    # f = open(textfile, "rb")
    # cnt = 0
    # for line in f:
    #   if cnt == 3 or cnt==4:
    
    #       l = np.array(line.strip().split(" "), 
    #                       dtype = np.float32).reshape(48,-1)
    #       print l
    
    #   else:
    #       if cnt not in [0,1,2]:
    #           break
    #   cnt+=1
    ##################################################################



