# -*- coding: utf-8 -*-
"""
Created on Sun May 17 11:55:05 2015

@author: JURONG
"""

import cv2
import glob
import time
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt

from pso import pso
#from scipy import optimize
from mpl_toolkits.mplot3d import axes3d
from visualiser_Vr2 import cmp_models

np.random.seed(10000)

class handmodel(object):
    
    def __init__(self):
        self.handgeo = np.array([ 
                31.1074037 ,  27.2800217 ,  22.00017175,  21.99995952,
                68.96344533,  30.80009179,  23.10057695,  20.90010475,
                74.27700757,  36.29982272,  24.20042803,  21.99987377,
                77.44081579,  31.9002325 ,  20.90000473,  19.7995486 ,
                80.15924911,  23.10022144,  14.29931098,  15.39998521]) / 10.
                # unit mm --> cm
        self.Sradius = np.array([
                18.26  ,  15.18  ,  16.61  ,  14.96  ,  13.53  ,  12.1   ,
                10.45  ,   8.8   ,  12.1   ,  11.66  ,  11.8067,  11.9533,
                11.11  ,  10.56  ,  10.12  ,   9.68  ,   9.46  ,   9.24  ,
                16.5   ,  15.0333,  13.5667,  12.1   ,  11.55  ,  11.    ,
                10.78  ,  10.56  ,  10.01  ,   9.46  ,  16.5   ,  10.56  ,
                12.54  ,  14.52  ,  10.12  ,   9.68  ,   9.46  ,   9.24  ,
                 8.8   ,   8.36  ,  12.1   ,  11.2933,  10.4867,   9.68  ,
                 9.46  ,   9.24  ,   8.8   ,   8.36  ,   7.81  ,   7.26  ]) / 10.
                 # unit ==  mm --> cm
        
        self.handcmc = np.deg2rad([150, 107.5, 89.8, 76.5, 59.6]) # deg --> rad
        self.handspacing = [-1.86, -1.86, 0, 1.91, 3.84] # unit cm
        
#        self.handcmc = np.deg2rad([45, 80, 90, 100, 115]) # deg --> rad
#        self.handspacing = [1.1, 1.1, 0.1, -1.2, -2.1] # unit cm
    
    
    def finger_model(self, thetas, g_pos, CMC, f_geometry, 
                     spacing, gb_trans):
    
        # extract the D-H parameters for forward kinematics
        MCP1, MCP2, PIP, DIP = thetas
        L4, L5, L6, L7 = f_geometry
        ux, uy, uz = g_pos
        
        TWS, ANG, ROT = gb_trans # z, y, x
        
        c = lambda x: np.cos(x)
        s = lambda x: np.sin(x)
        
        # define the transformations    
        TWS += np.pi # the y axis is inverted ==> need to rotate z by 180deg
        Rx = np.array([1,      0,       0, 0, 
                       0, c(ROT), -s(ROT), 0,
                       0, s(ROT),  c(ROT), 0,
                       0,      0,       0, 1]).reshape(4,4)
        
        Ry = np.array([ c(ANG), 0, s(ANG), 0,
                             0, 1,      0, 0,
                       -s(ANG), 0, c(ANG), 0,
                             0, 0,      0, 1]).reshape(4,4)
        
        Rz = np.array([c(TWS), -s(TWS), 0, 0,
                       s(TWS),  c(TWS), 0, 0,
                            0,       0, 1, 0,
                            0,       0, 0, 1]).reshape(4,4)
        
        Tgb = np.dot(np.dot(Rz,Ry), Rx)
        

        T01 = np.array([c(CMC), -s(CMC), 0, L4*c(CMC),
                        s(CMC),  c(CMC), 0, L4*s(CMC),
                             0,       0, 1,         0,
                             0,       0, 0,         1]).reshape(4,4)
        
        T12 = np.array([c(MCP1),  0,-s(MCP1), 0,
                        s(MCP1),  0, c(MCP1), 0,
                              0, -1,       0, 0,
                              0,  0,       0, 1]).reshape(4,4)
        
        T23 = np.array([c(MCP2), -s(MCP2), 0, L5*c(MCP2),
                        s(MCP2),  c(MCP2), 0, L5*s(MCP2),
                              0,        0, 1,          0,
                              0,        0, 0,          1]).reshape(4,4)
    
        T34 = np.array([c(PIP), -s(PIP), 0, L6*c(PIP),
                        s(PIP),  c(PIP), 0, L6*s(PIP),
                             0,       0, 1,         0,
                             0,       0, 0,         1]).reshape(4,4)
    
        T45 = np.array([c(DIP), -s(DIP), 0, L7*c(DIP),
                        s(DIP),  c(DIP), 0, L7*s(DIP),
                             0,       0, 1,         0,
                             0,       0, 0,         1]).reshape(4,4)
    
        T00 = np.array([1, 0, 0, ux,
                        0, 1, 0, uy,
                        0, 0, 1, uz,
                        0, 0, 0,  1]).reshape(4,4)
        

        current_pos = np.dot(T00, Tgb)
        joint_pos = []
        
        a = np.sqrt(L4**2+spacing**2-2*L4*spacing*c(CMC))
        beta = np.arcsin(s(CMC)*spacing/a)
    
        # print a, beta, spacing, "a, beta, spacing"
    
        T10 = np.array([c(beta), -s(beta), 0, -L4*s(CMC)*c(beta),
                        s(beta),  c(beta), 0, -L4*s(CMC)*s(beta),
                             0,       0, 1,         0,
                             0,       0, 0,         1]).reshape(4,4)

        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection="3d")
        # opos = np.array(g_pos)
        T123 = np.dot(T12,T23)
        cnt = 0
        for transform in [T01, T123, T34, T45]:
            if cnt == 1:
                temp = np.dot(current_pos, T10)
                joint_pos.insert(0, temp[:,-1][:-1])
            
            current_pos = np.dot(current_pos, transform)
            
            print current_pos
            npos = current_pos[:,-1][:-1]
            joint_pos.append(npos)
            
            cnt += 1
    

        return np.vstack(joint_pos)
    
    def thumb_model(self, theta_ij, g_pos, CMC, tb_geometry, 
                    spacing, gb_trans):
    
        TMC1, TMC2, MCP, IP = theta_ij
        L0, L1, L2, L3 = tb_geometry
        ux, uy, uz = g_pos
    
        c = lambda x: np.cos(x)
        s = lambda x: np.sin(x)
        
        TWS, ANG, ROT = gb_trans # z, y, x
        
        TWS += np.pi # the y axis is inverted ==> need to rotate z by 180deg
        Rx = np.array([1,      0,       0, 0, 
                       0, c(ROT), -s(ROT), 0,
                       0, s(ROT),  c(ROT), 0,
                       0,      0,       0, 1]).reshape(4,4)
        
        Ry = np.array([ c(ANG), 0, s(ANG), 0,
                             0, 1,      0, 0,
                       -s(ANG), 0, c(ANG), 0,
                             0, 0,      0, 1]).reshape(4,4)
        
        Rz = np.array([c(TWS), -s(TWS), 0, 0,
                       s(TWS),  c(TWS), 0, 0,
                            0,       0, 1, 0,
                            0,       0, 0, 1]).reshape(4,4)
        
        Tgb = np.dot(np.dot(Rz,Ry), Rx)
    
    
        Trf = np.array([c(CMC), -s(CMC), 0, L0*c(CMC),
                        s(CMC),  c(CMC), 0, L0*s(CMC),
                             0,       0, 1,         0,
                             0,       0, 0,         1]).reshape(4,4)
    
        T01 = np.array([c(TMC1), 0,-s(TMC1), 0,
                        s(TMC1), 0, c(TMC1), 0,
                        0      ,-1,       0, 0,
                        0      , 0,       0, 1]).reshape(4,4)
    
    
        # need to rotate the x-axis back by CMC to ensure correct motion of thumb
        cMc = CMC-np.pi # pCMC in C++ implementation
        T12 = np.array([
            c(TMC2), -s(TMC2)*c(cMc), s(TMC2)*s(cMc), L1*c(TMC2),
            s(TMC2),  c(TMC2)*c(cMc),-c(TMC2)*s(cMc), L1*s(TMC2),
                  0,          s(cMc),         c(cMc),          0,
                  0,               0,              0,          1]).reshape(4,4)
    
        # 0deg rotation of x-axis
        T23 = np.array([c(MCP), -s(MCP), 0, L2*c(MCP),
                        s(MCP),  c(MCP), 0, L2*s(MCP),
                             0,       0, 1,     0,
                             0,       0, 0,     1]).reshape(4,4)
        
        
        T34 = np.array([c(IP), -s(IP), 0, L3*c(IP),
                        s(IP),  c(IP), 0, L3*s(IP),
                            0,      0, 1,        0,
                            0,      0, 0,        1]).reshape(4,4)
    
        T00 = np.array([1, 0, 0, ux,
                        0, 1, 0, uy,
                        0, 0, 1, uz,
                        0, 0, 0,  1]).reshape(4,4)

        
        current_pos = np.dot(T00, Tgb)
        joint_pos = []
        
        a = np.sqrt(L0**2+spacing**2-2*L0*spacing*c(CMC))
        beta = np.arcsin(s(CMC)*spacing/a)
        T10 = np.array([c(beta), -s(beta), 0, -a*c(beta),
                        s(beta),  c(beta), 0, -a*s(beta),
                             0,       0, 1,         0,
                             0,       0, 0,         1]).reshape(4,4)
        
        T012 = np.dot(T01, T12)
        cnt = 0
        for transform in [Trf, T012, T23, T34]:
            if cnt == 1:
                temp = np.dot(current_pos, T10)
                joint_pos.insert(0, temp[:,-1][:-1])
                
            current_pos = np.dot(current_pos, transform)
            npos = current_pos[:,-1][:-1]
    
            joint_pos.append(npos)
            
            cnt += 1
    
        return np.vstack(joint_pos)
        

    def build_hand_model(self, hand_params):
        """
            hand_params: a flattened array encodes the angles of each joint; 
                         i.e. [Thumb, Index, Middle, Ring, Little]
                         Thumb: 4 parameters
                         Index etc.: 5 parameters
                         len(hand_params) = 24
    
            hand_geometry: a flattened array of the lengths of each bone; 
                           i.e. [Thumb, Index, Middle, Ring, Little]
                           Thumb: 3 parameters
                           Index etc.: 4 parameters
                           len(hand_geometry) = 19
    
            ref_pos: an array of the x, y, z position of the reference;
                     i.e. 3D position of the wrist
        """
    
        hand_geometry = self.handgeo
        
        
        global_T = hand_params[:3] # 0, 1, 2 ==> rotation in x, y, z
        ref_pos = hand_params[3:6] # 3, 4, 5 ==> global (x, y, z) postion
        
        thumb_the = hand_params[6:10] # 6, 7, 8, 9 ==> TMC1, TMC2, MCP, IP
        thumb_geo = hand_geometry[:4]
    
        index_the = hand_params[10:14] # 10, 11, 12, 13 ==> MCP1, MCP2, PIP, DIP
        index_geo = hand_geometry[4:8]
    
        middle_the = hand_params[14:18]
        middle_geo = hand_geometry[8:12]
    
        ring_the = hand_params[18:22]
        ring_geo = hand_geometry[12:16]
    
        little_the = hand_params[22:] # 22, 23, 24, 25
        little_geo = hand_geometry[16:]
        

        thumb = self.thumb_model(thumb_the, ref_pos, self.handcmc[0],
                                 thumb_geo, self.handspacing[0], 
                                 global_T)
                                 
        index = self.finger_model(index_the, ref_pos, self.handcmc[1],
                                  index_geo, self.handspacing[1], 
                                  gb_trans=global_T)
                                  
        middle = self.finger_model(middle_the, ref_pos, self.handcmc[2],
                                   middle_geo, spacing=self.handspacing[2], 
                                   gb_trans=global_T)

        ring = self.finger_model(ring_the, ref_pos, self.handcmc[3],
                                 ring_geo, spacing=self.handspacing[3],
                                 gb_trans=global_T)
                                 
        little = self.finger_model(little_the, ref_pos, self.handcmc[4],
                                   little_geo, spacing=self.handspacing[4], 
                                   gb_trans=global_T)

        joints = [thumb, index, middle, ring, little]
        
#        fig = plt.figure();
#        ax = fig.add_subplot(111, projection="3d")
#        
#        for parts in joints:
#            X, Y, Z = parts.T
#            ax.plot(X,Y,Z,marker="o")
#        
#        plt.show()
        
        centers = self.build_spheres2(joints, draw=False)        
        
        return centers

    

    def build_spheres2(self, hand_joints, fg_num=[4,2,2,2], 
                       tb_num=[2,2,2,2], 
                       draw = False):
        
        tb, ix, md, rg, lt = hand_joints
        
        centers = []
        
        for i in range(len(tb)-1):
            joint1, joint2 = tb[i], tb[i+1]
            t = 1./tb_num[i]
            
            for j in range(1, tb_num[i]+1):
                cen = (1-t*j)*joint1 + t*j*joint2
                centers.append(cen)
        
        for elem in [ix, md, rg, lt]:
            for i in range(len(ix)-1):
                joint1, joint2 = elem[i], elem[i+1]
                if i == 0:
                    t = 1./(fg_num[i]-1)
                    for j in range(fg_num[i]):
                        cen = (1-t*j)*joint1  + t*j*joint2
                        centers.append(cen)
                else:
                    t = 1./(fg_num[i])
                    for j in range(1, fg_num[i]+1):
                        cen = (1-t*j)*joint1 + t*j*joint2
                        centers.append(cen)
        
        centers = np.vstack(centers)
        
        # need to ensure the coordinate system is consistent
        centers[:,1:] *= -1;
        
        if draw:
            
            ux, uy, uz = centers[0]
            sX, sY, sZ = centers.T
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            ax.plot(sX, sY, sZ, "b.")
            
            ax.set_autoscale_on(False)
            plt.show()
        
        return centers


class observation(object):
    
    def __init__(self):
        self.cdir = "C:/Users/juron/Documents/iWork/U-4/ENGN4200/" + \
                    "handModelling/Release_2014_5_28/Subject1/"
                    
        self.scale = 0
        self.depthmap = None
    
    def get_depthmap(self, filenum=1, downsample=False, 
                     plot=False, to_cm=True):
        ## read a flattened depth map from binfiles
        ## unit of mm
        ## reshape to 240x320
    
        mm_to_cm = 10. ## mm to cm conversion
        imh, imw = 320, 240
        binfiles = glob.glob(self.cdir+"/*.bin")
        data = np.fromfile(binfiles[filenum], dtype = np.float32)
        
        depth_map = data.reshape(imw, imh) / mm_to_cm
        
        self.depthmap = depth_map
        
    #    plt.figure()
    #    plt.imshow(depth_map)
    
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
        nY = Y[nonzeros] * -1
        nZ = Z[nonzeros] * -1
        
        XYZ = np.vstack((nX, nY, nZ))
        
#        sumdist = XYZ[:,:-1] - XYZ[:,1:]
#        self.scale = np.average(np.linalg.norm(sumdist, axis=0))
        
        cx, cy = imh/2, imw/2
        K = np.array([constant, 0, cx, 
                      0, constant, cy, 
                      0, 0, 1]).reshape(3,3)
        
        rad     = 2.
        sph_cen = np.vstack((X[nonzeros],Y[nonzeros],Z[nonzeros]))
        X_trans = X[nonzeros] + rad
        sph_edg = np.vstack((X_trans,Y[nonzeros],Z[nonzeros]))
        
        projection = np.dot(K, sph_cen) # shpheres.shape = (3, nspheres)
        projected1 = projection[:2,:] / projection[2,:] # convert to homogeneous 
        projected1 = np.floor(projected1.T) # shape == (-1, 2)
        
        projection = np.dot(K, sph_edg)
        projected2 = projection[:2,:] / projection[2,:]
        projected2 = np.floor(projected2.T)
        
        distnorm = np.sqrt(np.square(projected1[:,0] - projected2[:,0]) + 
                           np.square(projected1[:,1] - projected2[:,1]))
        
        nzero,     = np.nonzero(distnorm)       
        distnorm   = distnorm[nzero]
        cmPerPixel = rad / distnorm
        
        
        self.scale = np.average(cmPerPixel)
        
        if plot:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            ax.plot(nX,nY,nZ,"b.")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
    #        ax.axis("off")
            plt.show()
        
        if downsample:
            np.random.shuffle(XYZ.T)
            XYZ = XYZ[:,:256]
        
        return XYZ
    
    
    def invert_depthmap(self, filenum=1, imh=320, imw=240, dplot=False):

        mm_to_cm = 10. 
        binfiles = glob.glob(self.cdir+"/*.bin")
        data = np.fromfile(binfiles[filenum], dtype = np.float32)
        
        depth = data / mm_to_cm
        zeros = np.where(depth==0)
        nonzeros = np.where(depth!=0)
        depth[zeros] = 1.
        depth[nonzeros] = 0.
        inverted_depth = np.int8(depth.reshape(imw, imh))
        
        if dplot:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.imshow(depth)
            plt.show()
        
        dist_transform = cv2.distanceTransform(inverted_depth,
                                               cv2.cv.CV_DIST_L2, 5)
                    
#        plt.imshow(dist_transform)
#        print dist_transform.shape
        
        return dist_transform
    
    
    def read_spheres_txt(self, line=1, dplot=False):
        sphere_txt = self.cdir + "sphere.txt"
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
            
        return spheres # (48, 3)
    
    
def gen2dprojection(spheres, radii, scale, depthmap):
    
    # compute projection
    f = 241.42
    imH, imW = 320, 240
    cx, cy = imH/2, imW/2
    K = np.array([f, 0, cx, 0, f, cy, 0, 0, 1]).reshape(3,3)
    
    spheres[:,1:] *= -1
    
    projection = np.dot(K, spheres.T) # shpheres.shape = (3, nspheres)
    projected = projection[:2,:] / projection[2,:] # convert to homogeneous 
    projected = np.int32(np.floor(projected.T)) # shape == (-1, 2)
    
    # scale unit --> cm/pixel; need to convert radii from cm --> pixel
    radii /= scale 
    
    nptns, _d = projected.shape

    depthmap = np.zeros(depthmap.shape)    
    
    for i in xrange(nptns):
        
        dep = spheres[i,2]
        cen = projected[i]
        rad = int(radii[i])
        
        cv2.circle(depthmap, (cen[0], cen[1]), rad, (dep), -1)

    
    plt.figure()
    plt.imshow(depthmap)
    
    return depthmap


def calcost(spheres, radii, observed, distT, depthmap, scale):
    
    ## compute alignment cost
    observed = np.float32(observed).T
    spheres = np.float32(spheres)
    
    matches = []
    matcher_type = "brute_force"
    
    if matcher_type == "brute_force":
        bf = cv2.BFMatcher()
        
#        start = time.time()
        matches = bf.knnMatch(observed,spheres,k=1)
#        print time.time()-start
    
    elif matcher_type == "flann":
        # FLANN parameters
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 4)
        search_params = dict(checks=64)   # or pass empty dictionary
        
        flann = cv2.FlannBasedMatcher(index_params,search_params)
        
#        start = time.time()
        matches = flann.knnMatch(observed,spheres,k=1)
#        print "exec time = ", time.time()-start, "sec"
    else:
        print "matcher_type must be 'flann' or 'brute_force'"
    
    dist, index = [], []
    
    for match, in matches:
        idx = match.trainIdx
        index.append(idx)
        tdis = abs(match.distance - radii[idx])
        dist.append(tdis**2) # sum of distance squared
    
    
    # constant lambda scaling of the first term in cost function
    const = float(spheres.shape[0]) / observed.shape[0]
    align_cost = sum(dist) * const # apply the scaling
    
    
    ## compute depth projection cost
    f = 241.42
    imH, imW = 320, 240
    cx, cy = imH/2, imW/2
    K = np.array([f, 0, cx, 0, f, cy, 0, 0, 1]).reshape(3,3)
    
    spheres[:,1:] *= -1
    
    projection = np.dot(K, spheres.T) # shpheres.shape = (3, nspheres)
    projected = projection[:2,:] / projection[2,:] # convert to homogeneous 
    projected = np.floor(projected.T) # shape == (-1, 2)
    
    total = 0
    rgn = spheres.shape[0]
    for i in range(rgn):
        dx, dy = projected[i]
        xbounded = dx >= 0 and dx < depthmap.shape[1]
        ybounded = dy >= 0 and dy < depthmap.shape[0]
        
        if xbounded and ybounded:
            D_jc = depthmap[dy][dx]
            
#            print D_jc, "Djc\n"
            if D_jc != 0:
                D_cz = spheres[i,2]
                total += np.square(max(0, D_jc-D_cz))
                
#                print D_cz, max(0, D_jc-D_cz), "\n"
            else:
                total += np.square(distT[dy][dx] * scale)
#                print distT[dx][dy], "\n"
        else:
            total += np.square(distT.max() * scale)
#            print "out of bound"
    
    
    ## compute collision cost    
    tbs = spheres[2:8,:]
    ixs = spheres[12:18,:]
    mds = spheres[22:28,:]
    rgs = spheres[32:38,:]
    lts = spheres[42:,:]
    
    tbr = radii[2:8]
    ixr = radii[12:18]
    mdr = radii[22:28]
    rgr = radii[32:38]
    ltr = radii[42:]
    
    colspheres = [tbs, ixs, mds, rgs, lts]
    colradii = [tbr, ixr, mdr, rgr, ltr]
    collision_cost = 0
    
    for i in range(4):

        mat1 = np.matlib.repmat(colspheres[i], 1, 6).reshape(-1, 3)
        mat2 = np.matlib.repmat(colspheres[i+1], 6, 1)

        vec1 = np.matlib.repmat(colradii[i], 6, 1).T.flatten()
        vec2 = np.matlib.repmat(colradii[i+1], 1, 6)
        
        diff = mat2 - mat1
        dist = vec1 + vec2 - np.linalg.norm(diff, axis=1)
        
        ind = np.where(dist > 0)
        
        collision_cost += np.sum(dist[ind]**2)
        
    
#    print total, align_cost, collision_cost
    
    return total + align_cost + collision_cost

    
    
def psocost(array, *args):
    particle, radii, observed, disttran, depthmap, mapscale = args
    centres = particle.build_hand_model(array)
    
    cost = calcost(centres, radii, observed, disttran, depthmap, mapscale)
    
    return cost

def show_cpp_data(linenum=1, fname="d.txt", ftitle=""):
    data = np.loadtxt(fname, dtype=float)
    data[:3] = np.deg2rad(data[:3])
    data[6:] = np.deg2rad(data[6:])
    
    ptncldfile="data/frame_000"+str(linenum)+"/ptncld.dat"
    hand = handmodel()
    cens = hand.build_hand_model(data)
    cmp_models(line=linenum, spheres_pos=cens.T, 
               XYZ_file=ptncldfile, figtitle=ftitle)



def main():
    hand = handmodel()
    ptncloud = observation()
    
    tbT = np.deg2rad([6, 9, 8, 9])
    ixT = np.deg2rad([3, 9, 9, 6])
    meT = np.deg2rad([1, 9, 8, 7])
    rgT = np.deg2rad([4, 8, 7, 6])
    leT = np.deg2rad([2, 7, 7, 7])
    
    gpos = np.array([0, 3, 32]) #[0, 3, 32]
    global_trans = np.deg2rad([0,-10,-40]) #[0,-10,-40]

    the_params = np.hstack((global_trans, gpos, tbT, ixT, meT, rgT, leT))
    
    
    centres = hand.build_hand_model(the_params)
    radii = hand.Sradius
    observed = ptncloud.get_depthmap(filenum=1, downsample=True)
    disttran = ptncloud.invert_depthmap()
    depthmap = ptncloud.depthmap
    mapscale = ptncloud.scale
    gndtruth = ptncloud.read_spheres_txt()


    pro = gen2dprojection(centres,radii,mapscale,depthmap) 
    
    diff = np.sum(abs(pro-depthmap))
    print pro.min(), pro.max(), depthmap.min(), depthmap.max(), diff
    
    
#    print centres.shape
#    ptns = observed[:,:48].T
#    project_ptns(ptns, depthmap)
    
    
#    calcost(gndtruth, radii, observed, disttran, depthmap, mapscale)
##    
#    args = (hand, radii, observed, disttran, depthmap, mapscale)
#    
#    lowerbound = [-np.pi]*3 + [-100]*3 + [np.deg2rad(-15), 0, 0, 0]*5
#    upperbound = [np.pi]*3 + [100]*3 + [np.deg2rad(15), np.pi/2, 
#                                        np.deg2rad(110), np.pi/2]*5
#    std = [np.deg2rad(5)]*3 + [1.5]*3 + [np.deg2rad(5)]*4*5    
#    
##    phip = 2.8
##    phig = 1.3
##    thi = phip + phig
##    omega = 2./abs(2-(thi)-np.sqrt(thi**2-4*thi))
#
#    omega = 0.7298
#    phip = 1.49618
#    phig = 1.49618
#    
#    xopt, fopt = pso(psocost, the_params, std, 
#                     lowerbound, upperbound, args=args,
#                     swarmsize=30, omega=omega, phip=phip, phig=phig, 
#                     maxiter=20, minstep=1e-6, minfunc=1e-6, debug=False)
##
###    guess = np.array([ 
###         3.08148691e+00,  -1.60408594e-01,  -7.65264686e-01,
###        -2.18612734e-03,   4.01741190e+00,   3.20000014e+01,
###         1.04720945e-01,   1.57080005e-01,   1.39626340e-01,
###         1.57079633e-01,   5.23574172e-02,   1.57053159e-01,
###         1.57077607e-01,   1.04719755e-01,   1.93041386e-02,
###         1.56821844e-01,   1.39660332e-01,   1.22173048e-01,
###         6.66860632e-02,   1.55828265e-01,   1.37938256e-01,
###         1.20485920e-01,   2.43669020e-02,   1.41829211e-01,
###         1.22173156e-01,   1.22173048e-01])
###
###    xopt = optimize.fmin_cg(psocost, guess, args=args)
##
#    new_cen = hand.build_hand_model(xopt)
#    cmp_models(spheres_pos=new_cen.T, XYZ_pos=observed)
#    print fopt
#    
#    return xopt, the_params
#    
#    rot = init_pose(observed, global_trans)
#    newcen = np.dot(centres, rot)
#    cmp_models(spheres_pos=newcen.T)
#    
#    return centres
    



