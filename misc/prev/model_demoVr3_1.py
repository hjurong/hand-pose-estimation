import cv2
import numpy as np
import matplotlib.pyplot as plt

from read_binary import get_depthmap
from mpl_toolkits.mplot3d import axes3d
from matplotlib.widgets import Slider, Button, RadioButtons

#http://stackoverflow.com/questions/26973970/conversion-between-cvmat-and-armamat

def finger_model(thetas, g_pos, f_geometry, 
                 spacing = 0.5, gb_trans = np.deg2rad([30, 30])):

    # extract the D-H parameters for forward kinematics
    CMC, MCP1, MCP2, PIP, DIP = thetas
    L4, L5, L6, L7 = f_geometry
    ux, uy, uz = g_pos
    
    TWS, ANG = gb_trans    
    
    c = lambda x: np.cos(x)
    s = lambda x: np.sin(x)
    
    # define the transformations
    # T01 = np.array([c(CMC),  0,-s(CMC),L4*c(CMC),
    #                 s(CMC),  0, c(CMC),L4*s(CMC),
    #                      0,  1,      0,        0,
    #                      0,  0,      0,        1]).reshape(4,4)
    # T12 = np.array([c(MCP1),  0, s(MCP1), 0,
    #                 s(MCP1),  0,-c(MCP1), 0,
    #                       0, -1,       0, 0,
    #                       0,  0,       0, 1]).reshape(4,4)
    
    Tgb = np.array([c(TWS), -s(TWS)*c(ANG), s(TWS)*s(ANG), 0,
                    s(TWS),  c(TWS)*c(ANG),-c(TWS)*s(ANG), 0,
                          0,        s(ANG),        c(ANG), 0,
                          0,             0,             0, 1]).reshape(4,4)

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
    
    
#    current_pos = T00
#    joint_pos = [g_pos]
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
        
        npos = current_pos[:,-1][:-1]
        # print current_pos, "\n\n", transform, npos, "------------------\n"
        joint_pos.append(npos)
        
        cnt += 1

        # X, Y, Z = np.vstack([opos, npos]).T
        # print X, Y, Z, "/n"
       
        # ax.plot(X,Y,Z)
        # opos = npos
        # print current_pos
    
    # print np.vstack(joint_pos)
    return np.vstack(joint_pos)

def thumb_model(theta_ij, g_pos, tb_geometry, 
                spacing = 0.5, gb_trans = np.deg2rad([30, 30])):

    CMC, TMC1, TMC2, MCP, IP = theta_ij
    L0, L1, L2, L3 = tb_geometry
    ux, uy, uz = g_pos

    c = lambda x: np.cos(x)
    s = lambda x: np.sin(x)
    
    TWS, ANG = gb_trans

    Tgb = np.array([c(TWS), -s(TWS)*c(ANG), s(TWS)*s(ANG), 0,
                    s(TWS),  c(TWS)*c(ANG),-c(TWS)*s(ANG), 0,
                          0,        s(ANG),         c(ANG), 0,
                          0,                0,               0, 1]).reshape(4,4)

    Trf = np.array([c(CMC), -s(CMC), 0, L0*c(CMC),
                    s(CMC),  c(CMC), 0, L0*s(CMC),
                         0,       0, 1,         0,
                         0,       0, 0,         1]).reshape(4,4)

    T01 = np.array([c(TMC1), 0,-s(TMC1), 0,
                    s(TMC1), 0, c(TMC1), 0,
                    0      ,-1,       0, 0,
                    0      , 0,       0, 1]).reshape(4,4)


    # T12 = np.array([c(TMC2), -s(TMC2), 0, L1*c(TMC2),
    #                 s(TMC2),  c(TMC2), 0, L1*s(TMC2),
    #                       0,        0, 1,          0,
    #                       0,        0, 0,          1]).reshape(4,4)

    # need to rotate the x-axis back by CMC to ensure correct motion of thumb
    T12 = np.array([c(TMC2), -s(TMC2)*c(CMC), s(TMC2)*s(CMC), L1*c(TMC2),
                    s(TMC2),  c(TMC2)*c(CMC),-c(TMC2)*s(CMC), L1*s(TMC2),
                          0,          s(CMC),         c(CMC),          0,
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

#    current_pos = T00
#    joint_pos = [g_pos]
    
    current_pos = np.dot(T00, Tgb)
    joint_pos = []
    
    a = np.sqrt(L0**2+spacing**2-2*L0*spacing*c(CMC))
    beta = np.arcsin(s(CMC)*spacing/a)
    T10 = np.array([c(beta), -s(beta), 0, -L0*s(CMC)*c(beta),
                    s(beta),  c(beta), 0, -L0*s(CMC)*s(beta),
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



def hand_model(hand_params, hand_geometry, ref_pos, global_T):
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

    thumb_the = hand_params[:5]
    thumb_geo = hand_geometry[:4]

    index_the = hand_params[5:10]
    index_geo = hand_geometry[4:8]

    middle_the = hand_params[10:15]
    middle_geo = hand_geometry[8:12]

    ring_the = hand_params[15:20]
    ring_geo = hand_geometry[12:16]

    little_the = hand_params[20:]
    little_geo = hand_geometry[16:]

    thumb = thumb_model(thumb_the, ref_pos, 
                        thumb_geo, spacing=2.3, gb_trans=global_T)
    index = finger_model(index_the, ref_pos, 
                         index_geo, spacing=1.1, gb_trans=global_T)
    middle = finger_model(middle_the, ref_pos, 
                          middle_geo, spacing=0.1, gb_trans=global_T)
    ring = finger_model(ring_the, ref_pos, 
                        ring_geo, spacing=-1.2, gb_trans=global_T)
    little = finger_model(little_the, ref_pos, 
                          little_geo, spacing=-2.1, gb_trans=global_T)

#    fig = plt.figure()
#    ax = fig.add_subplot(111, projection="3d")
#    ax.set_xlabel("X")
#    ax.set_ylabel("y")
#    ax.set_zlabel("z")
#    def visualise(joints, ax1):
#        X, Y, Z = joints.T
#        ax1.plot(X,Y,Z,marker="o")
#    
#    for parts in [thumb, index, middle, ring, little]:
#        visualise(parts, ax)
#
#    X = ax.get_xlim()
#    Y = ax.get_ylim()
#    Z = ax.get_zlim()
#    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0
#    mean_x = X.mean()
#    mean_y = Y.mean()
#    mean_z = Z.mean()
#    ax.set_xlim(mean_x - max_range, mean_x + max_range)
#    ax.set_ylim(mean_y - max_range, mean_y + max_range)
#    ax.set_zlim(mean_z - max_range, mean_z + max_range)
#    
#    def update():
#        pass
#    axfreq = plt.axes([0.25, 0.1, 0.65, 0.03], axisbg='lightgoldenrodyellow')
#    sfreq = Slider(axfreq, 'Freq', 0.1, 30.0, valinit=3)
#    sfreq.on_changed()

    return [thumb, index, middle, ring, little]

def build_spheres(hand_joints, fg_sphereNum=[4,2,2,2], tb_sphereNum=[2,2,2,2],
                  drw_spheres=False):
    """
    hand_joints: list of arrays containing the joints of each finger
    spgereNum: sphere distribution ==> 
               4 between first two joints, 2 the next two etc.
    """

    def gen_midpnt_eqn(joint1, joint2, numSpheres):
        """
        generalised midpoint formula

        joint1: a,b,c
        joint2: x,y,z
        """

        a,b,c = joint1
        x,y,z = joint2 
        t = 1./(numSpheres*2)

        radius = np.linalg.norm((joint2-joint1)/(numSpheres)/2.)
        sph_cen, sph_rdi = [], [radius]*numSpheres

        for i in range(1,numSpheres+1):
            mapping = (i-1)*2+1
            nx = (1-t*mapping)*a+t*mapping*x
            ny = (1-t*mapping)*b+t*mapping*y
            nz = (1-t*mapping)*c+t*mapping*z
            sph_cen.append((nx,ny,nz))

            

        return sph_cen, sph_rdi

    # thumb, index, middle, ring, little = hand_joints
    centers, radii = [], []
    

    for cnt, part in enumerate(hand_joints):
        j = 0
        if cnt == 0: 
            sphereNum = tb_sphereNum
        else:
            sphereNum = fg_sphereNum
        for i in range(len(part)-1):
            cen, rdi = gen_midpnt_eqn(part[i], part[i+1], sphereNum[j])
            centers += cen
            radii += rdi
            j+=1
    
    
    centers = np.array(centers)
    radii = np.array(radii)
    
    def draw_spheres(centers_array, radii_array):
#        size = 4./3*np.pi*np.power(radii_array, 3)*100
        ux, uy, uz = centers_array[0]
        sX, sY, sZ = centers_array.T
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(sX, sY, sZ, "b.", markersize=20)
        ax.set_autoscale_on(False)
        ax.set_xlim((ux-20, ux+20))
        ax.set_ylim((uy-20, uy+20))
        ax.set_zlim((uz-20, uz+20))
        plt.show()
        return 0
    
    if drw_spheres:
        draw_spheres(centers, radii)
    
    return centers, radii



def visualise_hand_model(hand_params, hand_geo, ref, gtrans):
    """
        hand_joints: type = list, 
                     a list of arrays which contain the coordinates of every joint for each finger 
                     i.e. [thumb, index, middle, ring, little]
    """
    hand_joints = hand_model(hand_params, hand_geo, ref, global_T=gtrans)
    
#    print hand_joints[1:]
    span = 15
    ux, uy, uz = ref # i.e. gpos
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_autoscale_on(False)
    ax.set_xlim((ux-span, ux+span))
    ax.set_ylim((uy-span, uy+span))
    ax.set_zlim((uz-span, uz+span))
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
#    ax.axis("off")
    def draw_model(hand):
        ax.cla()
        ax.set_autoscale_on(False)
        ax.set_xlim((ux-span, ux+span))
        ax.set_ylim((uy-span, uy+span))
        ax.set_zlim((uz-span, uz+span))
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        for parts in hand:
            X, Y, Z = parts.T
            ax.plot(X,Y,Z,marker="o")
        
        plt.pause(0.01)
        #fig.canvas.draw_idle()

#        u = np.linspace(0, np.pi, 30)
#        v = np.linspace(0, 2*np.pi, 30)

#        x = np.outer(np.sin(u), np.sin(v))
#        y = np.outer(np.sin(u), np.cos(v))
#        z = np.outer(np.cos(u), np.ones_like(v))
    
    draw_model(hand_joints)
    
    axfreq = plt.axes([0.25, 0.1, 0.65, 0.03], axisbg='lightgoldenrodyellow')
    sfreq = Slider(axfreq, '2DOF', 0.1, 80.0, valinit=0)
    axamp = plt.axes([0.25, 0.15, 0.65, 0.03], axisbg='lightgoldenrodyellow')
    samp = Slider(axamp, '1DOF', -15, 15.0, valinit=0)
    
    def update(val):
        chand_params = hand_params.copy()
        chand_params = chand_params.reshape(5,-1)

        chand_params[:,2:] += np.deg2rad(sfreq.val)
        chand_params[:,1] += np.deg2rad(samp.val)


        nhand_joints = hand_model(chand_params.flatten(), hand_geo, ref)
        

        draw_model(nhand_joints)
#        ax.plot([3],[3],[sfreq.val], marker="o", color="k")
        plt.draw()
        

    sfreq.on_changed(update)
    samp.on_changed(update)
    

    return 0

def match_models(spheres_center, radii, plot=False):
    ## spheres_center has unit cm
    ptncloud = get_depthmap() ## shape = (3, nptn) --> need to take transpose
    ## convert to float32 as required by the BFMatcher
    ptncloud = np.float32(ptncloud).T
    spheres_center = np.float32(spheres_center)
    
#    print ptncloud.dtype, spheres_center.dtype
#    print ptncloud.shape, spheres_center.shape
    
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(ptncloud,spheres_center, k=1)
    
    dist, index = [], []
    
    for match, in matches:
        idx = match.trainIdx
        index.append(idx)
        tdis = abs(match.distance - radii[idx])
        dist.append(tdis)
    
    total = sum(dist)
    
    if plot:
        span = 12
        ux, uy, uz = spheres_center[0]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.set_autoscale_on(False)
        ax.set_xlim((ux-span, ux+span))
        ax.set_ylim((uy-span, uy+span))
        ax.set_zlim((uz-span, uz+span))
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.plot(spheres_center[:,0], spheres_center[:,1], 
                spheres_center[:,2], "b.")
        ax.plot(ptncloud[:,0], ptncloud[:,1], ptncloud[:,2], "g.")
        plt.show()
    
    return total, index

if __name__ == '__main__':
    
    thetas = np.deg2rad([5,40,10,10,10])
    geometry = np.linspace(5,5,4)
#    gpos = [5,0,5]
    gpos = [0, 4, 32]
    global_trans = np.deg2rad([90,-40])
    # tb_theta = np.deg2rad([45,50,75,75])
    # tb_geo = np.linspace(3,3,3)
    # f1_joints = finger_model(thetas, gpos, geometry)
    # tb_joints = thumb_model(tb_theta, gpos, tb_geo)
    # print f1_joints, tb_joints

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection="3d")
    # X1, Y1, Z1 = f1_joints.T 
    # X2, Y2, Z2 = tb_joints.T
    # ax.plot(X1, Y1, Z1)
    # ax.plot(X2, Y2, Z2)
    # plt.show()
    
    
    tbT = np.deg2rad([45, 10, 5, 6, 5])
    ixT = np.deg2rad([80, 3, 5, 6, 3])
    meT = np.deg2rad([90, 1, 2, 3, 4])
    rgT = np.deg2rad([100, 4, 8, 7, 6])
    leT = np.deg2rad([115, 2, 7, 7, 5])

    # tbG = np.array([2.5,1.5,1.5,1])
    # ixG = np.array([5,2.5,2,1])
    # meG = np.array([5,3,2,1])
    # rgG = np.array([5,2.5,2,1])
    # leG = np.array([5,1.5,1,0.7])

    tbG = np.array([3.6,3.7,3.6,2.8])
    ixG = np.array([7.8,4.7,2.7,2.4])
    meG = np.array([8.1,5.0,3.1,2.5])
    rgG = np.array([7.8,4.7,2.7,2.4])
    leG = np.array([7.7,3.5,2.3,2.2])

    the_params = np.hstack((tbT, ixT, meT, rgT, leT))
    geo_params = np.hstack((tbG, ixG, meG, rgG, leG))
    
    # a = ""
    # for elem in geo_params: 
    #    a = a + " << " + str(elem)
    # print a

    joints = hand_model(the_params, geo_params, gpos, global_trans)
    # print joints
    cen, rdi = build_spheres(joints, drw_spheres=False)
    # print cen, rdi
    tdist, idx = match_models(cen, rdi, plot=False)
    print tdist

#    visualise_hand_model(the_params, geo_params, gpos, gtrans=global_trans)
#    plt.show()

#    fparam = finger_model(tbT, gpos, tbG)
#    print(fparam)
#
#    print build_spheres([fparam])



