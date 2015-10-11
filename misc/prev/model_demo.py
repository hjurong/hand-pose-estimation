import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import axes3d


def finger_model(thetas, g_pos, f_geometry):

    # extract the D-H parameters for forward kinematics
    CMC, MCP1, MCP2, PIP, DIP = thetas
    L4, L5, L6, L7 = f_geometry
    ux, uy, uz = g_pos

    c = lambda x: np.cos(x)
    s = lambda x: np.sin(x)
    
    # define the transformations
    T01 = np.array([c(CMC),  0,-s(CMC),L4*c(CMC),
                    s(CMC),  0, c(CMC),L4*s(CMC),
                         0,  1,      0,        0,
                         0,  0,      0,        1]).reshape(4,4)

    T12 = np.array([c(MCP1),  0, s(MCP1), 5,
                    s(MCP1),  0,-c(MCP1), 5,
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
    
    
    current_pos = T00
    joint_pos = [g_pos]

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection="3d")
    # opos = np.array(g_pos)

    for transform in [T01, T12, T23, T34, T45]:
        current_pos = np.dot(current_pos, transform)

        npos = current_pos[:,-1][:-1]
        joint_pos.append(npos)

        # X, Y, Z = np.vstack([opos, npos]).T
        # print X, Y, Z, "/n"
       
        # ax.plot(X,Y,Z)
        # opos = npos
        # print current_pos
    
    # print np.vstack(joint_pos)
    return np.vstack(joint_pos)

def thumb_model(theta_ij, g_pos, tb_geometry):
	TMC1, TMC2, MCP, IP = theta_ij
	L1, L2, L3 = tb_geometry
	ux, uy, uz = g_pos

	c = lambda x: np.cos(x)
	s = lambda x: np.sin(x)

	T01 = np.array([c(TMC1), 0, s(TMC1), 0,
					s(TMC1), 0,-c(TMC1), 0,
					0	   , 1, 	  0, 0,
					0	   , 0,       0, 1]).reshape(4,4)

	T12 = np.array([c(TMC2), -s(TMC2), 0, L1*c(TMC2),
					s(TMC2),  c(TMC2), 0, L1*s(TMC2),
						  0,		1, 0,		   0,
						  0,		0, 0,		   1]).reshape(4,4)

	T23 = np.array([c(MCP), -s(MCP), 0, L2*c(MCP),
					s(MCP),  c(MCP), 0, L2*s(MCP),
						 0,	  	  1, 0,		0,
						 0,  	  0, 0,		1]).reshape(4,4)

	T34 = np.array([c(IP), -s(IP), 0, L3*c(IP),
					s(IP),  c(IP), 0, L3*s(IP),
						0,		1, 0,		 0,
					    0,		0, 0,		 1]).reshape(4,4)

	T00 = np.array([1, 0, 0, ux,
					0, 1, 0, uy,
					0, 0, 1, uz,
					0, 0, 0,  1]).reshape(4,4)

	current_pos = T00
	joint_pos = [g_pos]

	for transform in [T01, T12, T23, T34]:
		current_pos = np.dot(current_pos, transform)
		npos = current_pos[:,-1][:-1]

		joint_pos.append(npos)

	return np.vstack(joint_pos)

def hand_model(hand_params, hand_geometry, ref_pos):
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

	thumb_the = hand_params[:4]
	thumb_geo = hand_geometry[:3]

	index_the = hand_params[4:9]
	index_geo = hand_geometry[3:7]

	middle_the = hand_params[9:14]
	middle_geo = hand_geometry[7:11]

	ring_the = hand_params[14:19]
	ring_geo = hand_geometry[11:15]

	little_the = hand_params[19:]
	little_geo = hand_geometry[15:]

	thumb = thumb_model(thumb_the, ref_pos, thumb_geo)
	index = finger_model(index_the, ref_pos, index_geo)
	middle = finger_model(middle_the, ref_pos, middle_geo)
	ring = finger_model(ring_the, ref_pos, ring_geo)
	litte = finger_model(little_the, ref_pos, little_geo)

	fig = plt.figure()
	ax = fig.add_subplot(111, projection="3d")
	ax.set_xlabel("X")
	ax.set_ylabel("y")
	ax.set_zlabel("z")
	def visualise(joints, ax1):
		X, Y, Z = joints.T
		ax1.plot(X,Y,Z)
	
	for parts in [thumb, index, middle, ring, litte]:
		visualise(parts, ax)

	return 0

thetas = np.deg2rad([5,40,10,10,10])
geometry = np.linspace(5,5,4)
gpos = [3,3,3]

tb_theta = np.deg2rad([45,50,75,75])
tb_geo = np.linspace(3,3,3)
f1_joints = finger_model(thetas, gpos, geometry)
tb_joints = thumb_model(tb_theta, gpos, tb_geo)
# print f1_joints, tb_joints

# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")
# X1, Y1, Z1 = f1_joints.T 
# X2, Y2, Z2 = tb_joints.T
# ax.plot(X1, Y1, Z1)
# ax.plot(X2, Y2, Z2)
# plt.show()

tbT = np.deg2rad([45, 50, 60, 60])
ixT = np.deg2rad([60, 35, 10, 10, 10])
meT = np.deg2rad([90, 10, 12, 13, 15])
rgT = np.deg2rad([120, 20, 80, 70, 60])
leT = np.deg2rad([135, 42, 45, 40, 55])

tbG = np.array([5,3,2])
ixG = np.array([5,3.5,2.5,1.5])
meG = np.array([5,4,3,2])
rgG = np.array([5,3.5,2.5,1.5])
leG = np.array([5,2.5,1.5,1])

the_params = np.hstack((tbT, ixT, meT, rgT, leT))
geo_params = np.hstack((tbG, ixG, meG, rgG, leG))

hand_model(the_params, geo_params, gpos)
plt.show()


