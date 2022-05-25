from tkinter import E
import numpy as np


a = np.load('/home/clearlab/pybullet_robot/demos/test22.npy')

verts_array = np.array(a)
verts_x = verts_array[:,0]
verts_y = verts_array[:,1]
verts_z = verts_array[:,2]

min_idx_x = np.min(verts_x)
min_idx_y = np.min(verts_y)
min_idx_z = np.min(verts_z)

max_idx_x = np.max(verts_x)
max_idx_y = np.max(verts_y)
max_idx_z = np.max(verts_z)

print(min_idx_z)
print(max_idx_z)

c = np.abs(verts_y - 0.35) < 0.02
d = np.abs(verts_z - 0.85) < 0.02

e = np.logical_and(c, d)
#print(verts_z - 0.84)
b = np.where(e)
print(b)