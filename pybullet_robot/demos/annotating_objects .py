##  Helper script for annotating objects in blender.
## This file should be separated into four separate scripts inside blender.


## Script one: Loading_file
##

import bpy
import os
bpy.ops.object.mode_set(mode='OBJECT')
for key, scene_obj in dict(bpy.data.objects).items():
    print(scene_obj)
    scene_obj.select_set(True)
    bpy.context.view_layer.objects.active = scene_obj
    bpy.ops.object.delete()

datapath = '/home/clearlab/pybullet_robot/demos/assets/bags'
obj_name = 'cloth/tshirt_0.obj'
filepath = os.path.join(datapath, obj_name)
bpy.ops.import_scene.obj(filepath=filepath)
print(">>>>>>",dict(bpy.data.objects))
# edit mode in the first object
objs = dict(bpy.data.objects)
obj = list(objs.values())[0]
print( '>>>>>',obj)
obj.select_set(True)
bpy.context.view_layer.objects.active = obj
bpy.ops.object.mode_set(mode='EDIT')

## Script two: Loading_file
##

# Note: make sure to check "preserve vertex order" when exporting to obj.
# For OS X: run bender from command line, no other way to see console outputs.
import sys
import os
import subprocess
python_bin = os.path.join(sys.prefix, 'bin', 'python3.9')
#subprocess.call([python_bin, '-m', 'ensurepip'])
#subprocess.call([python_bin, '-m', 'pip', 'install', '--upgrade', 'pip'])
#subprocess.call([python_bin, '-m', 'pip', 'install', 'scipy'])

import bpy
import bmesh
import numpy as np
from scipy import spatial

v_ids = []
v_pts = []
obj=bpy.context.object
if obj.mode == 'EDIT':
    bm=bmesh.from_edit_mesh(obj.data)
    for v in bm.verts:
        if v.select:
            v_ids.append(v.index)
            v_pts.append(v.co)
    hull = spatial.ConvexHull(v_pts)
    hull_vertices = {}
    for s in hull.simplices:
        for idx in s:
            hull_vertices[v_ids[idx]] = ''
    hull_vertices = sorted(hull_vertices.keys())
    print('selected', len(v_ids), '/', len(bm.verts))
    print(len(hull_vertices), 'hull vertices:', hull_vertices)
    max_smpls = 100
    if len(hull_vertices) > max_smpls:
        hull_smpl = np.sort(np.random.choice(
            hull_vertices, size= max_smpls, replace=False))
        print('subsampled ', max_smpls , ':', repr(hull_smpl))
else:
    print('Object is not in edit mode.')


## Script three: Exporting file
##

import bpy
import os
datapath = '/home/pyshi/code/dedo/dedo/data/'
obj = 'cloth/tshirt_0.obj'
filepath = os.path.join(datapath, obj)
bpy.ops.export_scene.obj(filepath=filepath)
