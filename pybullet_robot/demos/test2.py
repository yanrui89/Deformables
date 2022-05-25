import pybullet as p
from time import sleep
import numpy as np
import os
import pybullet_data
import pybullet_utils.bullet_client as bclient

from utils.bullet_manipulator import BulletManipulator
from utils.load_deformables import *
from utils.task_info import DEFORM_INFO, ROBOT_INFO
from utils.args import get_args
from envs.deform_robot_env import DeformRobotEnv


def merge_traj(traj_a, traj_b):
    if traj_a.shape[0] != traj_b.shape[0]:  # padding is required
        n_pad = np.abs(traj_a.shape[0] - traj_b.shape[0])
        zero_pad = np.zeros((n_pad, traj_a.shape[1]))
        if traj_a.shape[0] > traj_b.shape[0]:  # pad ba
            traj_b = np.concatenate([traj_b, zero_pad, ], axis=0)
        else:  # pad a
            traj_a = np.concatenate([traj_a, zero_pad, ], axis=0)
    traj = np.concatenate([traj_a, traj_b, ], axis=-1)
    return traj



################################
# Add robot
##################################
args = get_args()
print(args)
args.env = 'HangGarmentRobot-v1'
robot_env = DeformRobotEnv(args)

'''

###############################################
#physicsClient = p.connect(p.GUI)
sim = bclient.BulletClient(connection_mode=p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.resetSimulation(p.RESET_USE_DEFORMABLE_WORLD)
gravZ=-10
p.setGravity(0, 0, gravZ)
p.setTimeStep(1/500)

planeOrn = [0,0,0,1]#p.getQuaternionFromEuler([0.3,0,0])
planeId = p.loadURDF("plane.urdf", [0,0,0],planeOrn)

'''

################################################
# ADD Deformables
'''
BAGS_TO_FILES = {
    1: 'assets/bags/bl_sphere_bag_rad_1.0_zthresh_0.1_numV_257.obj',
    2: 'assets/bags/bl_sphere_bag_rad_1.0_zthresh_0.3_numV_289.obj',
    3: 'assets/bags/bl_sphere_bag_rad_1.0_zthresh_0.4_numV_321.obj',
    4: 'assets/bags/bl_sphere_bag_rad_1.0_zthresh_0.6_numV_353.obj',
    5: 'assets/bags/bl_sphere_bag_rad_1.0_zthresh_0.8_numV_385.obj',
}
'''

bag_class = BagEnv()
bagOrn = p.getQuaternionFromEuler([1.57,0,0])
bag_id = bag_class.add_bag([10.3, 2.3, 5.9], bagOrn)
p.stepSimulation()
bag_class.add_zip_base(bag_id)
#bag_class.add_cylinder_base(bag_id = bag_id)


#tote_class = ToteBag()
#tote_id = tote_class.add_bag([1.0, 1.0, 0.0], [0, 0, -0.707, 0.707])
#tote_class.change_texture(tote_id)
#bag_class.understand_bag_top_ring([1.5, 0.3, 0.0])
#bag_class.add_cable_ring(bag_id = bag_id)

#boxId = p.loadURDF("cube.urdf", [0,1,2],useMaximalCoordinates = True)

#clothId = p.loadSoftBody("cloth_z_up.obj", basePosition = [0,0,1], scale = 0.5, mass = 1., useNeoHookean = 0, useBendingSprings=1,useMassSpring=1, springElasticStiffness=40, springDampingStiffness=.1, springDampingAllDirections = 1, useSelfCollision = 0, frictionCoeff = .5, useFaceContact=1)


'''
p.changeVisualShape(clothId, -1, flags=p.VISUAL_SHAPE_DOUBLE_SIDED)

p.createSoftBodyAnchor(clothId  ,24,-1,-1)
p.createSoftBodyAnchor(clothId ,20,-1,-1)
#p.createSoftBodyAnchor(clothId ,15,boxId,-1, [0.5,-0.5,0])
#p.createSoftBodyAnchor(clothId ,14,boxId,-1, [0.5,-0.5,0])
p.createSoftBodyAnchor(clothId ,10,boxId,-1, [0.5,-0.5,0])
#p.createSoftBodyAnchor(clothId ,19,boxId,-1, [-0.5,0.5,0])
p.setPhysicsEngineParameter(sparseSdfVoxelSize=0.25)

debug = True
if debug:
  data = p.getMeshData(clothId, -1, flags=p.MESH_DATA_SIMULATION_MESH)
  print("--------------")
  print("data=",data)
  print(data[0])
  print(data[1])
  text_uid = []
  for i in range(data[0]):
      pos = data[1][i]
      uid = p.addUserDebugText(str(i), pos, textColorRGB=[1,1,1])
      text_uid.append(uid)
'''
###############################
# Manipulate 2 robots
##############################

pos_r= robot_env.robot.get_ee_pos(left=False)
pos_l= robot_env.robot.get_ee_pos(left=True)

# build trajectories to manipulate bag
mid_point = (pos_r[0] + pos_l[0]) / 2
end_pos_r = np.arange(pos_r[0], mid_point + 1, -0.1)
end_pos_l = np.arange(pos_l[0], mid_point - 1, 0.1)


pos_r = np.tile(pos_r[1:], (np.shape(end_pos_r)[0],1))
pos_l = np.tile(pos_l[1:], (np.shape(end_pos_l)[0],1))


traj_r = np.column_stack((end_pos_r, pos_r))
traj_l = np.column_stack((end_pos_l, pos_l))

traj = merge_traj(traj_r, traj_l)
size_traj = traj.shape[0]
#print(traj.shape)
#print(size_traj)
# Move the robot


############################
#Loop
#############################
stp = 0
while p.isConnected():
  #p.getCameraImage(320,200)
  '''
  if debug:
    data = p.getMeshData(clothId, -1, flags=p.MESH_DATA_SIMULATION_MESH)
    for i in range(data[0]):
      pos = data[1][i]
      uid = p.addUserDebugText(str(i), pos, textColorRGB=[1,1,1], replaceItemUniqueId=text_uid[i])
  '''
  #print(pos_r)
  #print(pos_l)
  #p.setGravity(0,0,gravZ)
  if stp < size_traj:
    print('I am in here leh[hhhhhhhhhhhhhhhhhhhh')
    action = traj[stp]
    robot_env.step(action, fing_dist = -1, unscaled=True)
    stp += 1
  else:
    robot_env.step(action, fing_dist = 1, unscaled=True)
  #robot_env.robot.move_fing(2)
  
  #p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING,1)
  #sleep(1./240.)