import os

import numpy as np
import pybullet as p

from tqdm import tqdm
from env import ClutteredPushGrasp
from robot import Panda, UR5Robotiq85, UR5Robotiq140, PandaRobotiq140
from utilities import YCBModels, Camera
import time
import math
from read_write_helper import read_fep_dataset


def user_control_demo():
    ycb_models = YCBModels(
        os.path.join('./data/ycb', '**', 'textured-decmp.obj'),
    )
    camera = Camera((1, 1, 1),
                    (0, 0, 0),
                    (0, 0, 1),
                    0.1, 5, (320, 320), 40)
    camera = None
    # robot = Panda((0, 0.5, 0), (0, 0, math.pi))
    robot = PandaRobotiq140((0, 0.5, 0), (0, 0, math.pi))
    env = ClutteredPushGrasp(robot, ycb_models, camera, vis=True)


    env.reset()
    # env.SIMULATION_STEP_DELAY = 0

    DTYPE = 'float64'
    FEP_MOVEMENT_DATASET_PATH = "/home/clearlab/franka_emika_panda_pybullet/movement_datasets/fep_state_to_pid-corrected-torque_55s_dataset.csv"
    pos, _, _, _ = read_fep_dataset(FEP_MOVEMENT_DATASET_PATH, DTYPE)
    dataset_length = pos.shape[0]
    #move to home position
    print(pos[10])
    count = 0
    

    ############## Warm-up
    while True:
        #home_pos = [4.71, -1.07, 0, -2.30, 0, 1.57, 3.14]
        home_pos = [1.57079632679, -1.0471975512, 0, -2.35619449019, 0, 1.57079632679, 3.14159265359]
        
        obs, reward, done, info = env.step_auto(home_pos, env.read_debug_parameter())
        count += 1
        robot_id = robot.return_robot_id()
        block_id = env.return_block_id()
        table_id = env.return_table_id()
        trial_id = env.return_trial_id()
        #print(robot_id)
        #a = p.getLinkState(robot_id, 9)
        #obj_euler = p.getEulerFromQuaternion(a[1])
        #print(a)
        #print(obj_euler)
        #change block coefficient of friction
        p.changeDynamics(block_id, -1, mass=2, frictionAnchor = 1)
        #p.changeDynamics(robot_id, 13, lateralFriction=5)
        #p.changeDynamics(robot_id, 18, lateralFriction=5)
        a = p.getDynamicsInfo(block_id, -1)
        print(a)
        if count == 100:
            print('yes')
            break


    ############### Actual run
    #############
    # robot.move_ee([-4.0368805459527245e-05, 0.26201263031535293, 0.6305756808979718,-3.094850582043303, 0.25773188803907965, 1.7513099347692396], 'end')
    # Cartesian coordinates of the end effector
    
    robot.move_ee([0, 0, 0.5,-3.094850582043303, 0, 1.7513099347692396], 'end')
    robot.move_gripper(0.085)
    p.stepSimulation()
    
    time.sleep(1 / 240)

    for i in range(100):

        robot.move_ee([0, 0, 0.4,-3.094850582043303, 0, 1.7513099347692396], 'end')
        robot.move_gripper(0.085)
        p.stepSimulation()

        time.sleep(1 / 240)
        print('yes please')

    for i in range(100):
        
        robot.move_ee([0, 0, 0.4,-3.094850582043303, 0, 1.7513099347692396], 'end')
        robot.move_gripper(0.0)
        p.stepSimulation()

        time.sleep(1 / 240)
        print('no please')


    _link_name_to_index = {p.getBodyInfo(trial_id)[0].decode('UTF-8'):-1,}
    for _id in range(p.getNumJoints(trial_id)):
        _name = p.getJointInfo(trial_id, _id)[12].decode('UTF-8')
        _link_name_to_index[_name] = _id
    
    print(_link_name_to_index)
    print(p.getNumJoints(trial_id))
    ab = 1
    while True:
        #robot.move_ee([0, 0, 0.4,-3.094850582043303, 0, 1.7513099347692396], 'end')
        #robot.move_gripper(0.055)
        #p.applyExternalForce(block_id,-1,[0,0,10],[-0.4,0.4,0],p.WORLD_FRAME)
        #p.applyExternalForce(block_id,-1,[0,0,10],[-0.4,-0.4,0],p.WORLD_FRAME)
        #p.applyExternalForce(block_id,-1,[0,0,10],[0.4,-0.4,0],p.WORLD_FRAME)
        p.applyExternalForce(trial_id,0,[10000,1000,1000],[0,0, 0],p.WORLD_FRAME)
        #p.setJointMotorControl2(trial_id, 0, p.TORQUE_CONTROL, force= ab)
        p.stepSimulation()
        
        #print(ab)

        time.sleep(1 / 240)
        #print('no please')
        a = p.getContactPoints(table_id, block_id)
        #print(a)

    '''
    robot.move_ee([0, 0, 0.2,-3.094850582043303, 0, 1.7513099347692396], 'end')
    robot.move_gripper(0)
    p.stepSimulation()
    
    time.sleep(1 / 240)
    '''   

    
    
    
    '''
    while True:
        obs, reward, done, info = env.step(env.read_debug_parameter(), 'end')
        
        for i in range(dataset_length):
            current_pos = pos[i]
            obs, reward, done, info = env.step_auto(current_pos, env.read_debug_parameter())
        
        # print(obs, reward, done, info)
        #print('yey')

    '''
if __name__ == '__main__':
    user_control_demo()
