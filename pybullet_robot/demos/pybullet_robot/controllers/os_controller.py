import threading
import numpy as np
import time
import pybullet as pb
import pandas as pd


class OSControllerBase(object):

    def __init__(self, robot, config, break_condition=None):

        self._robot = robot # robot sim should not be in real-time. Step simulation will be called by controller.
        
        self._P_pos = np.diag(config['P_pos'])
        self._D_pos = np.diag(config['D_pos'])

        self._P_ori = np.diag(config['P_ori'])
        self._D_ori = np.diag(config['D_ori'])


        self._error_thresh = config['error_thresh']
        self._start_err = config['start_err']

        self._robot.set_ctrl_mode('tor')

        self._run_ctrl = False

        if break_condition is not None and callable(break_condition):
            self._break_condition = break_condition
        else:
            self._break_condition = lambda: False

        self._ctrl_thread = threading.Thread(target=self._control_thread)
        self._mutex = threading.Lock()

        self._sim_timestep = pb.getPhysicsEngineParameters()['fixedTimeStep']
        self._sim_time = 0.0

        if 'rate' not in config:
            self._ctrl_rate = 1./self._sim_timestep
        else:
            self._ctrl_rate = float(config['rate'])

        self.count = 0
        self.torque_list1 = []
        self.torque_list2 = []
        self.torque_list3 = []
        self.torque_list4 = []
        self.torque_list5 = []
        self.torque_list6 = []
        self.torque_list7 = []
        self.error_list1 = []
        self.error_list2 = []
        self.error_list3 = []
        self.error_list4 = []
        self.count_list = []
        self.x_position_list = []
        self.y_position_list = []

        self.moving_tau = np.zeros(7)

    def update_goal(self):
        """
        Has to be implemented in the inherited class.
        Should update the values for self._goal_pos and self._goal_ori at least.
        """
        raise NotImplementedError("Not implemented")

    def update_goal(self):
        """
        Has to be implemented in the inherited class.
        Should update the values for self._goal_pos and self._goal_ori at least.
        """
        raise NotImplementedError("Not implemented")

    def _compute_cmd(self):
        """
        Should be implemented in inherited class. Should compute the joint torques
        that are to be applied at every sim step.
        """
        raise NotImplementedError("Not implemented")

    def _control_thread(self):
        """
            Apply the torque command computed in _compute_cmd until any of the 
            break conditions are met.
        """
        while self._run_ctrl and not self._break_condition():
            error = self._start_err

            
            while np.any(error > self._error_thresh):
                #print(self.count)
                now = time.time()
                
                self._mutex.acquire()
                tau, error = self._compute_cmd()
                #moving average of tau
                if np.all(self.moving_tau) == 0:
                    self.moving_tau = tau

                #self.moving_tau = 0.6 * self.moving_tau + 0.4 * tau
                a,_,_,_ = self._robot.get_joint_state()
                #print(a)
                print(self.moving_tau)
                # command robot using the computed joint torques
                self._robot.exec_torque_cmd(tau)
                #print(tau.shape)
                self.move_table(self._table_vel)
                
                self.torque_list1.append(np.array(tau[0]))
                self.torque_list2.append(np.array(tau[1]))
                self.torque_list3.append(np.array(tau[2]))
                self.torque_list4.append(np.array(tau[3]))
                self.torque_list5.append(np.array(tau[4]))
                self.torque_list6.append(np.array(tau[5]))
                self.torque_list7.append(np.array(tau[6]))

                self.error_list1.append(np.array(error[0]))
                self.error_list2.append(np.array(error[1]))
                self.count_list.append(self.count)
                position_x = self._robot.ee_pose()[0]
                self.x_position_list.append(np.array(position_x[0]))
                self.y_position_list.append(np.array(position_x[1]))
                self.count += 1

                self._robot.step_if_not_rtsim()
                self._sim_time += self._sim_timestep
                self._mutex.release()

                # self._rate.sleep()
                elapsed = time.time() - now
                sleep_time = (1./self._ctrl_rate) - elapsed
                if sleep_time > 0.0:
                    time.sleep(sleep_time)

                if self.count == 9000:
                    data = {'count': self.count_list, 'torque1': self.torque_list1, 'torque2': self.torque_list2, 'torque3': self.torque_list3, 'torque4': self.torque_list4, 'torque5': self.torque_list5, 'torque6': self.torque_list6 , 'torque7': self.torque_list7, 'error1':self.error_list1, 'error2':self.error_list2, 'positionx': self.x_position_list, 'positiony': self.y_position_list}
                    df = pd.DataFrame(data)
                    df.to_csv('/home/clearlab/test.csv')

    def _initialise_goal(self):
        """
        Should initialise _goal_pos, _goal_ori, etc. for controller to start the loop.
        Ideally these should be the current value of the robot's end-effector.
        """
        raise NotImplementedError("Not implemented")

    def start_controller_thread(self):
        self._initialise_goal()
        self._run_ctrl = True
        self._ctrl_thread.start()

    def stop_controller_thread(self):
        self._run_ctrl = False
        if self._ctrl_thread.is_alive():
            self._ctrl_thread.join()

    def __del__(self):
        self.stop_controller_thread()

    
    def move_table(self):
        """
        Should be implemented in inherited class. Should compute the joint torques
        that are to be applied at every sim step.
        """
        raise NotImplementedError("Not implemented")
