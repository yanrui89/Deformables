import numpy as np
import pybullet as pb
from pybullet_robot.controllers.utils import quatdiff_in_euler
from pybullet_robot.controllers.os_controller import OSControllerBase
from pybullet_robot.controllers.ctrl_config import OSHybConfig


class OSHybridController(OSControllerBase):

    def __init__(self, robot, objects, config=OSHybConfig, **kwargs):

        OSControllerBase.__init__(self, robot=robot, config=config, **kwargs)

        self._P_ft = np.diag(np.append(config['P_f'],config['P_tor']))
        self._I_ft = np.diag(np.append(config['I_f'],config['I_tor']))

        self._null_Kp = np.diag(config['null_stiffness'])

        self._windup_guard = np.asarray(config['windup_guard']).reshape([6,1])

        self.change_ft_directions(np.asarray(config['ft_directions'], int))
        self._objects = objects

    def change_ft_directions(self, dims):
        self._mutex.acquire()
        self._ft_dir = np.diag(dims)
        self._pos_dir = np.diag([1, 1, 1, 1, 1, 1]) ^ self._ft_dir
        self._I_term = np.zeros([6, 1])
        self._mutex.release()

    def update_goal(self, goal_pos, goal_ori, goal_force = np.zeros(3), goal_torque = np.zeros(3), table_vel=np.zeros(1)):
        self._mutex.acquire()
        self._goal_pos = np.asarray(goal_pos).reshape([3, 1])
        self._goal_ori = np.asarray(goal_ori)
        self._goal_ft = -np.append(np.asarray(goal_force), np.asarray(goal_torque)).reshape([6, 1])
        self._table_vel = np.asarray(table_vel)
        self._mutex.release()

    def _compute_cmd(self):
        """
        Actual control loop. Uses goal pose from the feedback thread
        and current robot states from the subscribed messages to compute
        task-space force, and then the corresponding joint torques.
        """
        ## MOTION CONTROL
        curr_pos, curr_ori = self._robot.ee_pose()   #curr_pos is vector of 3. curr ori is quaternion

        #print(curr_pos)
        delta_pos = self._goal_pos - curr_pos.reshape([3, 1])
        delta_ori = quatdiff_in_euler(
            curr_ori, self._goal_ori).reshape([3, 1])   #delta ori is difference in angle in terms of euler angles

        curr_vel, curr_omg = self._robot.ee_velocity()  #Find the eef link velocity


        # Desired task-space motion control PD law
        F_motion = self._pos_dir.dot(np.vstack([self._P_pos.dot(delta_pos), self._P_ori.dot(delta_ori)]) - \
                                     np.vstack([self._D_pos.dot(curr_vel.reshape([3, 1])),
                                                self._D_ori.dot(curr_omg.reshape([3, 1]))]))
        ## FORCE CONTROL
        last_time = self._last_time if self._last_time is not None else self._sim_time  #se;f.sim time is the time since simulation
        current_time = self._sim_time
        delta_time = max(0.,current_time - last_time)

        curr_ft = self._robot.get_ee_wrench(local=False).reshape([6, 1])   #current force torque of a eef


        delta_ft = self._ft_dir.dot(self._goal_ft - curr_ft)   #There will only be 1 value in the z component of the vector
        self._I_term += delta_ft * delta_time   #seems like force times delta time. vector of 6 x 1

        # print np.diag(self._pos_dir), np.diag(self._ft_dir)
        self._I_term[self._I_term+self._windup_guard < 0.] = -self._windup_guard[self._I_term+self._windup_guard < 0.]
        self._I_term[self._I_term-self._windup_guard > 0.] = self._windup_guard[self._I_term-self._windup_guard > 0.]
        # after this part self.Iterm doesnt seem to change. still 6 x 1 with only entry in the z axis
        # Desired task-space force control PI law
        
        F_force = self._P_ft.dot(delta_ft) + self._I_ft.dot(self._I_term) + self._goal_ft
        #print(F_force)
        
        F = F_motion - F_force # force control is subtracted because the computation is for the counter force

        

        error = np.asarray([(np.linalg.norm(self._pos_dir[:3, :3].dot(delta_pos))), np.linalg.norm(self._pos_dir[3:, 3:].dot(delta_ori)),
                        np.linalg.norm(delta_ft[3:]), np.linalg.norm(delta_ft[3:])])

        

        J = self._robot.jacobian()

        self._last_time = current_time

        cmd = np.dot(J.T, F)   #Convert to joint space
        #print(self._null_Kp)
        null_space_filter = self._null_Kp.dot(
            np.eye(7) - J.T.dot(np.linalg.pinv(J.T, rcond=1e-3)))

        #print(null_space_filter.dot((self._robot._tuck-self._robot.angles()).reshape([7,1])))

        cmd += null_space_filter.dot((self._robot._tuck-self._robot.angles()).reshape([7,1]))
        # print null_space_filter.dot(
            # (self._robot._tuck-self._robot.angles()).reshape([7, 1]))
        # joint torques to be commanded
        return cmd, error

    def _initialise_goal(self):
        self._last_time = None
        self._I_term = np.zeros([6,1])
        self.update_goal(self._robot.ee_pose()[0], self._robot.ee_pose()[1], np.zeros(3), np.zeros(3))

    def move_table(self, table_vel):
        trial_id = self._objects['table'] 

        pb.setJointMotorControl2(trial_id, 0, pb.VELOCITY_CONTROL, targetVelocity= table_vel)
        
