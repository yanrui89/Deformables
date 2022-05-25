from collections import deque
import numpy as np
import pybullet as pb
from pybullet_robot.worlds import SimpleWorld, add_PyB_models_to_path
from pybullet_robot.robots import PandaArm
from pybullet_robot.controllers import OSHybridController
import time
import matplotlib.pyplot as plt
import threading
#from utils import COLORS as U
import os


def plot_thread():

    plt.ion()
    while True:
        plt.clf()
        plt.plot(fx_deque, 'r', label='x')
        plt.plot(fy_deque, 'g', label='y')
        plt.plot(fz_deque, 'b', label='z')
        plt.legend()
        plt.draw()
        plt.pause(0.000001)
        if done:
            break

BAGS_TO_FILES = {
    1: 'assets/bags/bl_sphere_bag_rad_1.0_zthresh_0.1_numV_257.obj',
    2: 'assets/bags/bl_sphere_bag_rad_1.0_zthresh_0.3_numV_289.obj',
    3: 'assets/bags/bl_sphere_bag_rad_1.0_zthresh_0.4_numV_321.obj',
    4: 'assets/bags/bl_sphere_bag_rad_1.0_zthresh_0.6_numV_353.obj',
    5: 'assets/bags/bl_sphere_bag_rad_1.0_zthresh_0.8_numV_385.obj',
}

class BagEnv():
    """Superclass to reduce code duplication.

    Gripping parameter: the threshold should probably be lower compared to
    the cloth tasks, since for our bags, we generally want to grip the beads,
    instead of the bag vertices.
    """

    def __init__(self):
        super().__init__()
        self.ee = 'suction'
        self.primitive = 'pick_place'
        self.max_steps = 11
        self._IDs = {}
        self._debug = True
        self._settle_secs = 1

        # Gripping parameters. See docs above.
        self._def_threshold = 0.020
        self._def_nb_anchors = 1

        # Scale the bag / zone. The zone.obj ranges from (-20,20).
        self._zone_scale = 0.0130
        self._bag_scale = 0.10
        self._zone_length = (20. * self._zone_scale)
        self.zone_size = (20. * self._zone_scale, 20. * self._zone_scale,  0.0)
        self._bag_size = (  1. * self._bag_scale,   1. * self._bag_scale, 0.01)

        # Bag type (or resolution?) and parameters.
        self._bag = 4
        self._mass = 1.0
        self._scale = 0.25
        self._collisionMargin = 0.003
        self._base_orn = [np.pi / 2.0, 0.0, 0.0]
        self._f_bag = BAGS_TO_FILES[self._bag]
        self._drop_height = 5000.50



    def add_bag(self, base_pos, base_orn, bag_color='yellow'):
        """Adding a bag from an .obj file."""
        bag_id = pb.loadSoftBody(
                fileName=self._f_bag,
                basePosition=base_pos,
                baseOrientation=base_orn,
                collisionMargin=self._collisionMargin,
                scale=self._bag_scale,
                mass=self._mass,
                useNeoHookean=0,
                useBendingSprings=1,
                useMassSpring=1,
                springElasticStiffness=40,
                springDampingStiffness=0.1,
                springDampingAllDirections=1,
                useSelfCollision=0,
                frictionCoeff=0.5,
                useFaceContact=1)

        # Only if using more recent PyBullet versions.
        '''
        p_version = pkg_resources.get_distribution('pybullet').version
        '''
        #if p_version == '3.0.4':
        color = [237.0 / 255.0, 201.0 / 255.0, 072.0 / 255.0] + [1]
        pb.changeVisualShape(bag_id, -1, flags=pb.VISUAL_SHAPE_DOUBLE_SIDED,
                            rgbaColor=color)
        #else:
        #    raise ValueError(p_version)
        

        # For tracking IDs and consistency with existing ravens code.
        self._IDs[bag_id] = 'bag'
        #self.object_points[bag_id] = np.float32((0, 0, 0)).reshape(3, 1)
        #env.objects.append(bag_id)

        # To help environment pick-place method track all deformables.
        #self.def_IDs.append(bag_id)

        return bag_id

    def add_cable_ring(self, bag_id=None):
        """Make the cable beads coincide with the vertices of the top ring.

        This should lead to better physics and will make it easy for an
        algorithm to see the bag's top ring. Please see the cable-ring env
        for details, or `scratch/cable_ring_MWE.py`. Notable differences
        (or similarities) between this and `dan_cables.py`:

        (1) We don't need to discretize rotations and manually compute bead
        positions, because the previously created bag 'creates' it for us.

        (2) Beads have anchors with vertices, in addition to constraints with
        adjacent beads.

        (3) Still use `self.cable_bead_IDs` as we want that for the reward.
        """
        num_parts = len(self._top_ring_idxs)
        radius = 0.005
        color = [078.0 / 255.0, 121.0 / 255.0, 167.0 / 255.0] + [1]
        beads = []
        bead_positions_l = []
        part_shape = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=[radius]*3)
        part_visual = pb.createVisualShape(pb.GEOM_SPHERE, radius=radius*1.5)

        # All tasks OTHER than bag-color-goal use self.bag_id. So if we are not
        # passing it in as an argument, we better have defined self.bag_id to use.
        if bag_id is None:
            bag_id = self.bag_id

        # Fortunately `verts_l` coincides with `self._top_ring_idxs`.
        _, verts_l = pb.getMeshData(bag_id, -1, flags=pb.MESH_DATA_SIMULATION_MESH)

        # Iterate through parts and create constraints as needed.
        for i in range(num_parts):
            bag_vidx = self._top_ring_idxs[i]
            bead_position = np.float32(verts_l[bag_vidx])
            part_id = pb.createMultiBody(0.01, part_shape, part_visual,
                    basePosition=bead_position)
            pb.changeVisualShape(part_id, -1, rgbaColor=color)

            if i > 0:
                parent_frame = bead_position - bead_positions_l[-1]
                constraint_id = pb.createConstraint(
                        parentBodyUniqueId=beads[-1],
                        parentLinkIndex=-1,
                        childBodyUniqueId=part_id,
                        childLinkIndex=-1,
                        jointType=pb.JOINT_POINT2POINT,
                        jointAxis=(0, 0, 0),
                        parentFramePosition=parent_frame,
                        childFramePosition=(0, 0, 0))
                pb.changeConstraint(constraint_id, maxForce=100)

            # Make a constraint with i=0. Careful with `parent_frame`!
            if i == num_parts - 1:
                parent_frame = bead_positions_l[0] - bead_position
                constraint_id = pb.createConstraint(
                        parentBodyUniqueId=part_id,
                        parentLinkIndex=-1,
                        childBodyUniqueId=beads[0],
                        childLinkIndex=-1,
                        jointType=pb.JOINT_POINT2POINT,
                        jointAxis=(0, 0, 0),
                        parentFramePosition=parent_frame,
                        childFramePosition=(0, 0, 0))
                pb.changeConstraint(constraint_id, maxForce=100)

            # Create constraint between a bead and certain bag vertices.
            _ = pb.createSoftBodyAnchor(
                    softBodyBodyUniqueId=bag_id,
                    nodeIndex=bag_vidx, 
                    bodyUniqueId=part_id,
                    linkIndex=-1,)

            # Track beads.
            beads.append(part_id)
            bead_positions_l.append(bead_position)

            # The usual for tracking IDs. Four things to add.
            #self.cable_bead_IDs.append(part_id)
            #self._IDs[part_id] = f'cable_part_{str(part_id).zfill(2)}'
            #env.objects.append(part_id)
            #self.object_points[part_id] = np.float32((0, 0, 0)).reshape(3, 1)

    def fit_circle(self,points_l, scale, debug=False):
        """Get information about a circle from a list of points `points_l`.

        This may involve fitting a circle or ellipse to a set of points?

        pip install circle-fit
        https://github.com/AlliedToasters/circle-fit

        Assuing for now that points_l contains a list of (x,y,z) points, so we
        take only (x,y) and scale according to `scale`. Both methods return a
        tuple of four values:

        xc: x-coordinate of solution center (float)
        yc: y-coordinate of solution center (float)
        R: Radius of solution (float)
        variance or residual (float)

        These methods should be identical if we're querying this with actual
        circles. Returning the second one for now.
        """
        from circle_fit import hyper_fit, least_squares_circle
        data = [ (item[0]*scale, item[1]*scale) for item in points_l ]
        data = np.array(data)
        circle_1 = hyper_fit(data)
        circle_2 = least_squares_circle(data)
        xc_1, yc_1, r_1, _ = circle_1
        xc_2, yc_2, r_2, _ = circle_2
        if debug:
            print(f'(hyperfit) rad {r_1:0.4f}, center ({xc_1:0.4f},{yc_1:0.4f})')
            print(f'(least-sq) rad {r_2:0.4f}, center ({xc_2:0.4f},{yc_2:0.4f})')
        return circle_2

    def understand_bag_top_ring(self, base_pos):
        """By our circular bag design, there exists a top ring file.

        Reading it gives us several important pieces of information. We assign to:

            _top_ring_idxs: indices of the vertices (out of entire bag).
            _top_ring_posi: their starting xyz positions (BEFORE simulation
                or applying pose transformations). This way we can get the
                area of the circle. We can't take the rotated bag and map
                vertices to the xy plane, because any rotation will make the
                area artificially smaller.

        The .txt file saves in (x,y,z) order but the .obj files put z second.
        Make sure vertex indices are MONOTONICALLY INCREASING since I use
        that assumption to 'assign' vertex indices in order to targets.

        Input: base_pos, the center of the bag's sphere.
        """
        self._top_ring_f = (self._f_bag).replace('.obj', '_top_ring.txt')
        self._top_ring_f = os.path.join(self._top_ring_f)
        self._top_ring_idxs = [] # is this the same as p.getMeshData?
        self._top_ring_posi = [] # for raw, non-scaled bag
        with open(self._top_ring_f, 'r') as fh:
            for line in fh:
                ls = (line.rstrip()).split()
                vidx = int(ls[0])
                vx, vy, vz = float(ls[1]), float(ls[2]), float(ls[3])
                if len(self._top_ring_idxs) >= 1:
                    assert vidx > self._top_ring_idxs[-1], \
                            f'Wrong: {vidx} vs {self._top_ring_idxs}'
                self._top_ring_idxs.append(vidx)
                self._top_ring_posi.append((vx,vy,vz))

        # Next, define a target zone. This makes a bunch of plus signs in a
        # circular fashion from the xy projection of the ring.
        self._target_positions = []
        for item in self._top_ring_posi:
            sx, sy, _ = item
            sx = sx * self._bag_scale + base_pos[0]
            sy = sy * self._bag_scale + base_pos[1]
            self._target_positions.append( (sx,sy,0) )
            self._targets_visible = False
            if self._targets_visible:
                square_pose = ((sx,sy,0.001), (0,0,0,1))
                square_template = 'assets/square/square-template-allsides-green.urdf'
                replace = {'DIM': (0.004,), 'HALF': (0.004 / 2,)}
                urdf = self.fill_template(square_template, replace)
                #env.add_object(urdf, square_pose, fixed=True)
                os.remove(urdf)


        # Fit a circle and print some statistics, can be used by demonstrator.
        # We should be careful to consider nonplanar cases, etc.
        xc, yc, rad, _ = self.fit_circle(self._top_ring_posi, self._bag_scale, debug=False)
        self._circle_area = np.pi * (rad ** 2)
        self._circle_center = (xc * self._bag_scale + base_pos[0],
                               yc * self._bag_scale + base_pos[1])


class ClothEnv():
    """Superclass for cloth environments.

    Reward: cloth-flat-easy and cloth-pick-place use coverage. See comments
    in task.py, it's slightly roundabout. Demonstrator: corner-pulling
    demonstrator who uses access to underlying state information.

    In reset(), use the `replace` dict to adjust size of zone, and `scale`
    for the size of cloth. Note: the zone.obj ranges from (-10,10) whereas my
    cloth files from Blender scale from (-1,1), hence the cloth scale needs
    to be 10x *larger* than the zone scale. If this convention changes with
    any of the files, adjust accordingly. That will change how the zone looks
    in the images, and its corner indices.

    We also use `zone_size` to determine the object size for sampling. (And,
    in other environments where we must push items to a target, `zone_size`
    determines boundaries.)

    If I want at most N actions per episode, set max_steps=N+1 because the
    first has no primitive for whatever reason.
    """

    def __init__(self):
        super().__init__()
        self.ee = 'suction'
        self.primitive = 'pick_place'
        self.max_steps = 11
        self._IDs = {}
        self._debug = True
        self._settle_secs = 0

        # Gripping parameters. I think _def_threshold = 0.020 is too low.
        self._def_threshold = 0.025
        self._def_nb_anchors = 1

        # See scaling comments above.
        self._zone_scale = 0.01
        self._cloth_scale = 0.10
        self._cloth_length = (2.0 * self._cloth_scale)
        self._zone_length = (20.0 * self._zone_scale)
        self.zone_size = (20.0 * self._zone_scale, 20.0 * self._zone_scale, 0)
        self._cloth_size = (self._cloth_length, self._cloth_length, 0.01)
        assert self._cloth_scale == self._zone_scale * 10, self._cloth_scale

        # Cloth resolution and corners (should be clockwise).
        self.n_cuts = 10
        if self.n_cuts == 5:
            self.corner_indices = [0, 20, 24, 4]  # actual corners
        elif self.n_cuts == 10:
            self.corner_indices = [11, 81, 88, 18]  # one corner inwards
        else:
            raise NotImplementedError(self.n_cuts)
        self._f_cloth = 'assets/cloth/bl_cloth_{}_cuts.obj'.format(
                str(self.n_cuts).zfill(2))

        # Other cloth parameters.
        self._mass = 0.5
        self._edge_length = (2.0 * self._cloth_scale) / (self.n_cuts - 1)
        self._collisionMargin = self._edge_length / 5.0

        # IoU/coverage rewards (both w/zone or goal images). Pixels w/255 are targets.
        self.target_hull_bool = None
        self.zone_ID = -1

    def add_cloth(self, base_pos, base_orn):
        """Adding a cloth from an .obj file."""
        cloth_id = pb.loadSoftBody(
                fileName=self._f_cloth,
                basePosition=base_pos,
                baseOrientation=base_orn,
                collisionMargin=self._collisionMargin,
                scale=self._cloth_scale,
                mass=self._mass,
                useNeoHookean=0,
                useBendingSprings=1,
                useMassSpring=1,
                springElasticStiffness=40,
                springDampingStiffness=0.1,
                springDampingAllDirections=0,
                useSelfCollision=1,
                frictionCoeff=1.0,
                useFaceContact=1,)

        # Only if using more recent PyBullet versions.
    

        color = [237.0 / 255.0, 201.0 / 255.0, 072.0 / 255.0] + [1]
        pb.changeVisualShape(cloth_id, -1, flags=pb.VISUAL_SHAPE_DOUBLE_SIDED,
                            rgbaColor=color)

        # For tracking IDs and consistency with existing ravens code.
        self._IDs[cloth_id] = 'cloth'
        #self.object_points[cloth_id] = np.float32((0, 0, 0)).reshape(3, 1)
        #env.objects.append(cloth_id)

        # To help environment pick-place method track all deformables.
        #self.def_IDs.append(cloth_id)

        # Sanity checks.
        nb_vertices, _ = pb.getMeshData(cloth_id, -1, flags=pb.MESH_DATA_SIMULATION_MESH)
        assert nb_vertices == self.n_cuts * self.n_cuts
        return cloth_id



if __name__ == "__main__":
    robot = PandaArm()
    pb.resetSimulation(pb.RESET_USE_DEFORMABLE_WORLD)
    add_PyB_models_to_path()
    pb.setTimeStep(1/1080)
    bag_class = BagEnv()
    bag_id = bag_class.add_bag([0.5, 0.3, 0.0], [0, 0, -0.707, 0.707])
    counttt = 0
    plane = pb.loadURDF('plane.urdf')
    while counttt < 1000:
        pb.stepSimulation()
        counttt += 1
    a = pb.getMeshData(bag_id)
    print('meshdata is here')
    b = np.array(a[1])
    print(b.shape)
    c = np.mean(b, axis = 0)
    print(c)
    bag_class.understand_bag_top_ring([1.5, 0.3, 0.0])
    bag_class.add_cable_ring(bag_id = bag_id)

    while True:
        pb.stepSimulation()
    



    #clothenv = ClothEnv()
    #cloth_id = clothenv.add_cloth([0.5, 0.3, 0.0], [0, 0, -0.707, 0.707])

    
    #table = pb.loadURDF('table/table.urdf',
    #                    useFixedBase=True, globalScaling=0.5)
    #cube = pb.loadURDF('cube_small.urdf', useFixedBase=True, globalScaling=1.)
    trialID = pb.loadURDF("/home/clearlab/pybullet_robot/demos/pybullet_robot/robots/bullet_panda/models/trial.urdf",
                        [0.0, 0.0, 0.0],
                        # p.getQuaternionFromEuler([0, 1.5706453, 0]),
                        pb.getQuaternionFromEuler([0, 0, 0]),
                        useFixedBase = True,
                        flags= pb.URDF_MERGE_FIXED_LINKS | pb.URDF_USE_SELF_COLLISION)
    pb.resetBasePositionAndOrientation(
        trialID, [0.2, 0.3, 0.0], [0, 0, -0.707, 0.707])

    pb.resetBasePositionAndOrientation(bag_id, [0.3, 0.0, 2.0], [0, 0, -0.707, 0.707])

    #objects = {'plane': plane,
    #           'table': trialID}

    #world = SimpleWorld(robot, objects)
    pb.changeDynamics(world.objects.table, -1,
                      lateralFriction=0.1, restitution=0.9)
    #slow_rate = 100.

    #goal_pos, goal_ori = world.robot.ee_pose()

    #controller = OSHybridController(robot, objects)

    #print ("started")

    #z_traj = np.linspace(goal_pos[2], 0.3, 550)
    #print(z_traj)

    #plot_t = threading.Thread(target=plot_thread)
    fx_deque = deque([0],maxlen=1000)
    fy_deque = deque([0],maxlen=1000)
    fz_deque = deque([0],maxlen=1000)

    controller.start_controller_thread()

    done = False
    #plot_t.start()

    i = 0
    f_ctrl = True
    while i < z_traj.size:
        now = time.time()

        ee_pos, _ = world.robot.ee_pose()
        wrench = world.robot.get_ee_wrench(local=False)
        # print wrench
        if abs(wrench[2]) >= 10.:
            break

        goal_pos[2] = z_traj[i]

        controller.update_goal(goal_pos, goal_ori)

        fx_deque.append(wrench[0])
        fy_deque.append(wrench[1])
        fz_deque.append(wrench[2])
        
        elapsed = time.time() - now
        sleep_time = (1./slow_rate) - elapsed
        #if sleep_time > 0.0:
            #time.sleep(sleep_time)

        i += 1
        pb.applyExternalForce(bag_id, -1, [5,0,20],[0,0,0],pb.LINK_FRAME)
        
    else:
        print ("Never reached force threshold for switching controller")
        
        
    
    if f_ctrl:
    
        print ("Switching to force control along Z axis")
        #y_traj = np.linspace(goal_pos[1], goal_pos[1]-0.2, 400)
        #define half a circular trajectory of radius 0.2
        '''
        x_traj = np.linspace(goal_pos[0], goal_pos[0]+0.4, 800)
        x_traj_tmp = goal_pos[0] + 0.2 -x_traj
        y_traj = np.sqrt(0.04 - (np.square(x_traj_tmp)))
        tmp = np.isnan(y_traj)
        y_traj[tmp] = 0.0

        y_traj2 = -y_traj
        x_traj2 = np.sort(x_traj)[::-1]
        x_traj = np.concatenate((x_traj, x_traj2))
        y_traj = np.concatenate((y_traj, y_traj2))
        '''
        '''
        x_traj = np.linspace(goal_pos[0], goal_pos[0]+0.4, 400)
        x_traj2 = np.sort(x_traj)[::-1]
        x_traj = np.concatenate((x_traj, x_traj2))
        for i in range(10):
            x_traj = np.concatenate((x_traj, x_traj))
        '''
        y_traj = np.linspace(goal_pos[1], goal_pos[1]+0.2, 400)
        y_traj2 = np.sort(y_traj)[::-1]
        y_traj = np.concatenate((y_traj, y_traj2))
        y_traj2 = -y_traj
        y_traj = np.concatenate((y_traj, y_traj2))

        x_traj = np.ones(y_traj.shape)
        x_traj = x_traj * goal_pos[0] + 0.05
        #print(x_traj)

        for i in range(10):
            print('yes')
            y_traj = np.concatenate((y_traj, y_traj))
            curr_x_traj = x_traj + 0.05
            #print(curr_x_traj)
            x_traj = np.concatenate((x_traj, curr_x_traj))

        

        
        
        controller.change_ft_directions([0,0,1,0,0,0])
        target_force = -20

        p_slider = pb.addUserDebugParameter('p_f',0.1,2.,controller._P_ft[2, 2])
        i_slider = pb.addUserDebugParameter('i_f',0.0,100.,controller._I_ft[2, 2])
        w_slider = pb.addUserDebugParameter('windup',0.0,100.,controller._windup_guard[2, 0])

        
        i = 0

        countt = 0
        table_vel = 0.05
        max_height = 0
        while i < y_traj.size-1:
            now = time.time()
            #print(goal_pos)
            ee_pos, _ = world.robot.ee_pose()
            wrench = world.robot.get_ee_wrench(local=False)
            # print wrench
            print(wrench)
            goal_pos[1] = y_traj[i]
            goal_pos[0] = x_traj[i]

            controller._P_ft[2, 2] = pb.readUserDebugParameter(p_slider)
            controller._I_ft[2, 2] = pb.readUserDebugParameter(i_slider)
            controller._windup_guard[2, 0] = pb.readUserDebugParameter(w_slider)
    
            #get link state for the table
            table_LS = pb.getLinkState(trialID, 0)
            table_height = table_LS[0][2]
            
            if table_height > 0.40:
                '''
                old_table_vel = table_vel
                table_vel = 0
                countt += 1
                print(countt)
                '''
                table_vel = -table_vel
                max_height = 1

            if table_height < 0.22 and max_height == 1:
                table_vel = -table_vel
                max_height = 0
                
            pb.applyExternalForce(bag_id, -1, [5,0,20],[0,0,0],pb.LINK_FRAME)
            '''
            if countt == 10000:
                total_maxmin = reach_max + reach_min
                table_vel = -old_table_vel
                countt = 0
            '''
            #print('table_height')
            #print(goal_pos[0])
            
            controller.update_goal(
                goal_pos, goal_ori, np.asarray([0., 0., target_force]), table_vel = table_vel)

            fx_deque.append(wrench[0])
            fy_deque.append(wrench[1])
            fz_deque.append(wrench[2])

            elapsed = time.time() - now
            sleep_time = (1./slow_rate) - elapsed
            if sleep_time > 0.0:
                time.sleep(sleep_time)

            if i < y_traj.size-1:
                i += 1
            #print(y_traj.size)
            #print(wrench[2])
        #print(fz_deque)
    #print('i am here')
    controller.stop_controller_thread()
    #print('i am here')
    done = True
    #print('i am here')
    #plot_t.join()
