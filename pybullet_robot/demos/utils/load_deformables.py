import pybullet as p
from time import sleep
import numpy as np
import os


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
        self._bag_scale = 3
        self._zone_length = (20. * self._zone_scale)
        self.zone_size = (20. * self._zone_scale, 20. * self._zone_scale,  0.0)
        self._bag_size = (  1. * self._bag_scale,   1. * self._bag_scale, 0.01)

        # Bag type (or resolution?) and parameters.
        self._bag = 4
        self._mass = 1.0
        self._scale = 0.25
        self._collisionMargin = 0.003
        self._base_orn = [np.pi / 2.0, 0.0, 0.0]
        #self._f_bag = '/home/clearlab/pybullet_robot/demos/assets/bags/bl_sphere_bag_rad_1.0_zthresh_0.6_numV_353.obj'
        self._f_bag = '/home/clearlab/pybullet_robot/demos/assets/bags/bag_with_smallopening2.obj'
        self._drop_height = 5000.50



    def add_bag(self, base_pos, base_orn, bag_color='yellow'):
        """Adding a bag from an .obj file."""
        bag_id = p.loadSoftBody(
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
                useSelfCollision=1,
                frictionCoeff=0.5,
                useFaceContact=1)

        # Only if using more recent PyBullet versions.
        '''
        p_version = pkg_resources.get_distribution('pybullet').version
        '''
        #if p_version == '3.0.4':
        color = [237.0 / 255.0, 201.0 / 255.0, 072.0 / 255.0] + [1]
        p.changeVisualShape(bag_id, -1, flags=p.VISUAL_SHAPE_DOUBLE_SIDED,
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
        radius = 0.05
        color = [078.0 / 255.0, 121.0 / 255.0, 167.0 / 255.0] + [1]
        beads = []
        bead_positions_l = []
        part_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[radius]*3)
        part_visual = p.createVisualShape(p.GEOM_SPHERE, radius=radius*1.5)

        # All tasks OTHER than bag-color-goal use self.bag_id. So if we are not
        # passing it in as an argument, we better have defined self.bag_id to use.
        if bag_id is None:
            bag_id = self.bag_id

        # Fortunately `verts_l` coincides with `self._top_ring_idxs`.
        _, verts_l = p.getMeshData(bag_id, -1, flags=p.MESH_DATA_SIMULATION_MESH)

        # Iterate through parts and create constraints as needed.
        for i in range(num_parts):
            bag_vidx = self._top_ring_idxs[i]
            bead_position = np.float32(verts_l[bag_vidx])
            part_id = p.createMultiBody(0.01, part_shape, part_visual,
                    basePosition=bead_position)
            p.changeVisualShape(part_id, -1, rgbaColor=color)

            if i > 0:
                parent_frame = bead_position - bead_positions_l[-1]
                constraint_id = p.createConstraint(
                        parentBodyUniqueId=beads[-1],
                        parentLinkIndex=-1,
                        childBodyUniqueId=part_id,
                        childLinkIndex=-1,
                        jointType=p.JOINT_POINT2POINT,
                        jointAxis=(0, 0, 0),
                        parentFramePosition=parent_frame,
                        childFramePosition=(0, 0, 0))
                p.changeConstraint(constraint_id, maxForce=100)

            # Make a constraint with i=0. Careful with `parent_frame`!
            if i == num_parts - 1:
                parent_frame = bead_positions_l[0] - bead_position
                constraint_id = p.createConstraint(
                        parentBodyUniqueId=part_id,
                        parentLinkIndex=-1,
                        childBodyUniqueId=beads[0],
                        childLinkIndex=-1,
                        jointType=p.JOINT_POINT2POINT,
                        jointAxis=(0, 0, 0),
                        parentFramePosition=parent_frame,
                        childFramePosition=(0, 0, 0))
                p.changeConstraint(constraint_id, maxForce=100)

            # Create constraint between a bead and certain bag vertices.
            _ = p.createSoftBodyAnchor(
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

    def add_cylinder_base(self, bag_id=None):
        lowest_idx = 86
        #num_parts = len(self._top_ring_idxs)
        radius = 0.25
        color = [078.0 / 255.0, 121.0 / 255.0, 167.0 / 255.0] + [1]
        #beads = []
        #bead_positions_l = []
        part_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius*3, height=0.25)
        #part_visual = pb.createVisualShape(pb.GEOM_SPHERE, radius=radius*1.5)

        # All tasks OTHER than bag-color-goal use self.bag_id. So if we are not
        # passing it in as an argument, we better have defined self.bag_id to use.
        if bag_id is None:
            bag_id = self.bag_id

        # Fortunately `verts_l` coincides with `self._top_ring_idxs`.
        _, verts_l = p.getMeshData(bag_id, -1, flags=p.MESH_DATA_SIMULATION_MESH)
        np.save('test22.npy', verts_l)
        verts_array = np.array(verts_l)
        verts_x = verts_array[:,0]
        verts_y = verts_array[:,1]
        verts_z = verts_array[:,2]

        min_idx_x = np.argmin(verts_x)
        min_idx_y = np.argmin(verts_y)
        min_idx_z = np.argmin(verts_z)


        # Iterate through parts and create constraints as needed.
      
        bag_vidx = min_idx_x
        bead_position = np.float32(verts_l[bag_vidx]) + np.array([0.25,0,0])
        part_id = p.createMultiBody(5, part_shape, -1,
                basePosition=bead_position, baseOrientation= p.getQuaternionFromEuler([0,1.57,0]))
        p.changeVisualShape(part_id, -1, rgbaColor=color)

        #aa = np.array([13,  24,  35,  46,  57,  68,  79,  90, 101, 112, 123, 134, 145, 157, 168, 179, 190, 201, 212, 223, 234, 245, 256, 267, 278, 289,300, 311, 322, 333, 344, 350])
        aa = np.array([ 11,  22,  33,  44,  55,  66,  77,  88,  99, 110, 121, 132, 143, 155, 166, 177, 188, 199, 210, 221, 232, 243, 254, 265, 276, 287,298, 309, 320, 331, 342, 348])
        # Create constraint between a bead and certain bag vertices.
        _ = p.createSoftBodyAnchor(
                softBodyBodyUniqueId=bag_id,
                nodeIndex=bag_vidx, 
                bodyUniqueId=part_id,
                linkIndex=-1)

        for i in aa:
            _ = p.createSoftBodyAnchor(
                softBodyBodyUniqueId=bag_id,
                nodeIndex=i, 
                bodyUniqueId=part_id,
                linkIndex=-1)

        # Track beads.
        #beads.append(part_id)
        #bead_positions_l.append(bead_position)

            # The usual for tracking IDs. Four things to add.
            #self.cable_bead_IDs.append(part_id)
            #self._IDs[part_id] = f'cable_part_{str(part_id).zfill(2)}'
            #env.objects.append(part_id)
            #self.object_points[part_id] = np.float32((0, 0, 0)).reshape(3, 1)

    def add_zip_base(self, bag_id=None):
        lowest_idx = 86
        #num_parts = len(self._top_ring_idxs)
        radius = 0.05
        color = [078.0 / 255.0, 121.0 / 255.0, 167.0 / 255.0] + [1]
        #beads = []
        #bead_positions_l = []
        part_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.3,0.01,0.1])

        #zip_base = p.loadURDF('/home/clearlab/pybullet_robot/demos/assets/bags/zipbase.urdf')
        
        #part_visual = pb.createVisualShape(pb.GEOM_SPHERE, radius=radius*1.5)

        # All tasks OTHER than bag-color-goal use self.bag_id. So if we are not
        # passing it in as an argument, we better have defined self.bag_id to use.
        if bag_id is None:
            bag_id = self.bag_id

        # Fortunately `verts_l` coincides with `self._top_ring_idxs`.
        _, verts_l = p.getMeshData(bag_id, -1, flags=p.MESH_DATA_SIMULATION_MESH)
        #np.save('test22.npy', verts_l)
        verts_array = np.array(verts_l)
        verts_x = verts_array[:,0]
        verts_y = verts_array[:,1]
        verts_z = verts_array[:,2]

        min_idx_x = np.argmin(verts_x)
        min_idx_y = np.argmin(verts_y)
        min_idx_z = np.argmin(verts_z)


        
        mean_x = (np.max(verts_x) + np.min(verts_x))/2
        # Find vertex nearest to mean_x
        delta_x = np.abs(verts_x - mean_x)
        some_x_idx = np.where(delta_x < 0.2)

        # Check the z coordinate of the vertex
        some_z = verts_z[some_x_idx[0]]
        delta_z = np.abs(some_z - np.max(some_z))
        some_z_idx = np.where(delta_z < 0.1)

        #check the y coordinate of the vertex
        some_y = verts_y[some_x_idx[0]]
        some_yy = some_y[some_z_idx]

        #check which is on the left and which on the right
        mean_y = np.mean(some_yy)
        some_yyy = some_yy - mean_y
        some_yyy_tmp = some_yyy
        some_yyy_tmp_idx = np.where(some_yyy > 0)
        some_yyy_tmp[some_yyy_tmp_idx] = 10000
        left_idx_tmp = np.argmin(some_yyy_tmp)

        some_yyy = some_yy - mean_y
        some_yyy_tmp = some_yyy
        some_yyy_tmp_idx = np.where(some_yyy < 0)
        some_yyy_tmp[some_yyy_tmp_idx] = -10000
        right_idx_tmp = np.argmax(some_yyy_tmp)

        #Get the index of the left and right vertex
        left_idx_tmp = some_z_idx[0][left_idx_tmp]
        right_idx_tmp = some_z_idx[0][right_idx_tmp]
        left_idx = some_x_idx[0][left_idx_tmp]
        right_idx  = some_x_idx[0][right_idx_tmp]

        

        # Iterate through parts and create constraints as needed.
      
        bag_vidx = min_idx_x
        
        bead_position = np.float32(verts_l[right_idx]) + np.array([0,0,0])
        #part_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=radius*1.5)
        #bead_position = np.float32(np.array([10.6,2,7]))
        '''
        part_id = p.createMultiBody(0.001, part_shape, -1,
                basePosition=bead_position, baseOrientation= p.getQuaternionFromEuler([0,0,0]))
        p.changeVisualShape(part_id, -1, rgbaColor=color)
        '''

        part_id = p.loadURDF('/home/clearlab/pybullet_robot/demos/assets/bags/zipbase.urdf')
        p.resetBasePositionAndOrientation(part_id, posObj=bead_position, ornObj=p.getQuaternionFromEuler([0.1,0,3.14]))

        bead_position = np.float32(verts_l[left_idx]) + np.array([-0.2,0.02,0])

        part_id1 = p.loadURDF('/home/clearlab/pybullet_robot/demos/assets/bags/zipbase.urdf')
        p.resetBasePositionAndOrientation(part_id1, posObj=bead_position, ornObj=p.getQuaternionFromEuler([0,0,0]))


        '''
        part_id1 = p.createMultiBody(0.001, part_shape, -1,
                basePosition=bead_position, baseOrientation= p.getQuaternionFromEuler([0,0,0]))
        p.changeVisualShape(part_id1, -1, rgbaColor=color)
        '''

        #aa = np.array([13,  24,  35,  46,  57,  68,  79,  90, 101, 112, 123, 134, 145, 157, 168, 179, 190, 201, 212, 223, 234, 245, 256, 267, 278, 289,300, 311, 322, 333, 344, 350])
        #aa = np.array([ 11,  22,  33,  44,  55,  66,  77,  88,  99, 110, 121, 132, 143, 155, 166, 177, 188, 199, 210, 221, 232, 243, 254, 265, 276, 287,298, 309, 320, 331, 342, 348])
        # Create constraint between a bead and certain bag vertices.
        _ = p.createSoftBodyAnchor(
                softBodyBodyUniqueId=bag_id,
                nodeIndex=right_idx, 
                bodyUniqueId=part_id,
                linkIndex=-1)

        _ = p.createSoftBodyAnchor(
                softBodyBodyUniqueId=bag_id,
                nodeIndex=left_idx, 
                bodyUniqueId=part_id1,
                linkIndex=-1)

        ######################
        # find all index near to left and right idx

        aa = np.abs(verts_y - verts_y[left_idx]) < 0.03
        bb = np.abs(verts_z - verts_z[left_idx]) < 0.03
        cc = np.logical_and(aa,bb)

        dd = np.where(cc)

        
        for i in dd[0]:
            _ = p.createSoftBodyAnchor(
                softBodyBodyUniqueId=bag_id,
                nodeIndex=i, 
                bodyUniqueId=part_id1,
                linkIndex=-1)


        aa = np.abs(verts_y - verts_y[right_idx]) < 0.03
        bb = np.abs(verts_z - verts_z[right_idx]) < 0.03
        cc = np.logical_and(aa,bb)

        dd = np.where(cc)

        for i in dd[0]:
            _ = p.createSoftBodyAnchor(
                softBodyBodyUniqueId=bag_id,
                nodeIndex=i, 
                bodyUniqueId=part_id,
                linkIndex=-1)

        #bead_position = bead_position + np.array([0,0,-0.5])
        #p.resetBasePositionAndOrientation(part_id1, posObj=bead_position, ornObj=p.getQuaternionFromEuler([0,0,0]))

        zip_id = p.loadURDF('/home/clearlab/pybullet_robot/demos/assets/bags/zip.urdf')
        #Find location to spawn zip
        avg_zip = ((np.array(verts_l[left_idx]) + np.array(verts_l[right_idx])) /2) + np.array([0,0,0.065])
        p.resetBasePositionAndOrientation(zip_id, posObj=avg_zip, ornObj=p.getQuaternionFromEuler([1.57,0,0]))


        #constraint to prismatic joint
        p.createConstraint(part_id, -1,zip_id, -1, p.JOINT_PRISMATIC, jointAxis=[1,0,0], parentFramePosition= [0,0,0], childFramePosition= [0,0,0])


        part_id2 = p.loadURDF('/home/clearlab/pybullet_robot/demos/assets/bags/zipbase.urdf', useFixedBase = 1)
        p.resetBasePositionAndOrientation(part_id2, posObj=np.array([10,0,0.2]), ornObj=p.getQuaternionFromEuler([0,0,3.14]))

        part_id3 = p.loadURDF('/home/clearlab/pybullet_robot/demos/assets/bags/zipbase.urdf', useFixedBase = 1)
        p.resetBasePositionAndOrientation(part_id3, posObj=np.array([10,-0.08,0.2]), ornObj=p.getQuaternionFromEuler([0,0,0]))

        zip_id1 = p.loadURDF('/home/clearlab/pybullet_robot/demos/assets/bags/zip.urdf')
        p.resetBasePositionAndOrientation(zip_id1, posObj=np.array([10,-0.04,0.27]), ornObj=p.getQuaternionFromEuler([1.57,0 ,0 ]))

        p.createConstraint(part_id2, 0,zip_id1, 0, p.JOINT_PRISMATIC, jointAxis=[1,0,0], parentFramePosition= [0,0,0], childFramePosition= [0,0,0])
        #p.createConstraint(part_id3, -1,zip_id1, 2, p.JOINT_PRISMATIC, jointAxis=[1,0,0], parentFramePosition= [0,0,0], childFramePosition= [0,0,0])




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


class ToteBag():
    def __init__(self):
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
        #self._f_bag = BAGS_TO_FILES[self._bag]
        self._drop_height = 5000.50


    def add_bag(self, base_pos, base_orn):
        deform_id = p.loadSoftBody(
                mass=1,  # 1kg is default; bad sim with lower mass
                fileName='/home/clearlab/pybullet_robot/demos/assets/bags/backpack_4.obj',
                scale=3,
                basePosition=base_pos,
                baseOrientation=base_orn,
                springElasticStiffness=100,
                springDampingStiffness=0.01,
                springBendingStiffness=10,
                frictionCoeff=0.1,
                # collisionMargin=0.003,  # how far apart do two objects begin interacting
                useSelfCollision=0,
                springDampingAllDirections=1,
                useFaceContact=True,
                useNeoHookean=0,
                useMassSpring=True,
                useBendingSprings=True,)
                # repulsionStiffness=10000000,)
        
        
        #texture_id = pb.loadTexture(plane_texture)

        return deform_id
    
    def change_texture(self, deform_id):
        texture_id = p.loadTexture('/home/clearlab/dedo/dedo/data/textures/deform/pb_white_knit.jpg')
        p.changeVisualShape(deform_id, -1, rgbaColor=[1,1,1,1], textureUniqueId=texture_id)