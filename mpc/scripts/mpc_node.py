#!/usr/bin/env python3
import math
from dataclasses import dataclass, field
import cvxpy
import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDrive, AckermannDriveStamped
from geometry_msgs.msg import PoseStamped, Point
from rclpy.node import Node
from scipy.linalg import block_diag
from scipy.sparse import block_diag, csc_matrix, diags
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import LaserScan
import utils
from numpy import linalg as LA
from visualization_msgs.msg import Marker, MarkerArray

# TODO CHECK: include needed ROS msg type headers and libraries


@dataclass
class mpc_config:
    NXK: int = 4  # length of kinematic state vector: z = [x, y, v, yaw]
    NU: int = 2  # length of input vector: u = [steering speed, acceleration]
    TK: int = 8  # finite time horizon length kinematic

    # ---------------------------------------------------
    # TODO: you may need to tune the following matrices
    Rk: list = field(
        default_factory=lambda: np.diag([0.01, 1.0])
    )  # input cost matrix, penalty for inputs - [accel, steering_speed]
    Rdk: list = field(
        default_factory=lambda: np.diag([0.01, 1.0])
    )  # input difference cost matrix, penalty for change of inputs - [accel, steering_speed]
    Qk: list = field(
        default_factory=lambda: np.diag([13.5, 13.5, 5.5, 13.0])
    )  # state error cost matrix, for the the next (T) prediction time steps [x, y, delta, v, yaw, yaw-rate, beta]
    Qfk: list = field(
        default_factory=lambda: np.diag([13.5, 13.5, 5.5, 13.0])
    )  # final state error matrix, penalty  for the final state constraints: [x, y, delta, v, yaw, yaw-rate, beta]
    # ---------------------------------------------------

    N_IND_SEARCH: int = 20  # Search index number
    DTK: float = 0.1  # time step [s] kinematic
    dlk: float = 0.03  # dist step [m] kinematic
    LENGTH: float = 0.58  # Length of the vehicle [m]
    WIDTH: float = 0.31  # Width of the vehicle [m]
    WB: float = 0.33  # Wheelbase [m]
    MIN_STEER: float = -0.4189  # maximum steering angle [rad]
    MAX_STEER: float = 0.4189  # maximum steering angle [rad]
    MAX_DSTEER: float = np.deg2rad(180.0)  # maximum steering speed [rad/s]
    MAX_SPEED: float = 6.0  # maximum speed [m/s]
    MIN_SPEED: float = 0.0  # minimum backward speed [m/s]
    MAX_ACCEL: float = 3.0  # maximum acceleration [m/ss]


@dataclass
class State:
    x: float = 0.0
    y: float = 0.0
    delta: float = 0.0
    v: float = 0.0
    yaw: float = 0.0
    yawrate: float = 0.0
    beta: float = 0.0

class MPC(Node):
    """ 
    Implement Kinematic MPC on the car
    This is just a template, you are free to implement your own node!
    """
    def __init__(self):
        super().__init__('mpc_node')
        # TODO: create ROS subscribers and publishers
        #       use the MPC as a tracker (similar to pure pursuit)
        
        # publishers
        self.drive_pub_ = self.create_publisher(AckermannDriveStamped, '/drive', 1) 
        self.ref_path_vis_pub_ = self.create_publisher(Marker, "/ref_path_vis", 1)
        self.pred_path_vis_pub_ = self.create_publisher(Marker, "/pred_path_vis", 1)
        self.waypoints_vis_pub_ = self.create_publisher(MarkerArray, "/waypoints", 1)

        # subscribers
        self.pose_sub_ = self.create_subscription(PoseStamped, '/pf/viz/inferred_pose', self.pose_callback, 1)

        # # visualization vars
        # self.marker_id = 1
     
        # TODO: get waypoints here
        # self.declare_parameter("waypoints_path", "/sim_ws/src/mpc/mpc/waypoints/waypoints_mpc.csv")
        # self.declare_parameter("waypoints_path", "/home/manasa/sim_ws/src/mpc/waypoints/waypoints_mpc.csv")
        self.declare_parameter("waypoints_path", "/home/nicholas/Documents/upenn/ESE615/sim_ws/src/lab-7-model-predictive-control-hot-wheels/mpc/waypoints/waypoints_mpc.csv")
        self.waypoints_path = self.get_parameter("waypoints_path").get_parameter_value().string_value
        print("waypoints path:", self.waypoints_path)
        self.waypoints_vis = MarkerArray()
        self.waypoints = self.get_waypoints(self.waypoints_path)

        self.config = mpc_config()
        self.odelta_v = None
        self.odelta = None
        self.oa = None
        self.init_flag = 0

        # initialize MPC problem
        self.mpc_prob_init()

    def pose_callback(self, pose_msg):

        # TODO: extract pose from ROS msg
        xp = pose_msg.pose.position.x
        yp = pose_msg.pose.position.y

        # Quaternion to Euler 
        q = [pose_msg.pose.orientation.x, pose_msg.pose.orientation.y, pose_msg.pose.orientation.z, pose_msg.pose.orientation.w]
        # num = 2 * (q[0] * q[1] + q[2] * q[3]) 
        # den = 1 - 2 * (q[1] * q[1] + q[2] * q[2])
        # yawp = atan2(num, den)

        quat = Rotation.from_quat(q)
        euler = quat.as_euler("zxy", degrees=False)
        yawp = euler[0] + np.pi

        # lin_speed = [pose_msg.twist.twist.x, pose_msg.twist.twist.y, pose_msg.twist.twist.z]
        # vp = LA.norm(lin_speed, 2)
        vp = 1.0
        vehicle_state = State(x=xp, y=yp, v=vp, yaw=yawp)

        # TODO: Calculate the next reference trajectory for the next T steps
        #       with current vehicle pose.
        #       ref_x, ref_y, ref_yaw, ref_v are columns of self.waypoints
        ref_x, ref_y, ref_yaw, ref_v = self.waypoints[:, 0], self.waypoints[:, 1], self.waypoints[:, 2], self.waypoints[:, 3]
        ref_path = self.calc_ref_trajectory(vehicle_state, ref_x, ref_y, ref_yaw, ref_v)
        # print("ref_path: ", ref_path)
        x0 = [vehicle_state.x, vehicle_state.y, vehicle_state.yaw, vehicle_state.v]

        # TODO: solve the MPC control problem
        (
            self.oa, # acceleration
            self.odelta_v, # steering angle
            ox, # state x
            oy, # state y
            oyaw, # yaw
            ov, # speed
            state_predict, # the predicted path for x steps
        ) = self.linear_mpc_control(ref_path, x0, self.oa, self.odelta_v)

        print("self.oa: ", self.oa)
        print("self.odelta_v: ", self.odelta_v)
        print("self.ox: ", ox)
        print("oy: ", oy)
        print("oyaw: ", oyaw)
        print("ov: ", ov)
        # print("state_predict: ", state_predict)
        mpc_path_vis = Marker()
        mpc_path_vis.header.frame_id = "map"
        mpc_path_vis.color.a = 1.0
        mpc_path_vis.color.r = 0.0
        mpc_path_vis.color.g = 1.0
        mpc_path_vis.color.b = 0.0
        mpc_path_vis.type = Marker.LINE_STRIP
        mpc_path_vis.scale.x = 0.1
        mpc_path_vis.id = 1000

        for i in range(len(ox)):
            mpc_path_vis.points.append(Point(x=ox[i], y=oy[i], z=0.0))

        # TODO: publish drive message.
  
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle = self.odelta_v[0]
        drive_msg.drive.speed = vehicle_state.v + self.oa[0] * self.config.DTK
        self.drive_pub_.publish(drive_msg)
        self.pred_path_vis_pub_.publish(mpc_path_vis)

    def mpc_prob_init(self):
        """
        Create MPC quadratic optimization problem using cvxpy, solver: OSQP
        Will be solved every iteration for control.
        More MPC problem information here: https://osqp.org/docs/examples/mpc.html
        More QP example in CVXPY here: https://www.cvxpy.org/examples/basic/quadratic_program.html
        """
        # Initialize and create vectors for the optimization problem
        # Vehicle State Vector
        self.xk = cvxpy.Variable(
            (self.config.NXK, self.config.TK + 1)
        )
        # Control Input vector
        self.uk = cvxpy.Variable(
            (self.config.NU, self.config.TK)
        )
        objective = 0.0  # Objective value of the optimization problem
        constraints = []  # Create constraints array

        # Initialize reference vectors
        self.x0k = cvxpy.Parameter((self.config.NXK,))
        self.x0k.value = np.zeros((self.config.NXK,))

        # Initialize reference trajectory parameter
        self.ref_traj_k = cvxpy.Parameter((self.config.NXK, self.config.TK + 1))
        self.ref_traj_k.value = np.zeros((self.config.NXK, self.config.TK + 1))

        # Initializes block diagonal form of R = [R, R, ..., R] (NU*T, NU*T)
        R_block = block_diag(tuple([self.config.Rk] * self.config.TK))

        # Initializes block diagonal form of Rd = [Rd, ..., Rd] (NU*(T-1), NU*(T-1))
        Rd_block = block_diag(tuple([self.config.Rdk] * (self.config.TK - 1)))

        # Initializes block diagonal form of Q = [Q, Q, ..., Qf] (NX*T, NX*T)
        Q_block = [self.config.Qk] * (self.config.TK)
        Q_block.append(self.config.Qfk)
        Q_block = block_diag(tuple(Q_block))

        # print("R_block: ", R_block.shape)   #(16, 16)
        # print("uk: ", self.uk.shape)        #(2, 8)
        # print("Rd_block: ", Rd_block.shape) #(14, 14)
        # print("xk: ", self.xk.shape)        #(4, 9)
        # print("Q_block: ", Q_block.shape)   #(36, 36)
        # print("ref_traj_k: ", self.ref_traj_k.shape) #(4, 9)

        # Formulate and create the finite-horizon optimal control problem (objective function)
        # The FTOCP has the horizon of T timesteps

        # --------------------------------------------------------
        # TODO: fill in the objectives here, you should be using cvxpy.quad_form() somewhere

        # TODO: Objective part 1: Influence of the control inputs: Inputs u multiplied by the penalty R
        objective += cvxpy.quad_form(cvxpy.reshape(self.uk, (16, 1), order='F'), R_block) 
        # TODO: Objective part 2: Deviation of the vehicle from the reference trajectory weighted by Q, including final Timestep T weighted by Qf
        objective += cvxpy.quad_form((cvxpy.reshape(self.ref_traj_k, (36, 1), order='F') - cvxpy.reshape(self.xk, (36, 1), order='F')), Q_block)
        # TODO: Objective part 3: Difference from one control input to the next control input weighted by Rd
        objective += cvxpy.quad_form(cvxpy.reshape((self.uk[:, 1:] - self.uk[:, :-1]), (14, 1), order='F'), Rd_block)


        # --------------------------------------------------------

        # Constraints 1: Calculate the future vehicle behavior/states based on the vehicle dynamics model matrices
        # Evaluate vehicle Dynamics for next T timesteps
        A_block = []
        B_block = []
        C_block = []
        # init path to zeros
        path_predict = np.zeros((self.config.NXK, self.config.TK + 1))
        for t in range(self.config.TK):
            A, B, C = self.get_model_matrix(
                path_predict[2, t], path_predict[3, t], 0.0
            )
            A_block.append(A)
            B_block.append(B)
            C_block.extend(C)

        A_block = block_diag(tuple(A_block))
        B_block = block_diag(tuple(B_block))
        C_block = np.array(C_block)

        # [AA] Sparse matrix to CVX parameter for proper stuffing
        # Reference: https://github.com/cvxpy/cvxpy/issues/1159#issuecomment-718925710
        m, n = A_block.shape
        self.Annz_k = cvxpy.Parameter(A_block.nnz)
        data = np.ones(self.Annz_k.size)
        rows = A_block.row * n + A_block.col
        cols = np.arange(self.Annz_k.size)
        Indexer = csc_matrix((data, (rows, cols)), shape=(m * n, self.Annz_k.size))

        # Setting sparse matrix data
        self.Annz_k.value = A_block.data

        # Now we use this sparse version instead of the old A_ block matrix
        self.Ak_ = cvxpy.reshape(Indexer @ self.Annz_k, (m, n), order="C")

        # Same as A
        m, n = B_block.shape
        self.Bnnz_k = cvxpy.Parameter(B_block.nnz)
        data = np.ones(self.Bnnz_k.size)
        rows = B_block.row * n + B_block.col
        cols = np.arange(self.Bnnz_k.size)
        Indexer = csc_matrix((data, (rows, cols)), shape=(m * n, self.Bnnz_k.size))
        self.Bk_ = cvxpy.reshape(Indexer @ self.Bnnz_k, (m, n), order="C")
        self.Bnnz_k.value = B_block.data

        # No need for sparse matrices for C as most values are parameters
        self.Ck_ = cvxpy.Parameter(C_block.shape)
        self.Ck_.value = C_block

        # -------------------------------------------------------------
        # TODO: Constraint part 1:
        #       Add dynamics constraints to the optimization problem
        #       This constraint should be based on a few variables:
        #       self.xk, self.Ak_, self.Bk_, self.uk, and self.Ck_
        # print("Ak_block: ", self.Ak_.shape) # (32, 32)
        # print("xk: ", self.xk.shape)        # (4, 9)
        # print("Bk_block: ", self.Bk_.shape) # (32, 16)
        # print("uk: ", self.uk.shape)        # (2, 8)
        # print("Ck_block: ", self.Ck_.shape) # (32, )
  
        constraints += [cvxpy.reshape(self.xk[:, 1:], (32, 1), order='F') == self.Ak_ @ cvxpy.reshape(self.xk[:, 1:], (32, 1), order='F') +
                         self.Bk_ @ cvxpy.reshape(self.uk, (16, 1), order='F') + cvxpy.reshape(self.Ck_, (32, 1))]
        # TODO: Constraint part 2:
        #       Add constraints on steering, change in steering angle
        #       cannot exceed steering angle speed limit. Should be based on:
        #       self.uk, self.config.MAX_DSTEER, self.config.DTK
        # constraints += [
        #     (self.uk[1, 1:]- self.uk[1, :-1])/self.config.DTK <= self.config.MAX_DSTEER, # max steering speed
        # ]
        
        # TODO: Constraint part 3:
        #       Add constraints on upper and lower bounds of states and inputs
        #       and initial state constraint, should be based on:
        #       self.xk, self.x0k, self.config.MAX_SPEED, self.config.MIN_SPEED,
        #       self.uk, self.config.MAX_ACCEL, self.config.MAX_STEER
        constraints += [
            self.xk[3, :] <= self.config.MAX_SPEED, # max speed
            self.xk[3, :] >= self.config.MIN_SPEED, # min speed
            self.xk[:, 0] == self.x0k, # initial state
            self.uk[0, :] <= self.config.MAX_ACCEL, # max acceleration
            # self.uk[0, :] >= -self.config.MAX_ACCEL, # min acceleration
            self.uk[0, :] >= 0, # min acceleration
            self.uk[1, :] <= self.config.MAX_STEER, # max steering
            self.uk[1, :] >= self.config.MIN_STEER, # max steering
        ]
        # -------------------------------------------------------------

        # Create the optimization problem in CVXPY and setup the workspace
        # Optimization goal: minimize the objective function
        
        self.MPC_prob = cvxpy.Problem(cvxpy.Minimize(objective), constraints)


    def calc_ref_trajectory(self, state, cx, cy, cyaw, sp):
        """
        calc referent trajectory ref_traj in T steps: [x, y, v, yaw]
        using the current velocity, calc the T points along the reference path
        :param cx: Course X-Position
        :param cy: Course y-Position
        :param cyaw: Course Heading
        :param sp: speed profile
        :dl: distance step
        :pind: Setpoint Index
        :return: reference trajectory ref_traj, reference steering angle
        """

        # Create placeholder Arrays for the reference trajectory for T steps
        ref_traj = np.zeros((self.config.NXK, self.config.TK + 1)) # (4x9)
        ncourse = len(cx)

        # Find nearest index/setpoint from where the trajectories are calculated
        # TODO: Check if there is a bug here since it may be finding the nearest point backwards
        # is time taken into consideration?
        _, _, _, ind = utils.nearest_point(np.array([state.x, state.y]), np.array([cx, cy]).T)

        # Load the initial parameters from the setpoint into the trajectory
        ref_traj[0, 0] = cx[ind]
        ref_traj[1, 0] = cy[ind]
        ref_traj[2, 0] = sp[ind]
        ref_traj[3, 0] = cyaw[ind]

        # based on current velocity, distance traveled on the ref line between time steps
        travel = abs(state.v) * self.config.DTK
        dind = travel / self.config.dlk
        ind_list = int(ind) + np.insert(
            np.cumsum(np.repeat(dind, self.config.TK)), 0, 0
        ).astype(int)
        ind_list[ind_list >= ncourse] -= ncourse
        ref_traj[0, :] = cx[ind_list]
        ref_traj[1, :] = cy[ind_list]
        ref_traj[2, :] = sp[ind_list]
        cyaw[cyaw - state.yaw > 4.5] = np.abs(
            cyaw[cyaw - state.yaw > 4.5] - (2 * np.pi)
        )
        cyaw[cyaw - state.yaw < -4.5] = np.abs(
            cyaw[cyaw - state.yaw < -4.5] + (2 * np.pi)
        )
        ref_traj[3, :] = cyaw[ind_list]
        
        self.visualize_ref_traj(ref_traj)

        return ref_traj

    def visualize_ref_traj(self, ref_traj):
        """
        A method used simply to visualze the computed reference trajectory
        for the mpc control problem.

        Inputs:
            ref_traj: reference trajectory ref_traj, reference steering angle
                      [x, y, v, yaw]
        """
        ref_strip = Marker()
        ref_strip.header.frame_id = "map"
        ref_strip.ns = "ref_traj"
        ref_strip.id = 10
        ref_strip.type = Marker.LINE_STRIP
        ref_strip.action = Marker.ADD
        ref_strip.scale.x = 0.2
        ref_strip.color.a = 0.4
        ref_strip.color.r = 1.0
        ref_strip.color.g = 0.0
        ref_strip.color.b = 1.0


        # make a list of points from the ref_traj
        ref_strip.points.clear()
        for i in range(ref_traj.shape[1]):
            # p = Point(ref_traj[0, i], ref_traj[1, i])
            p = Point()
            p.x = ref_traj[0, i]
            p.y = ref_traj[1, i]
            ref_strip.points.append(p)

        self.ref_path_vis_pub_.publish(ref_strip)
        # self.waypoints_vis_pub_.publish(self.waypoints_vis)



    def predict_motion(self, x0, oa, od, xref):
        path_predict = xref * 0.0
        for i, _ in enumerate(x0):
            path_predict[i, 0] = x0[i] 

        state = State(x=x0[0], y=x0[1], yaw=x0[3], v=x0[2])
        for (ai, di, i) in zip(oa, od, range(1, self.config.TK + 1)):
            state = self.update_state(state, ai, di)
            path_predict[0, i] = state.x
            path_predict[1, i] = state.y
            path_predict[2, i] = state.v
            path_predict[3, i] = state.yaw

        return path_predict

    def update_state(self, state, a, delta):

        # input check
        if delta >= self.config.MAX_STEER:
            delta = self.config.MAX_STEER
        elif delta <= -self.config.MAX_STEER:
            delta = -self.config.MAX_STEER

        state.x = state.x + state.v * math.cos(state.yaw) * self.config.DTK
        state.y = state.y + state.v * math.sin(state.yaw) * self.config.DTK
        state.yaw = (
            state.yaw + (state.v / self.config.WB) * math.tan(delta) * self.config.DTK
        )
        state.v = state.v + a * self.config.DTK

        if state.v > self.config.MAX_SPEED:
            state.v = self.config.MAX_SPEED
        elif state.v < self.config.MIN_SPEED:
            state.v = self.config.MIN_SPEED

   
        return state

    def get_model_matrix(self, v, phi, delta):
        """
        Calc linear and discrete time dynamic model-> Explicit discrete time-invariant
        Linear System: Xdot = Ax + Bu + C
        State vector: x=[x, y, v, yaw]
        :param v: speed
        :param phi: heading angle of the vehicle
        :param delta: steering angle: delta_bar
        :return: A, B, C
        """

        # State (or system) matrix A, 4x4
        A = np.zeros((self.config.NXK, self.config.NXK))
        A[0, 0] = 1.0
        A[1, 1] = 1.0
        A[2, 2] = 1.0
        A[3, 3] = 1.0
        A[0, 2] = self.config.DTK * math.cos(phi)
        A[0, 3] = -self.config.DTK * v * math.sin(phi)
        A[1, 2] = self.config.DTK * math.sin(phi)
        A[1, 3] = self.config.DTK * v * math.cos(phi)
        A[3, 2] = self.config.DTK * math.tan(delta) / self.config.WB

        # Input Matrix B: 4x2
        B = np.zeros((self.config.NXK, self.config.NU))
        B[2, 0] = self.config.DTK
        B[3, 1] = self.config.DTK * v / (self.config.WB * math.cos(delta) ** 2)

        C = np.zeros(self.config.NXK)
        C[0] = self.config.DTK * v * math.sin(phi) * phi
        C[1] = -self.config.DTK * v * math.cos(phi) * phi
        C[3] = -self.config.DTK * v * delta / (self.config.WB * math.cos(delta) ** 2)

        return A, B, C

    def mpc_prob_solve(self, ref_traj, path_predict, x0):
        self.x0k.value = x0

        A_block = []
        B_block = []
        C_block = []
        for t in range(self.config.TK):
            A, B, C = self.get_model_matrix(
                path_predict[2, t], path_predict[3, t], 0.0
            )
            A_block.append(A)
            B_block.append(B)
            C_block.extend(C)

        A_block = block_diag(tuple(A_block))
        B_block = block_diag(tuple(B_block))
        C_block = np.array(C_block)

        self.Annz_k.value = A_block.data
        self.Bnnz_k.value = B_block.data
        self.Ck_.value = C_block

        self.ref_traj_k.value = ref_traj

        # Solve the optimization problem in CVXPY
        # Solver selections: cvxpy.OSQP; cvxpy.GUROBI
        self.MPC_prob.solve(solver=cvxpy.OSQP, verbose=False, warm_start=True)

        if (
            self.MPC_prob.status == cvxpy.OPTIMAL
            or self.MPC_prob.status == cvxpy.OPTIMAL_INACCURATE
        ):
            ox = np.array(self.xk.value[0, :]).flatten()
            oy = np.array(self.xk.value[1, :]).flatten()
            ov = np.array(self.xk.value[2, :]).flatten()
            oyaw = np.array(self.xk.value[3, :]).flatten()
            oa = np.array(self.uk.value[0, :]).flatten()
            odelta = np.array(self.uk.value[1, :]).flatten()

        else:
            print("Error: Cannot solve mpc..")
            oa, odelta, ox, oy, oyaw, ov = None, None, None, None, None, None

        return oa, odelta, ox, oy, oyaw, ov

    def linear_mpc_control(self, ref_path, x0, oa, od):
        """
        MPC contorl with updating operational point iteraitvely
        :param ref_path: reference trajectory in T steps
        :param x0: initial state vector
        :param oa: acceleration of T steps of last time
        :param od: delta of T steps of last time
        """

        # print("Entering linear_mpc_control")
        if oa is None or od is None:
            oa = [0.0] * self.config.TK
            od = [0.0] * self.config.TK

        # Call the Motion Prediction function: Predict the vehicle motion for x-steps
        path_predict = self.predict_motion(x0, oa, od, ref_path)
        poa, pod = oa[:], od[:]

        # Run the MPC optimization: Create and solve the optimization problem
        mpc_a, mpc_delta, mpc_x, mpc_y, mpc_yaw, mpc_v = self.mpc_prob_solve(
            ref_path, path_predict, x0
        )

        return mpc_a, mpc_delta, mpc_x, mpc_y, mpc_yaw, mpc_v, path_predict
    
    def get_waypoints(self, path):
        # Get the waypoints from the path
        waypoints = np.empty((1,4))
        f = open(path, 'r')
        # waypoints_viz = MarkerArray()
        # waypoints_viz = Marker()
        marker_id = 1
        
        # self.wpt.action = self.wpt.ADD

        while(True):

            line = f.readline()
            if(not line):
                break
            x, y, yaw, v = line.split(',')
            v = 1.0
            waypoints = np.vstack((waypoints, np.array([float(x), float(y),float(yaw), float(v)]).reshape(1,4)))

            wpt = Marker()
            wpt.id = marker_id 
            wpt.type = Marker.SPHERE
            wpt.ns = "waypoints"
            wpt.scale.x = 0.15
            wpt.scale.y = 0.15
            wpt.scale.z = 0.15
            wpt.color.a = 0.01
            wpt.color.r = 0.0
            wpt.color.g = 0.0
            wpt.color.b = 1.0
            wpt.header.frame_id = "map"
            wpt.pose.position.x = float(x)
            wpt.pose.position.y = float(y)
            

            self.waypoints_vis.markers.append(wpt)
            # self.wpt.points.append(p)

            marker_id += 1


        waypoints = waypoints[1:]
        self.waypoints_vis_pub_.publish(self.waypoints_vis)
        # self.waypoints_vis_pub_.publish(self.wpt)

        return waypoints

def main(args=None):
    rclpy.init(args=args)
    print("MPC Initialized")
    mpc_node = MPC()
    rclpy.spin(mpc_node)

    mpc_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()