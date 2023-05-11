#! /usr/bin/env python
import rospy
import rospkg
import numpy as np
from copy import deepcopy
import time
from scipy.interpolate import InterpolatedUnivariateSpline

from filter import PythonBPF

from ur_kinematics.ur_kin_py import forward, forward_link
from kinematics import analytical_ik, nearest_ik_solution

from std_msgs.msg import Float64MultiArray, Header
from sensor_msgs.msg import JointState
from geometry_msgs.msg import WrenchStamped
from ur5teleop.msg import jointdata, Joint
from ur_dashboard_msgs.msg import SafetyMode
from ur_dashboard_msgs.srv import IsProgramRunning, GetSafetyMode
from std_msgs.msg import Bool

# Import the module
from urdf_parser_py.urdf import URDF
from pykdl_utils.kdl_parser import kdl_tree_from_urdf_model
from pykdl_utils.kdl_kinematics import KDLKinematics

r = rospkg.RosPack()
path = r.get_path('ur_teleop_controller')
robot = URDF.from_xml_file(path+"/config/ur5e.urdf")
dummy_arm = URDF.from_xml_file(path+"/config/dummy_arm.urdf")

# from controller_manager_msgs.srv import ListControllers
# from controller_manager_msgs.srv import SwitchController

joint_vel_lim = 1.0
sample_rate = 500
control_arm_saved_zero = np.array([0.51031649, 1.22624958, 3.31996918, 0.93126088, 3.1199832, 9.78404331])

#define initial state

#joint inversion - accounts for encoder axes being inverted inconsistently
joint_inversion = np.array([1,1,-1,1,1,1]) # digital encoder dummy arm
two_pi = np.pi*2
# # TODO Arm class
    #Breaking at shutdown
    #Follow traj
#test_point = np.array([0.04,0.0,-0.21,1]).reshape(-1,1)
# test_point = np.array([0.0,0.2,0.0,1]).reshape(-1,1)
gripper_collision_points =  np.array([[0.04, 0.0, -0.21, 1.0], #fingertip
                                      [0.05, 0.04, 0.09,  1.0],  #hydraulic outputs
                                      [0.05, -0.04, 0.09,  1.0]]).T

## TODO Move some of the params in to a configuration file
class ur5e_arm():
    '''Defines velocity based controller for ur5e arm for use in teleop project
    '''
    safety_mode = -1
    shutdown = False
    enabled = False
    jogging = False
    joint_reorder = [2,1,0,3,4,5]
    breaking_stop_time = 0.1 #when stoping safely, executes the stop in 0.1s Do not make large!

    #throws an error and stops the arm if there is a position discontinuity in the
    #encoder input freater than the specified threshold
    #with the current settings of 100hz sampling, 0.1 radiands corresponds to
    #~10 rps velocity, which is unlikely to happen unless the encoder input is wrong
    position_jump_error = 0.1
    # ains = np.array([10.0]*6) #works up to at least 20 on wrist 3
    joint_p_gains_varaible = np.array([5.0, 5.0, 5.0, 10.0, 10.0, 10.0]) #works up to at least 20 on wrist 3
    joint_ff_gains_varaible = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    default_pos = (np.pi/180)*np.array([90.0, -90.0, 90.0, -90.0, -45.0, 180.0])
    robot_ref_pos = deepcopy(default_pos)
    saved_ref_pos = None
    daq_ref_pos = deepcopy(default_pos)

    lower_lims = (np.pi/180)*np.array([5.0, -120.0, 5.0, -150.0, -175.0, 95.0])
    upper_lims = (np.pi/180)*np.array([175.0, 5.0, 175.0, 5.0, 5.0, 265.0])
    conservative_lower_lims = (np.pi/180)*np.array([45.0, -100.0, 45.0, -135.0, -135.0, 135.0])
    conservative_upper_lims = (np.pi/180)*np.array([135, -45.0, 140.0, -45.0, -45.0, 225.0])
    # max_joint_speeds = np.array([3.0, 3.0, 3.0, 3.0, 3.0, 3.0])
    max_joint_speeds = 3.0 * np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    max_joint_acc = 5.0 * np.array([1.0, 1.0, 1.0, 3.0, 3.0, 3.0])
    homing_joint_speeds = np.array([0.1, 0.1, 0.1, 0.2, 0.2, 0.2])
    jogging_joint_speeds = 2.0 * homing_joint_speeds
    homing_joint_acc = 2.0 * np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    # max_joint_speeds = np.array([3.0, 3.0, 3.0, 3.0, 3.0, 3.0])*0.1
    #default control arm setpoint - should be calibrated to be 1 to 1 with default_pos
    #the robot can use relative joint control, but this saved defailt state can
    #be used to return to a 1 to 1, absolute style control
    control_arm_def_config = np.mod(control_arm_saved_zero,np.pi*2)
    control_arm_ref_config = deepcopy(control_arm_def_config) #can be changed to allow relative motion

    #define fields that are updated by the subscriber callbacks
    current_joint_positions = np.zeros(6)
    current_joint_velocities = np.zeros(6)

    current_daq_positions = np.zeros(6)
    current_daq_velocities = np.zeros(6)
    #DEBUG
    current_daq_rel_positions = np.zeros(6) #current_daq_positions - control_arm_ref_config
    current_daq_rel_positions_waraped = np.zeros(6)

    first_daq_callback = True
    first_wrench_callback = True

    # define bandpass filter parameters
    fl = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    fh = [30.0, 30.0, 30.0, 30.0, 30.0, 30.0]
    fs = sample_rate
    filter = PythonBPF(fs, fl, fh)

    wrench = np.zeros(6)
    est_wrench_int_term = np.zeros(6)
    load_mass_publish_rate = 1
    load_mass_publish_index = 0

    ## control loop extra parameters
    current_wrench_global = np.zeros(6)
    current_joint_torque = np.zeros(6)

    joint_torque_error = np.zeros(6)
    wrench_global_error = np.zeros(6)

    joint_inertia = np.array([5.0, 5.0, 5.0, 1.0, 1.0, 1.0])
    joint_stiffness = 400 * np.array([0.6, 1.0, 1.0, 1.0, 1.0, 1.0])
    zeta = 1.0

    recent_data_focus_coeff = 0.99
    p = 1 / recent_data_focus_coeff

    last_command_joint_velocities = np.zeros(6)

    tree = kdl_tree_from_urdf_model(robot)
    # print tree.getNrOfSegments()
    # forwawrd kinematics
    chain = tree.getChain("base_link", "wrist_3_link")
    # print chain.getNrOfJoints()
    kdl_kin = KDLKinematics(robot, "base_link", "wrist_3_link")

    tree_op = kdl_tree_from_urdf_model(dummy_arm)
    chain_op = tree_op.getChain("base_link", "wrist_3_link")
    kdl_kin_op = KDLKinematics(dummy_arm, "base_link", "wrist_3_link")


    def __init__(self, test_control_signal = False, conservative_joint_lims = True):
        '''set up controller class variables & parameters'''

        if conservative_joint_lims:
            self.lower_lims = self.conservative_lower_lims
            self.upper_lims = self.conservative_upper_lims

        #keepout (limmited to z axis height for now)
        self.keepout_enabled = True
        self.z_axis_lim = -0.37 # floor 0.095 #short table # #0.0 #table

        #launch nodes
        rospy.init_node('teleop_controller', anonymous=True)
        #start subscribers
        if test_control_signal:
            print('Running in test mode ... no daq input')
            self.test_control_signal = test_control_signal
        else:
            rospy.Subscriber("daqdata_filtered", jointdata, self.daq_callback)

        #start robot state subscriber (detects fault or estop press)
        rospy.Subscriber('/ur_hardware_interface/safety_mode',SafetyMode, self.safety_callback)
        #joint feedback subscriber
        rospy.Subscriber("joint_states", JointState, self.joint_state_callback)
        #wrench feedback
        rospy.Subscriber("wrench", WrenchStamped, self.wrench_callback)
        #service to check if robot program is running
        rospy.wait_for_service('/ur_hardware_interface/dashboard/program_running')
        self.remote_control_running = rospy.ServiceProxy('ur_hardware_interface/dashboard/program_running', IsProgramRunning)
        #service to check safety mode
        rospy.wait_for_service('/ur_hardware_interface/dashboard/get_safety_mode')
        self.safety_mode_proxy = rospy.ServiceProxy('/ur_hardware_interface/dashboard/get_safety_mode', GetSafetyMode)
        #start subscriber for deadman enable
        rospy.Subscriber('/enable_move',Bool,self.enable_callback)
        #start subscriber for jogging enable
        rospy.Subscriber('/jogging',Bool,self.jogging_callback)

        #start vel publisher
        self.vel_pub = rospy.Publisher("/joint_group_vel_controller/command",
                            Float64MultiArray,
                            queue_size=1)

        #vertical direction force measurement publisher
        self.load_mass_pub = rospy.Publisher("/load_mass",
                            Float64MultiArray,
                            queue_size=1)
        self.test_data_pub = rospy.Publisher("/test_data",
                            Float64MultiArray,
                            queue_size=1)
        self.is_homing_pub = rospy.Publisher("/is_homing", Bool, queue_size=1)

        #ref pos publisher DEBUG
        self.daq_pos_pub = rospy.Publisher("/debug_ref_pos",
                            Float64MultiArray,
                            queue_size=1)
        self.daq_pos_wraped_pub = rospy.Publisher("/debug_ref_wraped_pos",
                            Float64MultiArray,
                            queue_size=1)
        self.ref_pos = Float64MultiArray(data=[0,0,0,0,0,0])
        #DEBUG
        # self.daq_pos_debug = Float64MultiArray(data=[0,0,0,0,0,0])
        # self.daq_pos_wraped_debug = Float64MultiArray(data=[0,0,0,0,0,0])

        #set shutdown safety behavior
        rospy.on_shutdown(self.shutdown_safe)
        time.sleep(0.5)
        self.stop_arm() #ensure arm is not moving if it was already

        self.velocity = Float64MultiArray(data = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.vel_ref = Float64MultiArray(data = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.load_mass = Float64MultiArray(data = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.test_data = Float64MultiArray(data = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.is_homing = False


        print("Joint Limmits: ")
        print(self.upper_lims)
        print(self.lower_lims)

        if not self.ready_to_move():
            print('User action needed before commands can be sent to the robot.')
            self.user_prompt_ready_to_move()
        else:
            print('Ready to move')

    def joint_state_callback(self, data):
        self.current_joint_positions[self.joint_reorder] = data.position
        self.current_joint_velocities[self.joint_reorder] = data.velocity

    def wrench_callback(self, data):
        self.current_wrench = np.array([data.wrench.force.x, data.wrench.force.y, data.wrench.force.z, data.wrench.torque.x, data.wrench.torque.y, data.wrench.torque.z])
        
        if self.first_wrench_callback:
            self.filter.calculate_initial_values(self.current_wrench)
        
        Ja = self.kdl_kin.jacobian(self.current_joint_positions)
        FK = self.kdl_kin.forward(self.current_joint_positions)
        p = FK[:3,3]
        R = FK[:3,:3]
        RT = np.transpose(R)

        Rp0 = np.cross(np.array(RT[:,0]).reshape(-1),np.array(p).reshape(-1))
        Rp1 = np.cross(np.array(RT[:,1]).reshape(-1),np.array(p).reshape(-1))
        Rp2 = np.cross(np.array(RT[:,2]).reshape(-1),np.array(p).reshape(-1))
        Rp = - np.array([Rp0, Rp1, Rp2])

        filtered_wrench = np.array(self.filter.filter(self.current_wrench))
        np.matmul(R, filtered_wrench[3:], out = self.current_wrench_global[3:])

        ## The second momentum calculation should be the correct one but needs to be tested
        # np.matmul(R, filtered_wrench[:3], out = self.current_wrench_global[:3])
        self.current_wrench_global[:3] = np.matmul(R, filtered_wrench[:3]) + np.matmul(Rp, filtered_wrench[3:])
        
        np.matmul(Ja.transpose(), self.current_wrench_global, out = self.current_joint_torque)
        self.first_wrench_callback = False

    def daq_callback(self, data):
        previous_positions = deepcopy(self.current_daq_positions)
        self.current_daq_positions[:] = [data.encoder1.pos, data.encoder2.pos, data.encoder3.pos, data.encoder4.pos, data.encoder5.pos, data.encoder6.pos]
        self.current_daq_velocities[:] = [data.encoder1.vel, data.encoder2.vel, data.encoder3.vel, data.encoder4.vel, data.encoder5.vel, data.encoder6.vel]
        self.current_daq_velocities *= joint_inversion #account for diferent conventions

        if not self.first_daq_callback and np.any(np.abs(self.current_daq_positions - previous_positions) > self.position_jump_error):
            print('stopping arm - encoder error!')
            print('Daq position change is too high')
            print('Previous Positions:\n{}'.format(previous_positions))
            print('New Positions:\n{}'.format(self.current_daq_positions))
            self.shutdown_safe()
        # np.subtract(,,out=) #update relative position
        self.current_daq_rel_positions = self.current_daq_positions - self.control_arm_ref_config
        self.current_daq_rel_positions *= joint_inversion
        self.current_daq_rel_positions_waraped = np.mod(self.current_daq_rel_positions+np.pi,two_pi)-np.pi
        self.first_daq_callback = False

    # def wrap_relative_angles(self):
    ## TODO Expand this to be a better exception handler 
    def safety_callback(self, data):
        '''Detect when safety stop is triggered'''
        self.safety_mode = data.mode
        if not data.mode == 1:
            #estop or protective stop triggered
            #send a breaking command
            print('\nFault Detected, sending stop command\n')
            self.stop_arm() #set commanded velocities to zero
            print('***Please clear the fault and restart the UR-Cap program before continuing***')

            #wait for user to fix the stop
            # self.user_wait_safety_stop()
    
    ## TODO Change function name, deadman_callback(self, data)
    def enable_callback(self, data):
        '''Detects the software enable/disable safety switch'''
        self.enabled = data.data

    ## DEPRECATED
    def jogging_callback(self, data):
        '''Detects the software jogging switch'''
        self.jogging = data.data

    ## TODO Merge this into exception handler
    def user_wait_safety_stop(self):
        #wait for user to fix the stop
        while not self.safety_mode == 1:
            raw_input('Safety Stop or other stop condition enabled.\n Correct the fault, then hit enter to continue')

    ## TODO Merge this into exception handler
    def ensure_safety_mode(self):
        '''Blocks until the safety mode is 1 (normal)'''
        while not self.safety_mode == 1:
            raw_input('Robot safety mode is not normal, \ncheck the estop and correct any faults, then restart the external control program and hit enter. ')

    def get_safety_mode(self):
        '''Calls get safet mode service, does not return self.safety_mode, which is updated by the safety mode topic, but should be the same.'''
        return self.safety_mode_proxy().safety_mode.mode

    ## TODO Safety handling
    def ready_to_move(self):
        '''returns true if the safety mode is 1 (normal) and the remote program is running'''
        return self.get_safety_mode() == 1 and self.remote_control_running()

    ## TODO Safety handling
    def user_prompt_ready_to_move(self):
        '''Blocking dialog to get the user to reset the safety warnings and start the remote program'''
        while True:
            if not self.get_safety_mode() == 1:
                print(self.get_safety_mode())
                raw_input('Safety mode is not Normal. Please correct the fault, then hit enter.')
            else:
                break
        while True:
            if not self.remote_control_running():
                raw_input('The remote control URCap program has been pause or was not started, please restart it, then hit enter.')
            else:
                break
        print('\nRemote control program is running, and safety mode is Normal\n')
    
    ## DEPRECATED
    def calibrate_control_arm_zero_position(self, interactive = True):
        '''Sets the control arm zero position to the current encoder joint states
        TODO: Write configuration to storage for future use'''
        if interactive:
            _ = raw_input("Hit enter when ready to save the control arm ref pos.")
        self.control_arm_def_config = np.mod(deepcopy(self.current_daq_positions),np.pi*2)
        self.control_arm_ref_config = deepcopy(self.control_arm_def_config)
        print("Control Arm Default Position Setpoint:\n{}\n".format(self.control_arm_def_config))

    def set_current_config_as_control_ref_config(self,
                                                 reset_robot_ref_config_to_current = True,
                                                 interactive = True):
        '''
        Initialize the encoder value of the dummy arm
        '''
        if interactive:
            _ = raw_input("Hit enter when ready to set the control arm ref pos.")
        self.control_arm_ref_config = np.mod(deepcopy(self.current_daq_positions),np.pi*2)
        if reset_robot_ref_config_to_current:
            self.robot_ref_pos = deepcopy(self.current_joint_positions)
        print("Control Arm Ref Position Setpoint:\n{}\n".format(self.control_arm_def_config))
    
    ## DEPRECATED
    def capture_control_arm_ref_position(self, interactive = True):
        '''Captures the current joint positions, and resolves encoder startup
        rollover issue. This adds increments of 2*pi to the control_arm_saved_zero
        to match the current joint positions to the actual saved position.'''
        max_acceptable_error = 0.6
        tries = 3
        for i in range(tries):
            if interactive:
                _ = raw_input("Hit enter when ready to capture the control arm ref pos. Try {}/{}".format(i+1,tries))
            #get current config
            control_arm_config = deepcopy(self.current_daq_positions)
            # print('Current DAQ Position:')
            # print(control_arm_config)
            #check if there is a significant error
            # config_variant1 = control_arm_config+2*np.pi
            # config_variant2 = control_arm_config-2*np.pi
            rot_offsets = [0, 2*np.pi, -2*np.pi]
            config_variants = [self.control_arm_def_config+off for off in rot_offsets]
            # error_set_1 = np.abs(self.control_arm_def_config - control_arm_config)
            # error_set_2 = np.abs(self.control_arm_def_config - config_variant1)
            # error_set_3 = np.abs(self.control_arm_def_config - config_variant2)
            error_sets = [np.abs(control_arm_config - var) for var in config_variants]
            print(error_sets)
            # print(error_set_2)
            # print(error_set_3)
            #if a 2*pi offset is a good match, reset the def_config to match
            error_too_great = [False]*6
            new_controll_config = deepcopy(self.control_arm_def_config)
            #TODO change behabior for base joint
            for joint in range(6):
                # configs = [control_arm_config, config_variant1, config_variant2]
                # offsets = [error_set_1[joint], error_set_2[joint], error_set_3[joint]]
                errors = [err[joint] for err in error_sets]
                min_error_idx = np.argmin(errors)
                if errors[min_error_idx]<max_acceptable_error:
                    new_controll_config[joint] = config_variants[min_error_idx][joint]
                    # new_controll_config[joint] = new_controll_config[joint]+rot_offsets[min_error_idx]
                else:
                    error_too_great[joint] = True
            if any(error_too_great):
                print('Excessive error. It may be necessary to recalibrate.')
                print('Make sure arm matches default config and try again.')
            else:
                print('Encoder Ref Capture successful.')
                print('New control arm config:\n{}'.format(new_controll_config))
                print('Updated from:')
                print(self.control_arm_def_config)
                time.sleep(1)
                self.control_arm_def_config = new_controll_config
                self.control_arm_ref_config = deepcopy(new_controll_config)
                break

    ## TODO Vanity check function
    def is_joint_position(self, position):
        '''Verifies that this is a 1dim numpy array with len 6'''
        if isinstance(position, np.ndarray):
            return position.ndim==1 and len(position)==6
        else:
            return False

    def shutdown_safe(self):
        '''Should ensure that the arm is brought to a stop before exiting'''
        self.shutdown = True
        print('Stopping -> Shutting Down')
        self.stop_arm()
        print('Stopped')

    def stop_arm(self, safe = False):
        '''Commands zero velocity until sure the arm is stopped. If safe is False
        commands immediate stop, if set to a positive value, will stop gradually'''

        if safe:
            loop_rate = rospy.Rate(200)
            start_time = time.time()
            start_vel = deepcopy(self.current_joint_velocities)
            max_accel = np.abs(start_vel/self.breaking_stop_time)
            vel_mask = np.ones(6)
            vel_mask[start_vel < 0.0] = -1
            while np.any(np.abs(self.current_joint_velocities)>0.0001) and not rospy.is_shutdown():
                command_vels = [0.0]*6
                loop_time = time.time() - start_time
                for joint in range(len(command_vels)):
                    vel = start_vel[joint] - vel_mask[joint]*max_accel[joint]*loop_time
                    if vel * vel_mask[joint] < 0:
                        vel = 0
                    command_vels[joint] = vel
                self.vel_pub.publish(Float64MultiArray(data = command_vels))
                if np.sum(command_vels) == 0:
                    break
                loop_rate.sleep()

        while np.any(np.abs(self.current_joint_velocities)>0.0001):
            self.vel_pub.publish(Float64MultiArray(data = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))

    ## TODO Vanity check function
    def in_joint_lims(self, position):
        '''expects an array of joint positions'''
        return np.all(self.lower_lims < position) and np.all(self.upper_lims > position)

    def identify_joint_lim(self, position):
        '''expects an array of joint positions. Prints a human readable list of
        joints that exceed limmits, if any'''
        if self.in_joint_lims(position):
            print("All joints ok")
            return True
        else:
            for i, pos in enumerate(position):
                if pos<self.lower_lims[i]:
                    print('Joint {}: Position {:.5} exceeds lower bound {:.5}'.format(i,pos,self.lower_lims[i]))
                if pos>self.upper_lims[i]:
                    print('Joint {}: Position {:.5} exceeds upper bound {:.5}'.format(i,pos,self.lower_lims[i]))
            return False

    ## DEPRECATED
    def remote_program_running(self):
        print('remote : ',self.remote_control_running().program_running)
        return self.remote_control_running().program_running

    # move_to() function based on joint admittance controller
    def move_to(self):
        result = False
        rate = rospy.Rate(sample_rate)
        self.init_joint_admittance_controller()
        while not rospy.is_shutdown():
            current_homing_pos = self.current_daq_positions * joint_inversion + self.daq_ref_pos
            if not self.identify_joint_lim(current_homing_pos):
                print('Homing desired position outside robot position limit. Please change dummy position')
                result = False
                break
            #check safety
            if not self.ready_to_move():
                self.user_prompt_ready_to_move()
                continue
            #check enabled
            if not self.enabled:
                result = False
                break
            
            self.joint_admittance_controller(ref_pos = current_homing_pos,
                                             max_speeds = self.jogging_joint_speeds,
                                             max_acc = self.homing_joint_acc)
            
            if not np.any(np.abs(current_homing_pos - self.current_joint_positions)>0.01):
                print("Home position reached")
                result = True
                break
            #wait
            rate.sleep()
        self.stop_arm(safe = True)
        self.vel_ref.data = np.array([0.0]*6)
        return result

    # homing() function: initialzing UR to default position
    def homing(self):
    
        rate = rospy.Rate(sample_rate)
        self.init_joint_admittance_controller()
        while not rospy.is_shutdown():
            #check safety
            if not self.ready_to_move():
                self.user_prompt_ready_to_move()
                continue
            #check enabled
            if not self.enabled:
                self.stop_arm(safe = True)
                self.vel_ref.data = np.array([0.0]*6)
                time.sleep(0.01)
                continue
            
            self.joint_admittance_controller(ref_pos = self.default_pos,
                                             max_speeds = self.homing_joint_speeds,
                                             max_acc = self.homing_joint_acc)

            if not np.any(np.abs(self.default_pos - self.current_joint_positions)>0.01):
                print("Home position reached")
                break

            #wait
            rate.sleep()
        
        self.stop_arm(safe = True)
        self.vel_ref.data = np.array([0.0]*6)

    ## TODO Refinement required
    def return_collison_free_config(self, reference_positon):
        '''takes the proposed set of joint positions for the real robot and
        checks the forward kinematics for collisions with the floor plane and the
        defined gripper points. Returns the neares position with the same orientation
        that is not violating the floor constraint.'''
        pose = forward(reference_positon)
        collision_positions = np.dot(pose, gripper_collision_points)

        min_point = np.argmin(collision_positions[2,:])
        collision = collision_positions[2,min_point] < self.z_axis_lim
        if collision:
            # print('Z axis overrun: {}'.format(pose[2,3]))
            #saturate pose
            diff = pose[2,3] - collision_positions[2][min_point]
            # print(diff)
            pose[2,3] = self.z_axis_lim + diff
            # pose[2,3] = self.z_axis_lim
            #get joint ref
            reference_positon = nearest_ik_solution(analytical_ik(pose,self.upper_lims,self.lower_lims),self.current_joint_positions,threshold=0.2)
        return reference_positon

    ## TODO Break down this into separate modular functions
    def move(self,
             capture_start_as_ref_pos = False,
             dialoge_enabled = True):
        '''Main control loop for teleoperation use.'''
        if not self.ready_to_move():
            self.user_prompt_ready_to_move()

        rate = rospy.Rate(sample_rate)
        self.init_joint_admittance_controller(capture_start_as_ref_pos, dialoge_enabled)

        while not self.shutdown and self.safety_mode == 1 and self.enabled: #shutdown is set on ctrl-c.
            self.joint_admittance_controller(ref_pos = self.current_daq_rel_positions_waraped + self.robot_ref_pos,
                                             max_speeds = self.max_joint_speeds,
                                             max_acc = self.max_joint_acc)

            #wait
            rate.sleep()
        self.stop_arm(safe = True)
        self.vel_ref.data = np.array([0.0]*6)

    # online torque error id
    def force_torque_error_estimation(self, position_error, force_error, force):
        if not np.any(np.abs(position_error)>0.01) and not np.any(np.abs(self.current_joint_velocities)>0.001):
            force_error += self.p / (1 + self.p) * (force - force_error)
            self.p = (self.p - self.p ** 2 / (1 + self.p)) / self.recent_data_focus_coeff
        return force_error
        
    # joint admittance controller initialization
    def init_joint_admittance_controller(self,
                                         capture_start_as_ref_pos = False,
                                         dialoge_enabled = True):
        
        self.joint_torque_error = deepcopy(self.current_joint_torque)
        self.vel_admittance = np.zeros(6)

        if capture_start_as_ref_pos:
            self.set_current_config_as_control_ref_config(interactive = dialoge_enabled)
            self.current_daq_rel_positions_waraped = np.zeros(6)

    # joint admittance controller
    def joint_admittance_controller(self,
                                    ref_pos,
                                    max_speeds,
                                    max_acc):
        np.clip(ref_pos, self.lower_lims, self.upper_lims, ref_pos)
        joint_pos_error = np.subtract(ref_pos, self.current_joint_positions)
        vel_ref_array = np.multiply(joint_pos_error, self.joint_p_gains_varaible)

        self.joint_torque_error = self.force_torque_error_estimation(joint_pos_error, self.joint_torque_error, self.current_joint_torque)
        joint_torque_after_correction = self.current_joint_torque - self.joint_torque_error

        acc = (joint_torque_after_correction + self.joint_stiffness * joint_pos_error
                + 0.5 * 2 * self.zeta * np.sqrt(self.joint_stiffness * self.joint_inertia) * (self.current_daq_velocities - self.current_joint_velocities)
                - 0.5 * 2 * self.zeta * np.sqrt(self.joint_stiffness * self.joint_inertia) * self.current_joint_velocities) / self.joint_inertia
        np.clip(acc, -max_acc, max_acc, acc)
        self.vel_admittance += acc / sample_rate

        vel_ref_array[0] += self.vel_admittance[0]
        vel_ref_array[1] += self.vel_admittance[1]
        vel_ref_array[2] += self.vel_admittance[2]

        np.clip(vel_ref_array, self.last_command_joint_velocities - max_acc / sample_rate, self.last_command_joint_velocities + max_acc / sample_rate, vel_ref_array)
        np.clip(vel_ref_array, -max_speeds, max_speeds, vel_ref_array)
        self.last_command_joint_velocities = vel_ref_array

        #publish
        self.vel_ref.data = vel_ref_array
        # self.ref_vel_pub.publish(self.vel_ref)
        self.vel_pub.publish(self.vel_ref)        

    # Cartesian admittance controller initialization

    # Cartesian admittance controller

    ## TODO Main loop, add saftey handling here
    def run(self):
        '''Run runs the move routine repeatedly, accounting for the
        enable/disable switch'''

        print('Put the control arm in start configuration.')
        print('Depress and hold the deadman switch when ready to move.')

        while not rospy.is_shutdown():
            #check safety
            if not self.safety_mode == 1:
                time.sleep(0.01)
                continue
            #check enabled
            if not self.enabled:
                time.sleep(0.01)
                continue
            # current_homing_pos = deepcopy(self.current_daq_positions * joint_inversion + self.daq_ref_pos)
            # if not self.identify_joint_lim(current_homing_pos):
            #     print('Homing desired position outside robot position limit. Please change dummy position')
            #     continue
            
            self.is_homing = True
            self.is_homing_pub.publish(self.is_homing)
            # reached_home = self.move_to(current_homing_pos,speed = 0.2,override_initial_joint_lims=True,require_enable = True)
            reached_home = self.move_to()
            self.is_homing = False
            self.is_homing_pub.publish(self.is_homing)
            if reached_home:
                #start moving
                print('Starting Free Movement')
                self.move(capture_start_as_ref_pos = True,
                          dialoge_enabled = False)



if __name__ == "__main__":
    #This script is included for testing purposes
    print("starting")

    arm = ur5e_arm(test_control_signal=False, conservative_joint_lims = False)
    time.sleep(1)
    arm.stop_arm()

    # print(arm.move_to_robost(arm.default_pos,
    #                          speed = 0.1,
    #                          override_initial_joint_lims=True,
    #                          require_enable = True))
    arm.homing()

    pose = forward(arm.current_joint_positions)
    print(pose)
    raw_input("Hit enter when ready to move")

    arm.run()

    arm.stop_arm()
