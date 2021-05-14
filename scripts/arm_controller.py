#! /usr/bin/env python
import rospy
import numpy as np
from copy import deepcopy
import time
from scipy.interpolate import InterpolatedUnivariateSpline

from ur_kinematics.ur_kin_py import forward
from kinematics import analytical_ik, nearest_ik_solution

from std_msgs.msg import Float64MultiArray, Header
from sensor_msgs.msg import JointState
from ur5teleop.msg import jointdata, Joint
from ur_dashboard_msgs.msg import SafetyMode
from ur_dashboard_msgs.srv import IsProgramRunning, GetSafetyMode
from std_msgs.msg import Bool

#TODO this is unused except in initialization. Change this behavior
control_arm_saved_zero = np.array([0.51031649, 1.22624958, 3.31996918, 0.93126088, 3.1199832, 9.78404331])

two_pi = np.pi*2

# gripper_collision_points =  np.array([[0.04, 0.0, -0.21, 1.0], #fingertip
#                                       [0.05, 0.04, 0.09,  1.0],  #hydraulic outputs
#                                       [0.05, -0.04, 0.09,  1.0]]).T
class ur5e_arm():
    '''Defines velocity based controller for ur5e arm for use in teleop project
    '''
    safety_mode = -1
    shutdown = False
    enabled = False
    joint_reorder = [2,1,0,3,4,5]
    breaking_stop_time = 0.1 #when stoping safely, executes the stop in 0.1s Do not make large!

    #read in settings
    #TODO add helpful error messages
    encoder_type = rospy.get_param("/encoder_type")
    floor_type = rospy.get_param("/floor_type")
    floor_height = rospy.get_param("/floor_height") if rospy.has_param('/floor_height') else None
    encoder_profiles = rospy.get_param("/encoder_profiles")
    joint_lims = rospy.get_param("/joint_lims")
    conservative_joint_lims_enabled = rospy.get_param("/conservative_joint_lims")
    gripper_collision_points = np.array(rospy.get_param("/gripper_collision_points"))
    assert(gripper_collision_points.shape[1] == 3)
    #reshape for multiplication with 4x4 pose matrix
    gripper_collision_points = np.vstack((gripper_collision_points.T, np.ones((1,gripper_collision_points.shape[0]))))
    # max_joint_speeds = np.array([3.0, 3.0, 3.0, 3.0, 3.0, 3.0])
    max_joint_speeds = np.array(rospy.get_param("/max_joint_speeds"))
    z_axis_lim_config = rospy.get_param("/z_axis_lim_config")
    #check user entered settings
    assert(floor_type in z_axis_lim_config.keys())
    assert(encoder_type in encoder_profiles.keys())

    #set the floor height variable
    #keepout (limmited to z axis height for now)
    # self.keepout_enabled = True
    # self.z_axis_lim = -0.37 # floor 0.095 #short table # #0.0 #table
    if not floor_height is None:
        assert(type(floor_height) is float)
        keepout_enabled = True
        z_axis_lim = floor_height
    else:
        keepout_enabled = False if floor_type == 'none' else True
        z_axis_lim = z_axis_lim_config[floor_type]


    #joint inversion - accounts for encoder axes being inverted inconsistently
    joint_inversion = encoder_profiles[encoder_type]['joint_inversion']

    #throws an error and stops the arm if there is a position discontinuity in the
    #encoder input freater than the specified threshold
    #with the current settings of 100hz sampling, 0.1 radiands corresponds to
    #~10 rps velocity, which is unlikely to happen unless the encoder input is wrong
    # position_jump_error = 0.1
    position_jump_error = rospy.get_param("/position_jump_error")

    #read in gains
    joint_p_gains = np.array(encoder_profiles[encoder_type]['gains']['joint_p_gains']) #works up to at least 20 on wrist 3
    joint_ff_gains = np.array(encoder_profiles[encoder_type]['gains']['joint_ff_gains'])
    # joint_p_gains = np.array([5.0, 5.0, 5.0, 10.0, 10.0, 10.0]) #works up to at least 20 on wrist 3
    # joint_ff_gains = np.array([0.0, 0.0, 0.0, 1.0, 1.1, 1.1])


    default_pos = (np.pi/180)*np.array(rospy.get_param("/default_pos"))
    robot_ref_pos = deepcopy(default_pos)
    saved_ref_pos = None

    # lower_lims = (np.pi/180)*np.array([0.0, -120.0, 0.0, -180.0, -180.0, 90.0])
    # upper_lims = (np.pi/180)*np.array([180.0, 0.0, 175.0, 0.0, 0.0, 270.0])
    # conservative_lower_lims = (np.pi/180)*np.array([45.0, -100.0, 45.0, -135.0, -135.0, 135.0])
    # conservative_upper_lims = (np.pi/180)*np.array([135, -45.0, 140.0, -45.0, -45.0, 225.0])
    if not conservative_joint_lims_enabled:
        lower_lims = (np.pi/180)*np.array(joint_lims['lower_lims'])
        upper_lims = (np.pi/180)*np.array(joint_lims['upper_lims'])
    else:
        lower_lims = (np.pi/180)*np.array(joint_lims['conservative_lower_lims'])
        upper_lims = (np.pi/180)*np.array(joint_lims['conservative_upper_lims'])

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

    def __init__(self):
        '''set up controller class variables & parameters'''

        #launch nodes
        rospy.init_node('teleop_controller', anonymous=True)
        #start subscribers
        # if test_control_signal:
        #     print('Running in test mode ... no daq input')
        #     self.test_control_signal = test_control_signal
        # else:
        rospy.Subscriber("daqdata_filtered", jointdata, self.daq_callback)

        #start robot state subscriber (detects fault or estop press)
        rospy.Subscriber('/ur_hardware_interface/safety_mode',SafetyMode, self.safety_callback)
        #joint feedback subscriber
        rospy.Subscriber("joint_states", JointState, self.joint_state_callback)
        #service to check if robot program is running
        rospy.wait_for_service('/ur_hardware_interface/dashboard/program_running')
        self.remote_control_running = rospy.ServiceProxy('ur_hardware_interface/dashboard/program_running', IsProgramRunning)
        #service to check safety mode
        rospy.wait_for_service('/ur_hardware_interface/dashboard/get_safety_mode')
        self.safety_mode_proxy = rospy.ServiceProxy('/ur_hardware_interface/dashboard/get_safety_mode', GetSafetyMode)
        #start subscriber for deadman enable
        rospy.Subscriber('/enable_move',Bool,self.enable_callback)

        #start vel publisher
        self.vel_pub = rospy.Publisher("/joint_group_vel_controller/command",
                            Float64MultiArray,
                            queue_size=1)

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

    def daq_callback(self, data):
        previous_positions = deepcopy(self.current_daq_positions)
        self.current_daq_positions[:] = [data.encoder1.pos, data.encoder2.pos, data.encoder3.pos, data.encoder4.pos, data.encoder5.pos, data.encoder6.pos]
        self.current_daq_velocities[:] = [data.encoder1.vel, data.encoder2.vel, data.encoder3.vel, data.encoder4.vel, data.encoder5.vel, data.encoder6.vel]
        self.current_daq_velocities *= self.joint_inversion #account for diferent conventions

        if not self.first_daq_callback and np.any(np.abs(self.current_daq_positions - previous_positions) > self.position_jump_error):
            print('stopping arm - encoder error!')
            print('Daq position change is too high')
            print('Previous Positions:\n{}'.format(previous_positions))
            print('New Positions:\n{}'.format(self.current_daq_positions))
            self.shutdown_safe()
        # np.subtract(,,out=) #update relative position
        self.current_daq_rel_positions = self.current_daq_positions - self.control_arm_ref_config
        self.current_daq_rel_positions *= self.joint_inversion
        self.current_daq_rel_positions_waraped = np.mod(self.current_daq_rel_positions+np.pi,two_pi)-np.pi
        self.first_daq_callback = False

    # def wrap_relative_angles(self):
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

    def enable_callback(self, data):
        '''Detects the software enable/disable safety switch'''
        self.enabled = data.data

    def user_wait_safety_stop(self):
        #wait for user to fix the stop
        while not self.safety_mode == 1:
            raw_input('Safety Stop or other stop condition enabled.\n Correct the fault, then hit enter to continue')

    def ensure_safety_mode(self):
        '''Blocks until the safety mode is 1 (normal)'''
        while not self.safety_mode == 1:
            raw_input('Robot safety mode is not normal, \ncheck the estop and correct any faults, then restart the external control program and hit enter. ')

    def get_safety_mode(self):
        '''Calls get safet mode service, does not return self.safety_mode, which is updated by the safety mode topic, but should be the same.'''
        return self.safety_mode_proxy().safety_mode.mode

    def ready_to_move(self):
        '''returns true if the safety mode is 1 (normal) and the remote program is running'''
        return self.get_safety_mode() == 1 and self.remote_control_running()

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
        if interactive:
            _ = raw_input("Hit enter when ready to set the control arm ref pos.")
        self.control_arm_ref_config = np.mod(deepcopy(self.current_daq_positions),np.pi*2)
        if reset_robot_ref_config_to_current:
            self.robot_ref_pos = deepcopy(self.current_joint_positions)
        print("Control Arm Ref Position Setpoint:\n{}\n".format(self.control_arm_def_config))

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
        # self.stop_arm()

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

    def remote_program_running(self):
        print('remote : ',self.remote_control_running().program_running)
        return self.remote_control_running().program_running

    def move_to_robost(self,
                position,
                speed = 0.25,
                error_thresh = 0.01,
                override_initial_joint_lims = False,
                require_enable = False):
        '''Calls the move_to method as necessary to ensure that the goal position
        is reached, accounting for interruptions due to safety faults, and the
        enable deadman if require_enable is selected'''

        if require_enable:
            print('Depress and hold the deadman switch when ready to move.')
            print('Release to stop')

        while not rospy.is_shutdown():
            #check safety
            if not self.ready_to_move():
                self.user_prompt_ready_to_move()
                continue
            #check enabled
            if not self.enabled:
                time.sleep(0.01)
                continue
            #start moving
            print('Starting Trajectory')
            result = self.move_to(position,
                                  speed = speed,
                                  error_thresh = error_thresh,
                                  override_initial_joint_lims = override_initial_joint_lims,
                                  require_enable = require_enable)
            if result:
                break
        print('Reached Goal')


    def move_to(self,
                position,
                speed = 0.25,
                error_thresh = 0.01,
                override_initial_joint_lims = False,
                require_enable = False):
        '''CAUTION - use joint lim override with extreme caution. Intended to
        allow movement from outside the lims back to acceptable position.

        Defines a simple joing controller to bring the arm to a desired
        configuration without teleop input. Intended for testing or to reach
        present initial positions, etc.'''

        #ensure safety sqitch is not enabled
        if not self.ready_to_move():
            self.user_prompt_ready_to_move()

        #define max speed slow for safety
        if speed > 0.5:
            print("Limiting speed to 0.5 rad/sec")
            speed = 0.5

        #calculate traj from current position
        start_pos = deepcopy(self.current_joint_positions)
        max_disp = np.max(np.abs(position-start_pos))
        end_time = max_disp/speed

        #make sure this is a valid joint position
        if not self.is_joint_position(position):
            print("Invalid Joint Position, Exiting move_to function")
            return False

        #check joint llims
        if not override_initial_joint_lims:
            if not self.identify_joint_lim(start_pos):
                print("Start Position Outside Joint Lims...")
                return False
        if not self.identify_joint_lim(position):
            print("Commanded Postion Outside Joint Lims...")
            return False


        print('Executing Move to : \n{}\nIn {} seconds'.format(position,end_time))
        #list of interpolators ... this is kind of dumb, there is probably a better solution
        traj = [InterpolatedUnivariateSpline([0.,end_time],[start_pos[i],position[i]],k=1) for i in range(6)]

        position_error = np.array([1.0]*6) #set high position error
        pos_ref = deepcopy(start_pos)
        rate = rospy.Rate(500) #lim loop to 500 hz
        start_time = time.time()
        reached_pos = False
        while not self.shutdown and not rospy.is_shutdown() and self.safety_mode == 1: #chutdown is set on ctrl-c.
            if require_enable and not self.enabled:
                print('Lost Enable, stopping')
                break

            loop_time = time.time()-start_time
            if loop_time < end_time:
                pos_ref[:] = [traj[i](loop_time) for i in range(6)]
            else:
                pos_ref = position
                # break
                if np.all(np.abs(position_error)<error_thresh):
                    print("reached target position")
                    self.stop_arm()
                    reached_pos = True
                    break

            position_error = pos_ref - self.current_joint_positions
            vel_ref_temp = self.joint_p_gains*position_error
            #enforce max velocity setting
            np.clip(vel_ref_temp,-self.max_joint_speeds,self.max_joint_speeds,vel_ref_temp)
            self.vel_ref.data = vel_ref_temp
            self.vel_pub.publish(self.vel_ref)
            # print(pos_ref)
            #wait
            rate.sleep()

        #make sure arm stops
        self.stop_arm(safe = True)
        return reached_pos

    def return_collison_free_config(self, reference_positon):
        '''takes the proposed set of joint positions for the real robot and
        checks the forward kinematics for collisions with the floor plane and the
        defined gripper points. Returns the neares position with the same orientation
        that is not violating the floor constraint.'''
        pose = forward(reference_positon)
        collision_positions = np.dot(pose, self.gripper_collision_points)

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

    def move(self,
             capture_start_as_ref_pos = False,
             dialoge_enabled = True):
        '''Main control loop for teleoperation use.'''
        if not self.ready_to_move():
            self.user_prompt_ready_to_move()

        max_pos_error = 0.5 #radians/sec
        low_joint_vel_lim = 0.5

        position_error = np.zeros(6)
        absolute_position_error = np.zeros(6)
        position_error_exceeded_by = np.zeros(6)
        vel_ref_array = np.zeros(6)
        ref_pos = deepcopy(self.current_joint_positions)
        rate = rospy.Rate(500)

        if capture_start_as_ref_pos:
            self.set_current_config_as_control_ref_config(interactive = dialoge_enabled)
            self.current_daq_rel_positions_waraped = np.zeros(6)
        print('safety_mode',self.safety_mode,self.enabled,self.shutdown)
        while not self.shutdown and self.safety_mode == 1 and self.enabled: #chutdown is set on ctrl-c.
            #get ref position inplace - avoids repeatedly declaring new array
            # np.add(self.default_pos,self.current_daq_rel_positions,out = ref_pos)
            np.add(self.robot_ref_pos,self.current_daq_rel_positions_waraped,out = ref_pos)
            #

            #enforce joint lims
            np.clip(ref_pos, self.lower_lims, self.upper_lims, ref_pos)

            #check that it is not hitting the table/floor
            if self.keepout_enabled:
                # #run forward kinematcs
                # pose = forward(ref_pos)
                # test_point_pos = np.dot(pose, test_point).reshape(-1)
                # if test_point_pos[2] < self.z_axis_lim:
                #     #saturate pose
                #     diff = pose[2,3] - test_point_pos[2]
                #     pose[2,3] = self.z_axis_lim + diff
                #     #get joint ref
                #     new_ref = nearest_ik_solution(analytical_ik(pose,self.upper_lims,self.lower_lims),self.current_joint_positions,threshold=0.2)
                #     ref_pos = new_ref
                ref_pos = self.return_collison_free_config(ref_pos)

            self.ref_pos.data = ref_pos
            self.daq_pos_pub.publish(self.ref_pos)

            #inplace error calculation
            np.subtract(ref_pos, self.current_joint_positions, position_error)


            #calculate vel signal
            np.multiply(position_error,self.joint_p_gains,out=vel_ref_array)
            vel_ref_array += self.joint_ff_gains*self.current_daq_velocities
            #enforce max velocity setting
            np.clip(vel_ref_array,-self.max_joint_speeds,self.max_joint_speeds,vel_ref_array)

            #publish
            self.vel_ref.data = vel_ref_array
            # self.ref_vel_pub.publish(self.vel_ref)
            self.vel_pub.publish(self.vel_ref)
            #wait
            rate.sleep()
        self.stop_arm(safe = True)

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
            #start moving
            print('Starting Free Movement')
            self.move(capture_start_as_ref_pos = True,
                      dialoge_enabled = False)



if __name__ == "__main__":

    print("starting")

    arm = ur5e_arm()
    time.sleep(1)
    arm.stop_arm()


    # print(arm.remote_control_running().program_running)
    # raw_input('waiting')
    # print(arm.remote_control_running().program_running)

    # arm.calibrate_control_arm_zero_position(interactive = True)
    # print(arm.move_to(arm.default_pos, speed = 0.1, override_initial_joint_lims=True))
    print(arm.move_to_robost(arm.default_pos,
                             speed = 0.1,
                             override_initial_joint_lims=True,
                             require_enable = True))


    # arm.capture_control_arm_ref_position()
    # print("Current Arm Position")
    # print(arm.current_joint_positions)
    # print("DAQ position:")
    # print(arm.current_daq_positions)
    # daq_pos = deepcopy(arm.current_daq_positions)
    # daq_offset = arm.current_daq_positions - arm.control_arm_def_config
    # print("DAQ offset from default pos: \n{}".format(daq_offset))
    # if 'y'==raw_input('Execute Move? (y/n)'):
    #     target_pos = arm.default_pos + daq_offset
    #     arm.move_to(target_pos, speed = 0.1, override_initial_joint_lims=False)
    pose = forward(arm.current_joint_positions)
    print(pose)
    raw_input("Hit enter when ready to move")
    # arm.move()
    # arm.move(capture_start_as_ref_pos=True)
    arm.run()

    arm.stop_arm()


    # target_pos = arm.default_pos
    # target_pos[5]+=1.0
    # arm.move_to(target_pos, speed = 0.1, override_initial_joint_lims=True)
