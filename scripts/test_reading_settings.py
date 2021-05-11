#! /usr/bin/env python
import rospy
import time
import numpy as np

if __name__=="__main__":
    node = rospy.init_node('test_reader', anonymous=True)
    time.sleep(1)

    data = rospy.get_param("/encoder_profile")
    print(data)
    print(data['digital'])

    gripper_collision_points = rospy.get_param("/gripper_collision_points")
    print(gripper_collision_points)
    points = np.array(gripper_collision_points)
    # points = np.hstack((points.T, np.array([[1.0, 1.0, 1.0]])))
    print(points)

    rospy.spin()
