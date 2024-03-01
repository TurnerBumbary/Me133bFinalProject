#!/usr/bin/env python3
#
#   noisylocalization.py
#
#   Simulate noisy odometry or drifting localization.
#
#   Node:       /localization
#
#   Subscribe:  /odom                   sensor_msgs/Lsaerscan
#   Publish:    -nothing-
#
#   TF Broadcast:  odom frame in map fram
#
import numpy as np

from math                       import pi, sin, cos, atan2, sqrt

# ROS Imports
import rclpy

from rclpy.node                 import Node
from rclpy.time                 import Time

from tf2_ros                    import TransformBroadcaster

from geometry_msgs.msg          import Point, Quaternion, Pose
from geometry_msgs.msg          import Transform, TransformStamped
from nav_msgs.msg               import Odometry


#
#   Global Definitions
#
R = 0.3                 # Radius to convert angle to distance.

NOISE = 0.12            # Noise fraction


#
#   Angle Wrapping
#
def wrapto90(angle):
    return angle -   pi * round(angle/(  pi))
def wrapto180(angle):
    return angle - 2*pi * round(angle/(2*pi))


#
#   Custom Node Class
#
class CustomNode(Node):
    # Initialization.
    def __init__(self, name):
        # Initialize the node, naming it as specified
        super().__init__(name)

        # Set the initial drift value.
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

        # Set the initial odometry reading.
        self.odom = None

        # Initialize the transform broadcaster
        self.tfBroadcaster = TransformBroadcaster(self)

        # Create a subscriber to the odometry topic.
        self.create_subscription(Odometry, '/odom', self.odomCB, 1)

    # Shutdown
    def shutdown(self):
        # No particular cleanup, just shut down the node.
        self.destroy_node()


    # Odometry CB.  See how far we've moved and drift accordingly.
    def odomCB(self, msg):
        # Check if we have an old value.
        if self.odom is None:
            self.odom = msg.pose.pose
            return

        # Grab the new odometry reading.
        x1 = msg.pose.pose.position.x
        y1 = msg.pose.pose.position.y
        t1 = 2 * atan2(msg.pose.pose.orientation.z,
                       msg.pose.pose.orientation.w)

        # Grab the old odometry reading.
        x0 = self.odom.position.x
        y0 = self.odom.position.y
        t0 = 2 * atan2(self.odom.orientation.z,
                       self.odom.orientation.w)

        # Save the new reading.
        self.odom = msg.pose.pose

        # Compute the magnitude of the difference.
        d = sqrt((x1-x0)**2 + (y1-y0)**2 + (R*wrapto180(t1-t0))**2)

        # Drift we want on the new position.
        if True:
            dx = np.random.uniform(-d  *NOISE, d  *NOISE)
            dy = np.random.uniform(-d  *NOISE, d  *NOISE)
            dt = np.random.uniform(-d/R*NOISE, d/R*NOISE)
        else:
            dx = -0.1 * (x1-x0)
            dy = -0.1 * (y1-y0)
            dt = -0.1 * wrapto180(t1-t0)

        # Matching drift to the odometry frame.
        dx += (1-cos(dt))*x1 + sin(dt)*y1
        dy += (1-cos(dt))*y1 - sin(dt)*x1

        # Update the odometry frame.
        self.x     += cos(self.theta) * dx - sin(self.theta) * dy
        self.y     += sin(self.theta) * dx + cos(self.theta) * dy
        self.theta += dt

        # Broadcast the new drift.
        tfmsg = TransformStamped()

        tfmsg.header.stamp            = msg.header.stamp
        tfmsg.header.frame_id         = 'map'
        tfmsg.child_frame_id          = 'odom'
        tfmsg.transform.translation.x = self.x
        tfmsg.transform.translation.y = self.y
        tfmsg.transform.rotation.z    = sin(self.theta/2)
        tfmsg.transform.rotation.w    = cos(self.theta/2)

        self.tfBroadcaster.sendTransform(tfmsg)

        #print('Sent transform (%6.3f, %6.3f, %6.3f)' %
        #      (self.x, self.y, self.theta))


#
#   Main Code
#
def main(args=None):
    # Initialize ROS.
    rclpy.init(args=args)

    # Instantiate the node.
    node = CustomNode('odometry')

    # Spin, until interrupted.
    rclpy.spin(node)

    # Shutdown the node and ROS.
    node.shutdown()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
