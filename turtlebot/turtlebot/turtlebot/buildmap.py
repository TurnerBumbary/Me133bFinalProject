#!/usr/bin/env python3
#
#   slowbuildmap.py
#
#   Build a map...
#
#   This explores the occupancy grid map.
#
#   This "slow" version explicitly consideres the timestamps, so it
#   can properly connect data even if things are significantly
#   slowed/delayed.
#
#   Node:       /buildmap
#
#   Subscribe:  /scan                   sensor_msgs/Lsaerscan
#   Publish:    /map                    nav_msgs/OccupancyGrid
#
import numpy as np

# ROS Imports
import rclpy

from rclpy.node                 import Node
from rclpy.time                 import Time, Duration
from rclpy.executors            import MultiThreadedExecutor
from rclpy.qos                  import QoSProfile, DurabilityPolicy

from tf2_ros                    import TransformException
from tf2_ros.buffer             import Buffer
from tf2_ros.transform_listener import TransformListener

from geometry_msgs.msg          import Point, Quaternion, Pose
from geometry_msgs.msg          import Transform, TransformStamped
from nav_msgs.msg               import OccupancyGrid
from sensor_msgs.msg            import LaserScan

import curses
import numpy as np
import sys

# ROS Imports
import rclpy

from rclpy.node                 import Node
from rclpy.time                 import Time
from rclpy.callback_groups      import MutuallyExclusiveCallbackGroup

from geometry_msgs.msg          import Twist

import turtlebot.est as est
from turtlebot.constants import RESOLUTION, ORIGIN_X, ORIGIN_Y, WIDTH, HEIGHT, LOG_THRESHOLD, bresenham

#
#   Global Definitions
#

DSTEP = 0.3

LFREE     = -0.0157 * 3       # FIXME.  Set the log odds ratio of detecting freespace
LOCCUPIED = 0.0319 * 3        # FIXME.  Set the log odds ratio of detecting occupancy

VNOM = 0.5 # 0.25
WNOM = 0.50

GOAL_NODE = est.Node(5, 5)
START_NODE = est.Node(-1, 1)

KT = np.pi/2
KV = 4

WALL_LOGSODD = LOG_THRESHOLD
#
#   Custom Node Class
#
class CustomNode(Node):
    # Initialization.
    def __init__(self, name):

        # Initialize the node, naming it as specified
        super().__init__(name)

        # Save the parameters.
        self.vnom = VNOM
        self.wnom = WNOM

        # Initialize the (repeating) message data.
        self.msg = Twist()
        self.msg.linear.x = 0.0
        self.msg.linear.y = 0.0
        self.msg.linear.z = 0.0
        self.msg.angular.x = 0.0
        self.msg.angular.y = 0.0
        self.msg.angular.z = 0.0

        # Create a fixed rate to control the speed of sending commands.
        rate    = 10.0
        self.dt = 1/rate
        self.rate = self.create_rate(rate)

        # Create the log-odds-ratio grid.
        self.logoddsratio = np.zeros((HEIGHT, WIDTH))

        # self.get_logger().info("Instation before EST\n")
        
        self.path = est.est(START_NODE, GOAL_NODE, self.logoddsratio, logger=self.get_logger())
        self.get_logger().info("Path: " + str(self.path))
        self.get_logger().info("Length of path: " + str(len(self.path)))
        self.pathidx = 0

        # self.get_logger().info("Instation after EST\n")

        # Create a publisher to send the map data.  Note we use a
        # quality of service with durability TRANSIENT_LOCAL, so new
        # subscribers will get the last sent message.  RVIZ and others
        # expect this for map messages.
        quality = QoSProfile(durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             depth=1)
        self.pub = self.create_publisher(OccupancyGrid, '/map', quality)

        # Create a publisher to send twist commands.
        self.pub_vel = self.create_publisher(Twist, '/cmd_vel', 10)

        # Instantiate a TF listener. This implicitly fills a local
        # buffer, so we can quickly retrieve the transform information
        # we need.  The buffer is filled via incoming TF message
        # callbacks, so make sure this runs in a seperate thread.
        self.tfBuffer = Buffer()
        TransformListener(self.tfBuffer, self, spin_thread=True)

        # Create a subscriber to the laser scan.
        group = MutuallyExclusiveCallbackGroup()
        self.create_subscription(LaserScan, '/scan', self.laserCB, 1, callback_group=group)

        # Create a timer for sending the map data.
        self.timer = self.create_timer(1.0, self.sendMap)

    # Shutdown
    def shutdown(self):
        # Destroy the timer and node.
        self.destroy_timer(self.timer)
        self.destroy_node()


    ##################################################################
    # Send the map.  Called from the timer at 1Hz.
    def sendMap(self):
        # Convert the log odds ratio into a probability (0...1).
        # Remember: self.logsoddsratio is a 3460x240 NumPy array,
        # where the values range from -infinity to +infinity.  The
        # probability should also be a 360x240 NumPy array, but with
        # values ranging from 0 to 1, being the probability of a wall.
        # FIXME: Convert the log-odds-ratio into a probability.
        # probability = np.zeros(np.shape(self.logoddsratio))
        # for row in range(self.logoddsratio.shape[0]):
        #     for col in range(self.logoddsratio.shape[1]):
        #         probability[row][col] = np.exp(self.logoddsratio[row][col]) / \
        #                                 (1 + np.exp(self.logoddsratio[row][col]))
        
        probability = np.exp(self.logoddsratio) / (1 + np.exp(self.logoddsratio))

        # Perpare the message and send.  Note this converts the
        # probability into percent, sending integers from 0 to 100.
        now  = self.get_clock().now()
        data = (100 * probability).astype(int).flatten().tolist()

        self.map = OccupancyGrid()
        self.map.header.frame_id        = 'map'
        self.map.header.stamp           = now.to_msg()
        self.map.info.map_load_time     = now.to_msg()
        self.map.info.resolution        = RESOLUTION
        self.map.info.width             = WIDTH
        self.map.info.height            = HEIGHT
        self.map.info.origin.position.x = ORIGIN_X
        self.map.info.origin.position.y = ORIGIN_Y
        self.map.data                   = data

        self.pub.publish(self.map)


    ##################################################################
    # Utilities:
    # Set the log odds ratio value
    def set(self, u, v, value):
        # Update only if legal.
        if (u>=0) and (u<WIDTH) and (v>=0) and (v<HEIGHT):
            self.logoddsratio[v,u] = value
        else:
            self.get_logger().warn("Out of bounds (%d, %d)" % (u,v))

    # Adjust the log odds ratio value
    def adjust(self, u, v, delta):
        # Update only if legal.
        if (u>=0) and (u<WIDTH) and (v>=0) and (v<HEIGHT):
            self.logoddsratio[v,u] += delta
        else:
            self.get_logger().warn("Out of bounds (%d, %d)" % (u,v))
    
    
    # Get the current path from the robot
    def getPath(self):
        return self.path
    
    # Get teh current loggsodd ratio map from the robots
    def getLogOddsRatio(self):
        return self.logoddsratio


    ##################################################################
    # Laserscan CB.  Process the scans.
    def laserCB(self, msg):
        # Grab the transformation between map and laser's scan frames.
        # This checks the time the scan occured and grabs transform
        # from the same time, keeping things consistent.
        tscan = Time().from_msg(msg.header.stamp)
        # self.get_logger().info("TScan: " + str(tscan))
        # self.get_logger().info("Frame ID: " + str(msg.header.frame_id))
        try:
            tfmsg = self.tfBuffer.lookup_transform(
                'map', msg.header.frame_id, tscan,
                timeout = Duration(seconds=2.0))
        except TransformException as ex:
            self.get_logger().warn("Unable to get transform: %s" % (ex,))
            return

        # Extract the laser scanner's position and orientation.
        xc     = tfmsg.transform.translation.x
        yc     = tfmsg.transform.translation.y
        thetac = 2 * np.arctan2(tfmsg.transform.rotation.z,
                                tfmsg.transform.rotation.w)

        # Grab the rays: each ray's range and angle relative to the
        # turtlebot's position and orientation.
        rmin     = msg.range_min        # Sensor minimum range to be valid
        rmax     = msg.range_max        # Sensor maximum range to be valid
        ranges   = msg.ranges           # List of ranges for each angle

        thetamin = msg.angle_min        # Min angle (0.0)
        thetamax = msg.angle_max        # Max angle (2pi)
        thetainc = msg.angle_increment  # Delta between angles (2pi/360)
        thetas   = np.arange(thetamin, thetamax, thetainc)

        #############################################################
        # FIXME: PROCESS THE LASER SCAN TO UPDATE THE LOG ODDS RATIO!
        #############################################################

        # # Transfor the laser scanner's position into np.array coordinates and
        # # set the logodds ratio to one.
        u = int((xc - ORIGIN_X) / RESOLUTION)
        v = int((yc - ORIGIN_Y) / RESOLUTION)
        # self.set(u, v, 1.0)
        # # self.logoddsratio[v,u] = 1.0

        # # Mark each position the laser detects by settign the logs odd ratio
        # # of that location to one.
        # for i in range(len(ranges)):
        #     if (ranges[i] > rmin) and (ranges[i] < rmax):
        #         x = xc + ranges[i] * np.cos(thetac + thetas[i])
        #         y = yc + ranges[i] * np.sin(thetac + thetas[i])
        #         u = int((x - ORIGIN_X) / RESOLUTION)
        #         v = int((y - ORIGIN_Y) / RESOLUTION)
        #         self.set(u, v, 1.0)
        #         # self.logoddsratio[v,u] = 1.0   

        for i in range(len(ranges)):
            if (ranges[i] > rmin) and (ranges[i] < rmax):
                # # if we are moving forward and facing a node
                # if 0 - thetas[i] <= thetainc:
                #     u = int((xc - ORIGIN_X) / RESOLUTION)
                #     v = int((yc - ORIGIN_Y) / RESOLUTION) 
                #     if self.logoddsratio[u][v] >= WALL_LOGSODD:
                #         self.path = est.path(est.Node(xc, yc), GOAL_NODE)
                #         self.pathidx == 0
                
                
                # Calculate the minimum sensor reading distance in (x,y) coords
                x_min = xc  + rmin * np.cos(thetac + thetas[i])
                y_min = yc  + rmin * np.sin(thetac + thetas[i])

                # Convert minimum sensor reading distance to (u,v) coords
                u_min = int((x_min - ORIGIN_X) / RESOLUTION)
                v_min = int((y_min - ORIGIN_Y) / RESOLUTION)

                # Caclaulte the position of the obstacle in (x,y) coords
                x_object = xc + ranges[i] * np.cos(thetac + thetas[i])
                y_object = yc + ranges[i] * np.sin(thetac + thetas[i])

                # Conver the position of the obstacle to (u,v) coords
                u_object = int((x_object - ORIGIN_X) / RESOLUTION)
                v_object = int((y_object - ORIGIN_Y) / RESOLUTION)

                for (u,v) in bresenham((u_min, v_min), (u_object,v_object)):
                    self.adjust(u, v, LFREE)

                self.adjust(u_object, v_object, LOCCUPIED)

        # self.msg.linear.x  = 1.0
        # self.pub_vel.publish(self.msg)

        # Checks if we have reached the goal node
        if np.sqrt((xc - GOAL_NODE.coordinates()[0])**2 + (yc - GOAL_NODE.coordinates()[1])**2) <= DSTEP:
            self.msg.linear.x  = 0.0
            self.msg.angular.z = 0.0
            self.pub_vel.publish(self.msg)
        else:
            euc_dist = np.sqrt((self.path[self.pathidx].x - xc)**2 + (self.path[self.pathidx].y - yc)**2)
            # Checks if we have reached the next node 
            if euc_dist <= DSTEP:
                self.pathidx += 1
            
            
            
            # Calculate the heading to reach the next node
            (x2, y2) = self.path[self.pathidx].coordinates()
            theta_next = np.arctan2(y2 - yc, x2 -xc)
            
            # Chekcs if robot is at the desired heading
            if abs(thetac - theta_next) <= thetainc:
                self.get_logger().info("Going straight!")
                self.msg.angular.z = 0.0
                
                u_curr = int((xc - ORIGIN_X) / RESOLUTION)
                v_curr = int((yc - ORIGIN_Y) / RESOLUTION)
                u_next = int((self.path[self.pathidx].x - ORIGIN_X) / RESOLUTION)
                v_next = int((self.path[self.pathidx].y - ORIGIN_Y) / RESOLUTION)

                # self.get_logger().info("Current position: " + str(xc) + " " + str(yc))
                # self.get_logger().info("Next position: " + str(self.path[self.pathidx].x) + " " + str(self.path[self.pathidx].y))


                isNewPath = False
                for (u,v) in bresenham((u_curr, v_curr), (u_next, v_next)):
                    # self.get_logger().info("Bresenham u " + str(u) + " v " + str(v))
                    # x = (u * RESOLUTION) + ORIGIN_X
                    # y = RESOLUTION * v + ORIGIN_Y
                    # self.get_logger().info("X: "  + str(x) + " Y: " + str(y) + " LogOdds: " + str(self.logoddsratio[v,u]) + " >= " + str(WALL_LOGSODD))
                    if self.logoddsratio[v, u] >= WALL_LOGSODD or self.logoddsratio[v_next, u_next] >= WALL_LOGSODD or self.logoddsratio[v_curr, u_curr] >= WALL_LOGSODD:
                        self.get_logger().info("Wall ahead!")
                        isNewPath = True
                        self.path = est.est(est.Node(xc, yc), GOAL_NODE, self.logoddsratio, logger=self.get_logger())
                        self.get_logger().info("New Path: " + str(self.path))
                        self.get_logger().info("Length of path: " + str(len(self.path)))
                        self.pathidx = 0
                        self.msg.linear.x = 0.0

                if not isNewPath:                
                    self.msg.linear.x  = min(KV * euc_dist, self.vnom) * np.cos(thetac - theta_next) # whats d?  
            else:
                self.get_logger().info("Turning!")
                self.msg.linear.x  = 0.0
                self.msg.angular.z = min(max(-KT * (thetac - theta_next), -WNOM), WNOM)


            # self.get_logger().info(str(self.msg.linear.x))
            # self.get_logger().info(str(self.msg.angular.z))

            # # Update the message and publish.
            self.pub_vel.publish(self.msg)



# NODE = CustomNode()

#
#   Main Code
#
def main(args=None):

    # Initialize ROS.
    rclpy.init(args=args)

    # Instantiate the node.
    NODE = CustomNode('buildmap')
    # NODE.get_logger().warn("Linear Velocity: " + str(NODE.msg.linear.x))
    # NODE.get_logger().info("Angular Velocity: " + str(NODE.msg.angular.z))

    # Spin the node until interrupted.  To support the TF Listener
    # using another thread, use a multithreaded executor.
    executor = MultiThreadedExecutor()
    executor.add_node(NODE)
    try:
        executor.spin()
    except BaseException as ex:
        print("Ending due to exception: %s" % repr(ex))

    # Shutdown the node and ROS.
    NODE.shutdown()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
