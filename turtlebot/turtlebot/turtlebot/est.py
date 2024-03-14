#!/usr/bin/env python3
#
#   esttriangles.py
#
#   Use EST to find a path around triangular obstacles.
#
import matplotlib.pyplot as plt
import numpy as np
import random
import time

from math                import pi, sin, cos, atan2, sqrt, ceil
from scipy.spatial       import KDTree
from shapely.geometry    import Point, LineString, Polygon, MultiPolygon
from shapely.prepared    import prep
from turtlebot.constants import RESOLUTION, ORIGIN_X, ORIGIN_Y, WIDTH, HEIGHT, LOG_THRESHOLD, XMAX, YMAX, bresenham

######################################################################
#
#   Parameters
#
#   Define the step size.  Also set the maximum number of nodes.
#

DSTEP = 0.6
# PROBABILITY_THRESHOLD = 0.7
# LOG_THRESHOLD = np.log(PROBABILITY_THRESHOLD/(1-PROBABILITY_THRESHOLD))

# Maximum number of nodes.
NMAX = 1500


######################################################################
#
#   World Definitions
#
#   List of obstacles/objects as well as the start/goal.
#
(xmin, xmax) = (0, 14)
(ymin, ymax) = (0, 10)

# Collect all the triangle and prepare (for faster checking).
triangles = prep(MultiPolygon([
    Polygon([[ 2, 6], [ 3, 2], [ 4, 6], [ 2, 6]]),
    Polygon([[ 6, 5], [ 7, 7], [ 8, 5], [ 6, 5]]),
    Polygon([[ 6, 9], [ 8, 9], [ 6, 7], [ 6, 9]]),
    Polygon([[10, 3], [11, 6], [12, 3], [10, 3]])]))

# Define the start/goal states (x, y, theta), start to the right.
(xstart, ystart) = (13, 5)
(xgoal,  ygoal)  = ( 1, 5)


######################################################################
#
#   Utilities: Visualization
#
# Visualization Class
class Visualization:
    def __init__(self):
        # Create a publisher to send twist commands.
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)

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
        # Clear the current, or create a new figure.
        plt.clf()

        # Create a new axes, enable the grid, and set axis limits.
        plt.axes()
        plt.grid(True)
        plt.gca().axis('on')
        plt.gca().set_xlim(xmin, xmax)
        plt.gca().set_ylim(ymin, ymax)
        plt.gca().set_aspect('equal')

        # Show the triangles.
        for poly in triangles.context.geoms:
            plt.plot(*poly.exterior.xy, 'k-', linewidth=2)

        # Show.
        self.show()

    def show(self, text = ''):
        # Show the plot.
        plt.pause(0.001)
        # If text is specified, print and wait for confirmation.
        if len(text)>0:
            input(text + ' (hit return to continue)')

    def drawNode(self, node, *args, **kwargs):
        plt.plot(node.x, node.y, *args, **kwargs)

    def drawEdge(self, head, tail, *args, **kwargs):
        plt.plot((head.x, tail.x),
                 (head.y, tail.y), *args, **kwargs)

    def drawPath(self, path, *args, **kwargs):
        for i in range(len(path)-1):
            self.drawEdge(path[i], path[i+1], *args, **kwargs)


######################################################################
#
#   Node Definition
#
class Node:
    def __init__(self, x, y):
        # Define a parent (cleared for now).
        self.parent = None

        # Define/remember the state/coordinates (x,y).
        self.x = x
        self.y = y

    ############
    # Utilities:
    # In case we want to print the node.
    def __repr__(self):
        return ("<Point %5.2f,%5.2f>" % (self.x, self.y))

    # Compute/create an intermediate node.  This can be useful if you
    # need to check the local planner by testing intermediate nodes.
    def intermediate(self, other, alpha):
        return Node(self.x + alpha * (other.x - self.x),
                    self.y + alpha * (other.y - self.y))

    # Return a tuple of coordinates, used to compute Euclidean distance.
    def coordinates(self):
        return (self.x, self.y)

    # Compute the relative distance to another node.
    def distance(self, other):
        return sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

    ################
    # Collision functions:
    # Check whether in free space.
    def inFreespace(self, logOddsRatio):
        if self.x <= -XMAX/2 or self.x >= XMAX/2 or self.y <= -YMAX/2 or self.y >= YMAX/2:
            return False
        
        u = int((self.x - ORIGIN_X) / RESOLUTION)
        v = int((self.y - ORIGIN_Y) / RESOLUTION)
        if logOddsRatio[v, u] <= LOG_THRESHOLD:
            return True
        else:
            return False
            
    # Check the local planner - whether this connects to another node.
    def connectsTo(self, other, logOddsRatio):
        # logoddsratio = logOddsRatio
        # probability = np.exp(logoddsratio) / (1 + np.exp(logoddsratio))

        (x1, y1) = self.coordinates()
        (x2, y2) = other.coordinates()

        u_curr = int((x1 - ORIGIN_X) / RESOLUTION)
        v_curr = int((y1 - ORIGIN_Y) / RESOLUTION)
        u_next = int((x2 - ORIGIN_X) / RESOLUTION)
        v_next = int((y2 - ORIGIN_Y) / RESOLUTION)

        # Iterate betwene each space 
        for (u, v) in bresenham((u_curr, v_curr), (u_next, v_next)):
            # for i in range(-1, 1):
            #     for j in range (-1, 1):
            #         if (v+i) >= 0 and (v+i) < HEIGHT and (u+j) >= 0 and (u+j) < WIDTH:
            if logOddsRatio[v, u] >= LOG_THRESHOLD:
                return False
        return True




######################################################################
#
#   EST Functions
#
def est(startnode, goalnode, logOddsRatio, logger=None):
    # Start the tree with the startnode (set no parent just in case).
    startnode.parent = None
    tree = [startnode]

    if logger:
        logger.info("Starting w/ (%s) and (%s)" % (str(startnode), str(goalnode)))

    # Function to attach a new node to an existing node: attach the
    # parent, add to the tree, and show in the figure.
    def addtotree(oldnode, newnode):
        newnode.parent = oldnode
        tree.append(newnode)
        # visual.drawEdge(oldnode, newnode, color='g', linewidth=1)
        # visual.show()

    # Loop - keep growing the tree.
    while True:
        # Determine the local density by the number of nodes nearby.
        # KDTree uses the coordinates to compute the Euclidean distance.
        # It returns a NumPy array, same length as nodes in the tree.
        X = np.array([node.coordinates() for node in tree])
        kdtree  = KDTree(X)
        numnear = kdtree.query_ball_point(X, r=1.5*DSTEP, return_length=True)

        # Directly determine the distances to the goal node.
        distances = np.array([node.distance(goalnode) for node in tree])

        # Select the node from which to grow, which minimizes some metric.
        scale = 5.0
        cost = []
        for idx in range(0, len(tree)):
            cost.append(numnear[idx] + scale * distances[idx])
        min_elem = min(cost)
        min_idxs = []
        for idx in range(0, len(cost)):
            if cost[idx] == min_elem:
                min_idxs.append(idx)
        min_idx = random.choice(min_idxs)
        grownode = tree[min_idx]

        # Check the incoming heading, potentially to bias the next node.
        if grownode.parent is None:
            heading = 0
        else:
            heading = atan2(grownode.y - grownode.parent.y,
                            grownode.x - grownode.parent.x)

        # Find something nearby: keep looping until the tree grows.
        while True:
            # Pick the next node randomly.
            
            # Problem: 2a
            # angle = random.uniform(-pi, pi)
            # nextnode = Node(grownode.x + DSTEP * cos(angle),
            #     grownode.y + DSTEP * sin(angle))
            
            # Problem: 2b
            angle = random.normalvariate(heading, pi/2)
            nextnode = Node(grownode.x + DSTEP * cos(heading + angle),
                grownode.y + DSTEP * sin(heading + angle))

            # Try to connect.
            if nextnode.inFreespace(logOddsRatio) and grownode.connectsTo(nextnode, logOddsRatio):
                addtotree(grownode, nextnode)
                break

        # Once grown, also check whether to connect to goal.
        if nextnode.distance(goalnode) < DSTEP and nextnode.connectsTo(goalnode, logOddsRatio):
            addtotree(nextnode, goalnode)
            break

        # Check whether we should abort - too many nodes.
        if (len(tree) >= NMAX):
            if logger:
                logger.info("Aborted with the tree having %d nodes" % len(tree))
            return None

    # Build the path.
    path = [goalnode]
    while path[0].parent is not None:
        path.insert(0, path[0].parent)

    i = 0
    while (i < len(path)-2):
        if path[i].connectsTo(path[i+2], logOddsRatio):
            path.pop(i+1)
        else:
            i = i+1

    # Report and return.
    print("Finished  with the tree having %d nodes" % len(tree))
    return path


######################################################################
#
#  Main Code
#
def main():
    # Report the parameters.
    print('Running with step size ', DSTEP, ' and up to ', NMAX, ' nodes.')

    # Create the figure.
    # visual = Visualization()

    # Create the start/goal nodes.
    startnode = Node(xstart, ystart)
    goalnode  = Node(xgoal,  ygoal)

    # Show the start/goal nodes.
    # visual.drawNode(startnode, color='orange', marker='o')
    # visual.drawNode(goalnode,  color='purple', marker='o')
    # visual.show("Showing basic world")


    # Run the EST planner.
    print("Running EST...")
    # path = est(startnode, goalnode, visual)

    # If unable to connect, just note before closing.
    # if not path:
    #     visual.show("UNABLE TO FIND A PATH")
    #     return

    # Show the path.
    # visual.drawPath(path, color='r', linewidth=2)
    # visual.show("Showing the raw path")


    # Post process the path.
    # PostProcess(path)

    # # Show the post-processed path.
    # visual.drawPath(path, color='b', linewidth=2)
    # visual.show("Showing the post-processed path")


if __name__== "__main__":
    main()
