import numpy as np
XMAX = 18
YMAX = 12
RESOLUTION = 0.5
WIDTH  = int(XMAX/RESOLUTION)
HEIGHT = int(YMAX/RESOLUTION)


# Variables from buildmap code
ORIGIN_X   = -9.00              # Origin = location of lower-left corner
ORIGIN_Y   = -6.00

PROBABILITY_THRESHOLD = 0.7
LOG_THRESHOLD = np.log(PROBABILITY_THRESHOLD/(1-PROBABILITY_THRESHOLD))


# Return a list of all intermediate (integer) pixel coordinates
# from (start) to (end) coordinates (which could be non-integer).
# In classic Python fashion, this excludes the end coordinates.
def bresenham(start, end):
        # Extract the coordinates
        (xs, ys) = start
        (xe, ye) = end
        
        if start == end:
            return [(int(xs), int(ys))]

        # Move along ray (excluding endpoint).
        if (np.abs(xe-xs) >= np.abs(ye-ys)):
            if np.sign(xe-xs) == 0:
                return [start]
            return[(u, int(ys + (ye-ys)/(xe-xs) * (u+0.5-xs)))
                   for u in range(int(xs), int(xe), int(np.sign(xe-xs)))]
        else:
            if np.sign(ye-ys) == 0:
                return [start]
            return[(int(xs + (xe-xs)/(ye-ys) * (v+0.5-ys)), v)
                   for v in range(int(ys), int(ye), int(np.sign(ye-ys)))]
