import numpy as np
#-----------------------------------------------------------------------------------------#

def ellipse(b, x, a):
    """
	creating simple ellipse function
    """
    p1 = pow(x, 2)/pow(a, 2)
    p2 = np.sqrt(1000 - p1)
    y = b*p2
    return y
