import random

def rand_scale(s):
    """
    ----------
    Author: Damon Gwinn (gwinndr)
    ----------
    - Gets a randomized scaling based on given s term
    - Returned scaling can be less than or greater than 1
    ----------
    """

    scale = random.uniform(1.0, s)

    flip_scale = bool(random.randint(0,1))
    if(flip_scale):
        scale = 1.0 / scale

    return scale
