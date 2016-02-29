import math

g = 9810.0 
err = 127.0

def on_trajectory(v, a, x, y):
    vx = v * math.cos(math.radians(a))
    vy = v * math.sin(math.radians(a))
    t = x / vx
    y2 = - g * t * t / 2 + vy * t
    return abs(y-y2) < err


def velocity_from_max_height(a, y_max):
    v = math.sqrt(2 * y_max * g) / math.sin(math.radians(a))
    return v
