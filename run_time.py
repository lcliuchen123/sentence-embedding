
import time
from functools import wraps


def cost_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print("the cost time of %s is %f s" % (func.__name__, end-start))
        return result
    return wrapper
