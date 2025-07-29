import string
import random
from datetime import datetime



def timestamp_str() -> str:
    """ Get current time stamp string
    """
    now = datetime.now()
    return now.strftime("%Y-%m-%d_%H-%M-%S")

