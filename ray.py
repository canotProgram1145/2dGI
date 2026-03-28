import numpy as np
import math
class ray:
    O=np.array([0,0])
    P=np.array([0,0])
    def __init__(self,og,d):
        self.o=og
        self.d = d

    def ray(self):
        return self.o,self.d
