import numpy as np
from graphcut import Segmentation

#image
img = np.array([[12, 170, 160,10],
       [40, 30, 150,50],
       [50, 137, 130,60],
       [10,100,120,20]])
#label seed 
bkg = [(0,0),(1,0)]
obj = [(2,1),(2,2)]

#image segmentation via graph-cut
S = Segmentation(img,obj,bkg)
mask = S.run() 
