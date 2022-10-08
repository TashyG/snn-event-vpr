from scipy.spatial.distance import pdist
import numpy as np

speed1 = np.array([1, 5, 10,11,45,68,90])
difference = np.subtract([1, 5, 10,11,45,68,90], [1,1,1,1,1,1,1])
top = np.argsort(difference)[-3:] 

speed1 = speed1[top]

print(speed1)