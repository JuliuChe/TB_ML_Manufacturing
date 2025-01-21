'''
solving for x
e**x=b
'''
import numpy as np
import math
b=5.2
x=math.fsum([math.floor((np.log(b))*1000.0)/1000.0, 0.001, 0.00000000000000001])
print("x value is : ",x ) #This is natural log
print("b value is : ", math.floor((math.e**np.log(b))*1000)/1000+1/1000)

print(-np.log(0.7))

#Simulated output from the last neurons layer
softmax_output = [0.5,(1-0.5)/2.0,(1-0.5)/2.0]
target_output = [1,0,0]
loss=0

for tg, out in zip(target_output, softmax_output):
    print(tg, out)
    loss -= (tg*math.log(out))
print("loss is : ", loss)