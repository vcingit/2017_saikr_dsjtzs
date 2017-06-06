import numpy as np
a=[1,0,1,1,0,0]
b=[1,1,1,0,0,0]

a=np.array(a)
b=np.array(b)
print a[a==0]
p=float(np.sum((a==0)&(b==0)))/float(np.sum(b==0))
r=float(np.sum((a==0)&(b==0)))/float(np.sum(a==0))
print 5*p*r/(2*p+3*r)*100