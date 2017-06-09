import numpy as np
x=[[1.0,2.0,1900.0],[0.0,3.0,9810.0],[1.0,2.0,12010.0],[0.0,2.0,11810.0],\
	[1.0,1.0,10100.0],[0.0,1.0,11000.0],[1.0,2.0,15100.0],[0.0,1.0,11010.0],\
	[1.0,4.0,15146.0],[1.0,4.0,11555.0],[1.0,4.0,15416.0],[1.0,4.0,15155.0],\
	[1.0,2.0,17146.0],[0.0,1.0,19155.0],[1.0,2.0,16145.0],[1.0,3.0,13156.0],\
	[0.0,8.0,15146.0],[1.0,2.0,151055.0],[0.0,1.0,12167.0],[1.0,5.0,11155.0]]
y=[1.0,0,1,0,1,0,1,0,1,1,1,1,1,0,1,1,0,1,0,1]
z=[[1.0,1.0,0.0],[0.0,1.0,0.0],[1.0,1.0,0.0],[1.0,1.0,0.0]]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(np.array(x), np.array(y), test_size = 0.2)
from sklearn.svm import SVC
clf = SVC(kernel='linear')
# from sklearn import tree
# clf = tree.DecisionTreeClassifier()
# clf.fit(x_train, y_train)
# res=clf.predict(x_test)
# print res,y_test

# from sklearn import preprocessing
# scaler = preprocessing.StandardScaler().fit(x_train)
# scaler.transform(x_train)
# scaler.transform(x_test)
#print x_train,'\n',x_test
clf.fit(x_train, y_train)
res=clf.predict(x_test)
print res,y_test
