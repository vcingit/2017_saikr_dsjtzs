# -*- coding: utf-8 -*-
import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.pyplot import plot,savefig

#根据x和y坐标作图
def makeLineChart(x,y,target_x,target_y,label,pic_id,model):
	if len(x)==0 or len(y) ==0:
		pass
	plt.title('label:%s'%(label))
	plt.xlabel('x-axis')
	plt.ylabel('y-axis')
	plt.plot(x, y,'r',marker='o')
	#if target_x !=None and target_y !=None:
		#plt.plot(target_x,target_y,'b',label='point')
		#plt.plot(target_x,target_y,'b',marker='*')

	plt.legend(bbox_to_anchor=[0.3, 1])
	plt.grid()
	if model==0:
		plt.show()
	else:
		if label=='0':
			plt.savefig('../pic/0/%s'%(pic_id))
		else:
			plt.savefig('../pic/1/%s'%(pic_id))
		plt.close('all')

#读取文件
def readFile(filepath):
	print 'read%s'%(filepath)
	data = pd.read_table(filepath,header=None,encoding='utf-8',delim_whitespace=True,index_col=0)
	return data

#获取移动轨迹坐标和时间
def getTrail(row):
	trail_x,trail_y,trail_t=[],[],[]
	loc=str(row[1]).split(';')
	for x_y_t in loc:
		x_y=x_y_t.split(',')
		if len(x_y) >=3:
			trail_x.append(x_y[0])
			trail_y.append(x_y[1])
			trail_t.append(x_y[2])
	return trail_x,trail_y,trail_t

#将坐标在图中表示
def dataToChart(filepath):
	data=readFile(filepath)
	#row 1-移动轨迹坐标 2-目的坐标 3-标签
	for ix, row in data.iterrows():
		trail_x,trail_y,trail_t=getTrail(row)
		target=str(row[2]).split(',')
		label=str(row[3]) if len(row)==3 else -1
		makeLineChart(trail_x,trail_y,target[0],target[1],label,ix,0)

def writeResult(result,filepath):
	file=open(filepath,'w')  
	file.write('\n'.join(result));  
	file.close()

# def firstResult():
# 	data=readFile('../data/dsjtzs_txfz_test1.txt')
# 	res=[]
# 	#print 'end_x','end_y','end_t','target_x','target_y','x_D_value_rate','y_D_value_rate'
# 	for ix, row in data.iterrows():
# 		trail_x,trail_y,trail_t=getTrail(row)
# 		target=str(row[2]).split(',')
# 		label=str(row[3]) if len(row)==3 else -1
		
# 		end_x,end_y,end_t=float(trail_x[-1]),float(trail_y[-1]),float(trail_t[-1])
# 		target_x,target_y=float(target[0]),float(target[1])
		
# 		x_D_value_rate=abs(end_x-target_x)/end_x*100.0
# 		y_D_value_rate=abs(end_y-target_y)/end_y*100.0
		
		
# 		#print end_x,end_y,end_t,target_x,target_y,x_D_value_rate,y_D_value_rate
# 		if end_y<0 or end_x<0 or x_D_value_rate<2 or end_t>20000 or end_t<1000:
# 			res.append(0)
# 		else:
# 			res.append(1)
# 	result=[]
# 	for i in range(len(res)):
# 		if res[i]==0:
# 			result.append(str(i+1))
# 	writeResult(result,'../result/submit.txt')
def getFeature(trail_x,trail_y,trail_t,target,label):
	#x最大值 x最小值 y最大值 y最小值
	#起点x位置 起点y位置 终点x位置 终点y位置
	#x平均值 y平均值 x方差 y方差
	#总步数 总时间 x轴是否有退回 是否有异常时间
	#每一步时间间隔平均值 每一步时间间隔的方差
	#每一步前进距离平均值 每一步前进距离的方差 前进速度平均值 前进速度方差
	#图像围成的钝角数 图像围成的直角数 图像围成的锐角数 角度平均值 角度方差
	max_x,min_x,max_y,min_y=0,99999,0,99999
	sta_x,sta_y,end_x,end_y=0,0,0,0
	average_x,average_y,variance_x,variance_y=0,0,0,0
	all_step_num,all_time,if_x_back,if_t_back=0,0,0,0
	average_t,variance_t,average_sp,variance_sp=0,0,0,0
	each_step_distance_average,each_step_distance_variance=0,0
	obtuse_num,right_num,acute_num,average_degree,variance_degree=0,0,0,0,0


	#字符串列表转换为float ndarray
	trail_x=[float(x) for x in trail_x]
	trail_y=[float(y) for y in trail_y]
	trail_t=[float(t) for t in trail_t]
	ntrail_x=np.array(trail_x)
	ntrail_y=np.array(trail_y)
	ntrail_t=np.array(trail_t)

	#总步数 总时间
	all_step_num,all_time=ntrail_x.shape[0],ntrail_t[-1]-ntrail_t[0]

	#最大最小值
	max_x,min_x,max_y,min_y=np.max(ntrail_x),np.min(ntrail_x),np.max(ntrail_y),np.min(ntrail_y)
	#起点位置 终点位置
	sta_x,sta_y,end_x,end_y=ntrail_x[0],ntrail_y[0],ntrail_x[-1],ntrail_y[-1]

	#最大值 平均值 方差
	sum_x,sum_y,sum_t=ntrail_x.sum(),ntrail_y.sum(),ntrail_t.sum()
	if all_step_num>1:
		average_x,average_y,average_t=sum_x/all_step_num,sum_y/all_step_num,sum_t/all_step_num
	#variance_x,variance_y,variance_t=np.std(ntrail_x,ddof=1),np.std(ntrail_y,ddof=1),np.std(ntrail_t,ddof=1)
	#距离 速度 回退情况 角度
	dif_x,dif_y,dif_t=[],[],[]
	for i in range(all_step_num):
		if i!=0 and i!=all_step_num-1:
			tmp_x=trail_x[i]-trail_x[i-1]
			tmp_y=trail_y[i]-trail_y[i-1]
			tmp_t=trail_t[i]-trail_t[i-1]
			dif_x.append(tmp_x)
			dif_y.append(tmp_y)
			dif_t.append(tmp_t)
			if tmp_x<0:
				if_x_back=1

	#删除时间间隔为0或者负的点
	valid_index=[i for i in range(len(dif_t)) if dif_t[i] > 0]
	dif_x=[dif_x[i] for i in valid_index]
	dif_y=[dif_y[i] for i in valid_index]
	dif_t=[dif_t[i] for i in valid_index]

	dif_x,dif_y,dif_t=np.array(dif_x),np.array(dif_y),np.array(dif_t)
	if len(dif_t)>1:
		#print len(dif_t)
		if False:
			pass
		else:
			dis_xy=np.sqrt(dif_x**2+dif_y**2)
			sp_xyt=dis_xy/(dif_t)
			each_step_distance_average=dis_xy.sum()/dis_xy.shape[0]
			each_step_distance_variance=np.std(dis_xy,ddof=1)
			average_sp=sp_xyt.sum()/sp_xyt.shape[0]
			variance_sp=np.std(sp_xyt,ddof=1)
	
	return max_x,min_x,max_y,min_y,\
		sta_x,sta_y,end_x,end_y,\
		average_x,average_y,average_t,\
		variance_x,variance_y,variance_t,\
		all_step_num,all_time,if_x_back,if_t_back,\
		average_sp,variance_sp,\
		each_step_distance_average,each_step_distance_variance
		#obtuse_num,right_num,acute_num,average_degree,variance_degree

def get_score(predict,correct):
	G=float(np.sum((correct=='0')&(predict=='0')))
	P1=float(np.sum(predict=='0'))
	P2=float(np.sum(correct=='0'))
	if P1==0:
		P,R=0,0
		return 0,0,0
	else:
		P=G/P1
		R=G/P2
		return P,R,5*P*R/(2*P+3*R)*100
def loadDataSet(data):
	fea=[]
	labels=[]
	for ix, row in data.iterrows():
		trail_x,trail_y,trail_t=getTrail(row)
		target=str(row[2]).split(',')
		label=str(row[3]) if len(row)==3 else -1
		labels.append(label)
		#print ix,
		fea.append(getFeature(trail_x,trail_y,trail_t,target,label))
	nfea=np.array(fea)
	nlabels=np.array(labels)
	return nfea,nlabels

def cross_validation(x,y):
	from sklearn.model_selection import train_test_split
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
	from sklearn.svm import SVC
	clf = SVC(C=1.28,gamma=0.000001)
	clf.fit(x_train, y_train)
	res=clf.predict(x_test)
	print get_score(res,y_test)

def get_result(train,label,test):
	from sklearn.svm import SVC
	clf = SVC(C=1.28,gamma=0.000001)
	clf.fit(train, label)
	res=clf.predict(test)
	print res==0
	result=[]
	for i in range(len(res)):
		if res[i]==0:
			result.append(str(i+1))
	return result

if __name__ == "__main__":
	#dataToChart('../data/dsjtzs_txfz_training.txt')
	train_data=readFile('../data/dsjtzs_txfz_training.txt')
	test_data=readFile('../data/dsjtzs_txfz_test1.txt')
	nfea,nlabels=loadDataSet(train_data)
	test,tlabels=loadDataSet(test_data)
	#cross_validation(nfea,nlabels)
	result=get_result(nfea,nlabels,test)
	writeResult(result,'../result/submit.txt')