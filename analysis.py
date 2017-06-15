# -*- coding: utf-8 -*-
import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.pyplot import plot,savefig
import data_reduction as dr

def loadDataSet(data):
	print 'LoadDataSet...'
	fea=[]
	labels=[]
	for ix, row in data.iterrows():
		trail_x,trail_y,trail_t=dr.getTrail(row)
		target=str(row[2]).split(',')
		label=str(row[3]) if len(row)==3 else -1
		labels.append(label)
		#print ix,
		fea.append(getFeature(trail_x,trail_y,trail_t,target,label))
	# for i in range(len(fea)):
	# 	for j in fea[i]:
	# 		print str(j)+'\t',
	# 	print labels[i]
	print '------------------------------'
	return fea,labels

#使用规则预测分类
def regular(filepath):
	data=dr.readFile(filepath)
	res=[]
	#print 'end_x','end_y','end_t','target_x','target_y','x_D_value_rate','y_D_value_rate'
	for ix, row in data.iterrows():
		trail_x,trail_y,trail_t=dr.getTrail(row)
		target=str(row[2]).split(',')
		label=str(row[3]) if len(row)==3 else -1
		
		end_x,end_y,end_t=float(trail_x[-1]),float(trail_y[-1]),float(trail_t[-1])
		target_x,target_y=float(target[0]),float(target[1])
		
		x_D_value_rate=abs(end_x-target_x)/end_x*100.0
		y_D_value_rate=abs(end_y-target_y)/end_y*100.0
		
		
		#print end_x,end_y,end_t,target_x,target_y,x_D_value_rate,y_D_value_rate
		if end_y<0 or end_x<0 or x_D_value_rate<2 or end_t>20000 or end_t<1000:
			res.append('0')
		else:
			res.append('1')
	return res

#获取线下预测得分
def get_score(predict,correct):
	G=float(np.sum((correct=='0')&(predict=='0')))
	P1=float(np.sum(predict=='0'))
	P2=float(np.sum(correct=='0'))
	print 'predict/correct/hit:',P1,P2,G
	if P1==0:
		P,R=0,0
		return 0,0,0
	else:
		P=G/P1
		R=G/P2
		return P,R,5*P*R/(2*P+3*R)*100

#复制list
def copy_list(list):
	a=[]
	for i in list:
		a.append(i)
	return a

#交叉验证
def cross_validation(x,y):
	from sklearn.model_selection import train_test_split
	#print 'cross_validation...'
	#sklearn split
	#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
	#print 'machine num in train/test:',len([i for i in y_train if i=='0']),len([i for i in y_test if i=='0'])
	#clf,x_train,x_test=normalization(x_train,x_test)
	#clf.fit(x_train, y_train)
	#res=clf.predict(x_test)
	#print 'score is:',get_score(res,y_test)
	
	print 'five_cross_validation...'
	#five split
	train,label=dr.five_split_data(x,y)
	scores=[]
	for i in range(len(train)):
		x_train,y_train=copy_list(train[i]),copy_list(label[i])
		x_test,y_test=copy_list(train[4-i]),copy_list(label[4-i])
		for j in range(len(train)):
			if j!=4-i and j!=i:
				x_train+=copy_list(train[j])
				y_train+=copy_list(label[j])
		#print len(x_train),len(x_test)
		x_train,x_test,y_train,y_test=np.array(x_train),np.array(x_test),np.array(y_train),np.array(y_test)
		clf,x_train,x_test=normalization(x_train,x_test)
		clf.fit(x_train, y_train)
		res=clf.predict(x_test)
		#print res,y_test
		scores.append(get_score(res,y_test))
	#print 'reality:',y_test,'\npredict:',res
	print 'average score is:\n',\
		'(',' + '.join([str(round(i[2],2)) for i in scores]),\
		') /',len(scores),'=',\
		round(sum([i[2] for i in scores])/len(scores),2)
	print '------------------------------'

#使用分类器分类预测
def get_result(train,label,test):
	print 'training...'

	clf,train,test=normalization(train,test)

	clf.fit(train, label)
	res=clf.predict(test)
	#print 'predict res:',res
	result=convert_res_to_result(res)
	#print 'result:',result
	print '------------------------------'
	print 'predict num:',len(result),'result:',np.array(result)
	print '------------------------------'
	return result

#获取特征
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
	obtuse_rate,acute_rate=0.0,0.0

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
		variance_x,variance_y,variance_t=np.std(ntrail_x,ddof=1),np.std(ntrail_y,ddof=1),np.std(ntrail_t,ddof=1)
	#距离 速度 回退情况 角度
	#两点之间的x差、y差、t差
	dif_x,dif_y,dif_t=[],[],[]
	#三点之间的角度差
	degree=[]
	for i in range(all_step_num):
		if i!=0 and i!=all_step_num-1:
			tmp_x=trail_x[i]-trail_x[i-1]
			tmp_y=trail_y[i]-trail_y[i-1]
			tmp_t=trail_t[i]-trail_t[i-1]
			if tmp_t < 0 :
				continue
			dif_x.append(abs(tmp_x))
			dif_y.append(abs(tmp_y))
			dif_t.append(abs(tmp_t))
			if tmp_x<0:
				if_x_back=1

		if all_step_num>=5 and i<all_step_num-2:
			x0,y0=trail_x[i],trail_y[i]
			x1,y1=trail_x[i+1],trail_y[i+1]
			x2,y2=trail_x[i+2],trail_y[i+2]
			v1=[x1-x0,y1-y0]
			v2=[x2-x1,y2-y1]
			#排除异常情况
			if v1==[0,0] or v2==[0,0]:
				continue

			nv1=np.array(v1)
			nv2=np.array(v2)
			Lx=np.sqrt(nv1.dot(nv1))
			Ly=np.sqrt(nv2.dot(nv2))
			#print v1,v2,nv1.dot(nv2)/(Lx*Ly)
			degree.append(nv1.dot(nv2)/(Lx*Ly))

	#删除时间间隔为0或者负的点
	valid_index=[i for i in range(len(dif_t)) if dif_t[i] > 0]
	dif_x=[dif_x[i] for i in valid_index]
	dif_y=[dif_y[i] for i in valid_index]
	dif_t=[dif_t[i] for i in valid_index]

	if len(degree)>0:
		obtuse_num=len([i for i in degree if i<0])
		right_num=len([i for i in degree if i==0])
		acute_num=len([i for i in degree if i>0])
		average_degree=np.sum(np.array(degree))/len(degree)
		variance_degree=np.std(np.array(degree),ddof=1)


	acute_rate=float(acute_num)/float(right_num+acute_num+obtuse_num) if acute_num!=0 else 0
	obtuse_rate=float(obtuse_num)/float(right_num+acute_num+obtuse_num) if obtuse_num!=0 else 0
	right_rate=float(right_num)/float(right_num+acute_num+obtuse_num) if right_num!=0 else 0

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
	
	range_x=max_x-min_x
	range_y=max_y-min_y
	move_x=end_x-sta_x
	move_y=end_y-sta_y
	
	return each_step_distance_average,each_step_distance_variance,\
		average_sp,variance_sp,\
		all_time,all_step_num,if_x_back,if_t_back,\
		average_x,average_y,average_t,\
		variance_x,variance_y,variance_t,\
		obtuse_num,right_num,acute_num,\
		obtuse_rate,right_rate,acute_rate,\
		#average_degree,variance_degree

#特征标准化，切换分类器
def normalization(x_train,x_test):
	from sklearn import preprocessing

	#standard标准化
	# scaler = preprocessing.StandardScaler().fit(x_train)
	# scaler.transform(x_train)
	# scaler.transform(x_test)

	#最大最小归一化
	min_max_scaler = preprocessing.MinMaxScaler()
	x_train = min_max_scaler.fit_transform(x_train)
	x_test = min_max_scaler.fit_transform(x_test)
	from sklearn.svm import SVC
	clf = SVC(kernel='linear')

	# from sklearn.ensemble import RandomForestRegressor 
	# clf=RandomForestRegressor()

	# from sklearn import tree
	# clf = tree.DecisionTreeClassifier()

	
	return clf,x_train,x_test

#预测结果转换为提交结果
def convert_res_to_result(res):
	result=[]
	for i in range(len(res)):
		if res[i]=='0':
			result.append(str(i+1))
	return result

#选择线下测试或线上预测
def choose_model(fea,labels,model=0):
	if model==0:
		cross_validation(fea,labels)
		# l1=[fea[i] for i in range(len(fea)) if labels[i]=='1']
		# l2=[fea[i] for i in range(len(fea)) if labels[i]=='0']
		# x_1,y_1=[i[0] for i in l1],[i[1] for i in l1]
		# x_0,y_0=[i[0] for i in l2],[i[1] for i in l2]
		# makeScatterChart(x_1,y_1,x_0,y_0)
	elif model==1:
		nfea,nlabels=np.array(fea),np.array(labels)
		test_data=dr.readFile('../data/dsjtzs_txfz_test1.txt')
		test,tlabels=loadDataSet(test_data)
		result=get_result(nfea,nlabels,test)
		dr.writeResult(result,'../result/BDC1282.txt')

#主函数
if __name__ == "__main__":
	#dataToChart('../data/dsjtzs_txfz_test1.txt','test_pic',1)
	train_data=dr.readFile('../data/dsjtzs_txfz_training.txt')
	fea,labels=loadDataSet(train_data[:])
	choose_model(fea,labels,1)
	#dr.writeFea(fea,labels,'../result/fea.csv')
	#print 'score is:',get_score(np.array(regular('../data/dsjtzs_txfz_training.txt')),np.array(labels))