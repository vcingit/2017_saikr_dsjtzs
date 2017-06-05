# -*- coding: utf-8 -*-
import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.pyplot import plot,savefig

#根据x和y坐标作图
def makeLineChart(x,y,target_x,target_y,label,pic_id):
	if len(x)==0 or len(y) ==0:
		pass
	plt.title('label:%s'%(label))
	plt.xlabel('x-axis')
	plt.ylabel('y-axis')
	plt.plot(x, y,'r')
	#if target_x !=None and target_y !=None:
		#plt.plot(target_x,target_y,'b',label='point')
		#plt.plot(target_x,target_y,'b',marker='*')

	plt.legend(bbox_to_anchor=[0.3, 1])
	plt.grid()
	#plt.show()
	if label=='0':
		plt.savefig('../pic/0/%s'%(pic_id))
	else:
		plt.savefig('../pic/1/%s'%(pic_id))
	plt.close('all')

#读取文件
def readFile(filepath):
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
		makeLineChart(trail_x,trail_y,target[0],target[1],label,ix)

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

def getFeature(trail_x,trail_y,trail_t,target):
	#字符串列表转换为float ndarray
	trail_x=[float(x) for x in trail_x]
	trail_y=[float(y) for y in trail_y]
	trail_t=[float(t) for t in trail_t]
	ntrail_x=np.array(trail_x)
	ntrail_y=np.array(trail_y)
	ntrail_t=np.array(trail_t)
	#x最大值 x最小值 y最大值 y最小值
	#起点x位置 起点y位置 终点x位置 终点y位置
	#x平均值 y平均值 x方差 y方差
	#总步数 总时间 x轴是否有退回
	#每一步时间间隔平均值 每一步时间间隔的方差
	#每一步前进距离平均值 每一步前进距离的方差 前进速度平均值 前进速度方差
	#图像围成的钝角数 图像围成的直角数 图像围成的锐角数 角度平均值 角度方差
	max_x,min_x,max_y,min_y=0,99999,0,99999
	sta_x,sta_y,end_x,end_y=0,0,0,0
	average_x,average_y,variance_x,variance_y=0,0,0,0
	all_step_num,all_time,if_x_back=0,0,0
	average_t,variance_t,average_sp,variance_sp=0,0,0,0
	each_step_distance_average,each_step_distance_variance=0,0
	obtuse_num,right_num,acute_num,average_degree,variance_degree=0,0,0,0,0

	#起点位置 终点位置
	sta_x,sta_y,end_x,end_y=ntrail_x[0],ntrail_y[0],ntrail_x[-1],ntrail_y[-1]
	#总步数 总时间
	all_step_num,all_time=ntrail_x.shape[0],ntrail_t[-1]-ntrail_t[0]
	#最大值 平均值 方差
	sum_x,sum_y,sum_t=ntrail_x.sum(),ntrail_y.sum(),ntrail_t.sum()
	average_x,average_y=sum_x/all_step_num,sum_y/all_step_num
	variance_x,variance_y=np.std(ntrail_x,ddof=1),np.std(ntrail_y,ddof=1)
	average_t,variance_t=sum_t/all_step_num,np.std(ntrail_t,ddof=1)
	#距离 速度 回退情况
	dif_x,dif_y,dif_t=[],[],[]
	for i in range(all_step_num):
		if i!=0 and i!=all_step_num-1:
			tmp_x=trail_x[i]-trail_x[i-1]
			tmp_y=trail_y[i]-trail_y[i-1]
			tmp_t=trail_t[i]-trail_t[i-1]
			dif_x.append(tmp_x)
			dif_y.append(tmp_y)
			dif_t.append(tmp_t)
			if dif_x<0:
				if_x_back=1
	dif_x,dif_y,dif_t=np.array(dif_x),np.array(dif_y),np.array(dif_t)
	dis_xy=np.sqrt(dif_x**2+dif_y**2)
	sp_xyt=dis_xy/(dif_t+0.1)
	each_step_distance_average=dis_xy.sum()/dis_xy.shape[0]
	each_step_distance_variance=np.std(dis_xy,ddof=1)
	average_sp=sp_xyt.sum()/sp_xyt.shape[0]
	variance_sp=np.std(sp_xyt,ddof=1)
	#print np.argwhere(dif_t <= 0)
	print average_sp,variance_sp


if __name__ == "__main__":
	#dataToChart('../data/dsjtzs_txfz_training.txt')
	data=readFile('../data/dsjtzs_txfz_training_sample.txt')
	res=[]
	for ix, row in data.iterrows():
		trail_x,trail_y,trail_t=getTrail(row)
		target=str(row[2]).split(',')
		label=str(row[3]) if len(row)==3 else -1
		getFeature(trail_x,trail_y,trail_t,target)
