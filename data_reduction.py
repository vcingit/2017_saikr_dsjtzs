# -*- coding: utf-8 -*-
import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.pyplot import plot,savefig

#根据x和y坐标作图
def makeLineChart(x,y,target_x,target_y,label,pic_id,folder_name='train_pic',model=0):
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
			plt.savefig('../%s/0/%s'%(folder_name,pic_id))
		else:
			plt.savefig('../%s/1/%s'%(folder_name,pic_id))
		plt.close('all')

#散点图
def makeScatterChart(x_1,y_1,x_0,y_0):
	
	plt.xlabel('x-axis')
	plt.ylabel('y-axis')
	plt.scatter(x_1, y_1,color='r')
	plt.scatter(x_0, y_0,color='b')
	plt.show()

#读取文件
def readFile(filepath):

	print 'read%s'%(filepath)
	data = pd.read_table(filepath,header=None,encoding='utf-8',delim_whitespace=True,index_col=0)
	print '------------------------------'
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
def dataToChart(filepath,folder_name='train_pic',model=0):
	data=readFile(filepath)
	#row 1-移动轨迹坐标 2-目的坐标 3-标签
	for ix, row in data.iterrows():
		trail_x,trail_y,trail_t=getTrail(row)
		target=str(row[2]).split(',')
		label=str(row[3]) if len(row)==3 else -1
		makeLineChart(trail_x,trail_y,target[0],target[1],label,ix,folder_name,model)

#结果写入文件
def writeResult(result,filepath):
	file=open(filepath,'w')  
	file.write('\n'.join(result));  
	file.close()

#t特征写入文件
def writeFea(fea,label,filepath):
	df = pd.DataFrame(fea, columns=[\
		'each_step_distance_average','each_step_distance_variance',\
		'average_sp','variance_sp',\
		'all_time','all_step_num','if_x_back','if_t_back',\
		'average_x','average_y','average_t',\
		'variance_x','variance_y','variance_t',\
		'obtuse_num','right_num','acute_num',\
		'obtuse_rate','right_rate','acute_rate'])
	df['label']=label
	df.to_csv(filepath,index=False)

#五折交叉验证:产生交叉验证集
def five_split_data(data,label):
	x0=[data[i] for i in range(len(data)) if i % 5 == 0]
	x1=[data[i] for i in range(len(data)) if i % 5 == 1]
	x2=[data[i] for i in range(len(data)) if i % 5 == 2]
	x3=[data[i] for i in range(len(data)) if i % 5 == 3]
	x4=[data[i] for i in range(len(data)) if i % 5 == 4]
	y0=[label[i] for i in range(len(label)) if i % 5 == 0]
	y1=[label[i] for i in range(len(label)) if i % 5 == 1]
	y2=[label[i] for i in range(len(label)) if i % 5 == 2]
	y3=[label[i] for i in range(len(label)) if i % 5 == 3]
	y4=[label[i] for i in range(len(label)) if i % 5 == 4]
	return [x0,x1,x2,x3,x4],[y0,y1,y2,y3,y4]