# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import pandas as pd
#根据x和y坐标作图
def makeLineChart(x,y,target_x,target_y,label):
	if len(x)==0 or len(y) ==0:
		pass
	plt.title('label:%s'%(label))
	plt.xlabel('x-axis')
	plt.ylabel('y-axis')
	plt.plot(x, y,'r',label='line')
	if target_x !=None and target_y !=None:
		#plt.plot(target_x,target_y,'b',label='point')
		plt.plot(target_x,target_y,'b',marker='*')
		# plt.annotate("target", xy = (target_x, target_y), xytext = (-4, 50),\
		# 	arrowprops = dict(facecolor = "r", headlength = 10, headwidth = 30, width = 10))  

	plt.legend(bbox_to_anchor=[0.3, 1])
	plt.grid()
	plt.show()

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
		makeLineChart(trail_x,trail_y,target[0],target[1],label)

def writeResult(result,filepath):
	file=open(filepath,'w')  
	file.write('\n'.join(result));  
	file.close()

if __name__ == "__main__":
	data=readFile('data/dsjtzs_txfz_test1.txt')
	res=[]
	#print 'end_x','end_y','end_t','target_x','target_y','x_D_value_rate','y_D_value_rate'
	for ix, row in data.iterrows():
		trail_x,trail_y,trail_t=getTrail(row)
		target=str(row[2]).split(',')
		label=str(row[3]) if len(row)==3 else -1
		
		end_x,end_y,end_t=float(trail_x[-1]),float(trail_y[-1]),float(trail_t[-1])
		target_x,target_y=float(target[0]),float(target[1])
		
		x_D_value_rate=abs(end_x-target_x)/end_x*100.0
		y_D_value_rate=abs(end_y-target_y)/end_y*100.0
		
		
		#print end_x,end_y,end_t,target_x,target_y,x_D_value_rate,y_D_value_rate
		if end_y<0 or end_x<0 or x_D_value_rate<2 or end_t>20000 or end_t<1000:
			res.append(0)
		else:
			res.append(1)
	result=[]
	for i in range(len(res)):
		if res[i]==0:
			result.append(str(i+1))
	writeResult(result,'result/submit.txt')