# -*- coding: utf-8 -*-

#根据分数反推预测正确数
#预测数 正确数 得分
def judge_score(a,b,s):
	return int((2*b+3*a)*s/5)

if __name__ == "__main__":
	print judge_score(9678,20000,0.6421)