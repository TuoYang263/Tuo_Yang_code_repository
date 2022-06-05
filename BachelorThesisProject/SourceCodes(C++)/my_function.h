#include <cv.h>
#include <cxcore.h>
#include <highgui.h>
#include<stdlib.h> 
#include<time.h>  
#include "sift.h"
#include <fstream>
#include <iostream>
#include "imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"	

//#include<math.h>
using namespace cv;
using namespace std;

//圆周率
#define PI  3.1415926535

#define match_threshold 1.0

//将多个初始图片和一个实验图片进行比较
//图片数量的十位和个位，以及总数，通过改变个位和十位的数字，即可实现改变训练图像的数量
#define PhotoAmountDecade 1   
#define PhotoAmountUnit 4
#define Count PhotoAmountDecade*10+PhotoAmountUnit

//圈出物体的方框的高度比宽度
#define ratio 0.9      //玩具熊_1
//#define ratio 1.76   //圆柱盒子_2
//#define ratio 0.8    //砚台_3
//#define ratio 0.7    //鞋_4
//#define ratio 0.9      //玩具车_5

//方框内的匹配点书占总数的百分比
#define percentage 0.5

//表示匹配点的结构体
struct Match_point
{
	CvPoint coo;//匹配点的坐标
	double sum;//匹配度
	int x;//聚类分析时，表示从属于哪一个中心点    然后这个功能并没有什么卵用。。。2015.6.1
};


struct Key_point :Keypoint
{
	double distance;
	int match;
};

void Match_feature(vector<Keypoint>& feature_init, vector<Keypoint>& feature_final, vector<Keypoint>& feature_1, vector<Keypoint>& feature_2)
{

	int m = 0;
	for (int i = 0; i<feature_init.size(); i++)
	{
		//

		Key_point test;
		double min_1 = 0;//最近的
		double min_2 = 0;//次近的

		int x = 0;
		for (int j = 0; j<feature_final.size(); j++)
		{

			double sum = 0;
			for (int k = 0; k<128; k++)
			{
				sum = sum + (feature_init[i].descriptor[k] - feature_final[j].descriptor[k])*(feature_init[i].descriptor[k] - feature_final[j].descriptor[k]);
			}
			sum = sqrt(sum);
			if (j == 0)
			{
				min_1 = sum;
			}
			else
			{
				if (sum<min_1)
				{
					min_2 = min_1;
					min_1 = sum;
					x = j;
				}
				else if (sum>min_1&&sum<min_2)
				{
					min_2 = sum;
				}
			}

		}

		if (min_1 / min_2>0.8)continue;

		//对匹配的特征点计算方向的差值，将方向差别较大的匹配点删除


		test.octave = feature_init[i].octave;
		test.interval = feature_init[i].interval;
		test.offset_interval = feature_init[i].offset_interval;
		test.x = feature_init[i].x;
		test.y = feature_init[i].y;
		test.scale = feature_init[i].scale;
		test.dx = feature_init[i].dx;
		test.dy = feature_init[i].dy;
		test.offset_x = feature_init[i].offset_x;
		test.offset_y = feature_init[i].offset_y;
		test.octave_scale = feature_init[i].octave_scale;
		test.ori = feature_init[i].ori;
		test.descr_length = feature_init[i].descr_length;
		test.val = feature_init[i].val;
		for (int k = 0; k<FEATURE_ELEMENT_LENGTH; k++)
			test.descriptor[k] = feature_init[i].descriptor[k];
		//test.distance=min;
		test.match = m;

		feature_1.push_back(test);


		test.octave = feature_final[x].octave;
		test.interval = feature_final[x].interval;
		test.offset_interval = feature_final[x].offset_interval;
		test.x = feature_final[x].x;
		test.y = feature_final[x].y;
		test.scale = feature_final[x].scale;
		test.dx = feature_final[x].dx;
		test.dy = feature_final[x].dy;
		test.offset_x = feature_final[x].offset_x;
		test.offset_y = feature_final[x].offset_y;
		test.octave_scale = feature_final[x].octave_scale;
		test.ori = feature_final[x].ori;
		test.descr_length = feature_final[x].descr_length;
		test.val = feature_final[x].val;
		for (int k = 0; k<FEATURE_ELEMENT_LENGTH; k++)
			test.descriptor[k] = feature_final[x].descriptor[k];
		//test.distance=min;
		test.match = m;//记录相匹配的1图像中的特征点的顺序

		feature_2.push_back(test);
		m++;
	}

}


