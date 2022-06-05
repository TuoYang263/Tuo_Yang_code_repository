#include <iostream>
using namespace std;
#include <cv.h>
#include <cxcore.h>
#include <highgui.h>
using namespace cv;
#include "sift.h"
#include "my_function.h"
#include<opencv2/imgproc/imgproc.hpp>
#include "opencv2/calib3d/calib3d.hpp"	//findHomography所需头文件

Rect siftMatch(Mat srcImgObject, Mat srcImgScene,Rect lastRoi)
{
	//加载两幅图片
	Mat src1 = srcImgObject;
	Mat src2 = srcImgScene;

	//这四个坐标是模板图像中绿色方框的四个顶点
	Point2f m1(lastRoi.x,lastRoi.y), m2(lastRoi.x, lastRoi.y+lastRoi.height),
		m3(lastRoi.x+lastRoi.width, lastRoi.y+lastRoi.height), m4(lastRoi.x + lastRoi.width, lastRoi.y);
	std::vector<Point2f> obj_corners(4);
	obj_corners[0] = m1;
	obj_corners[1] = m2;
	obj_corners[2] = m3;
	obj_corners[3] = m4;

	//两个图像的特征点序列
	vector<Keypoint> feature_1, feature_2;
	
	//采用sift算法，计算特征点序列，这个SIFT函数是在另外的文件中写好的
	Sift(src1, feature_1, 1.6);
	Sift(src2, feature_2, 1.6);

	//feature_dis为带有距离的特征点结构体序列
	vector<Keypoint> feature_dis_1;  
	vector<Keypoint> feature_dis_2;
	vector<Keypoint> result;
	KeyPoint x;

	cout << "你好" << endl;
	//对特征点进行匹配，这个Match_feature是我自己写的，就是采用最近比次近小于0.8即为合适的匹配，这种匹配方式
	//openCV中并没有，所以我就自己写了
	Match_feature(feature_1, feature_2, feature_dis_1, feature_dis_2);
	cout << "二货" << endl;

	//从这里开始使用RANSAC方法进行运算
	//下面的程序都好无奈，所有的结构都只能转化成openCV的类型才能用openCV的函数。。
	Ptr<DescriptorMatcher> descriptor_matcher = DescriptorMatcher::create("BruteForce");//创建特征匹配器  
	int count = feature_dis_1.size();

	//把特征点序列转化成openCV能够使用的类型
	vector<KeyPoint>Key_points1, Key_points2;
	KeyPoint keyp;
	for (int i = 0; i<count; i++)
	{
		keyp.pt.x = feature_dis_1[i].dx;
		keyp.pt.y = feature_dis_1[i].dy;
		Key_points1.push_back(keyp);
		keyp.pt.x = feature_dis_2[i].dx;
		keyp.pt.y = feature_dis_2[i].dy;
		Key_points2.push_back(keyp);
	}

	cout << "三逼" << endl;

	Mat descriptors1(count, FEATURE_ELEMENT_LENGTH, CV_32F);
	Mat descriptors2(count, FEATURE_ELEMENT_LENGTH, CV_32F);

	for (int i = 0; i<count; i++)
	{
		for (int j = 0; j<FEATURE_ELEMENT_LENGTH; j++)
		{
			descriptors1.at<float>(i, j) = feature_dis_1[i].descriptor[j];
			descriptors2.at<float>(i, j) = feature_dis_2[i].descriptor[j];
		}
	}

	Mat p1(feature_dis_1.size(), 2, CV_32F);
	Mat p2(feature_dis_1.size(), 2, CV_32F);
	for (int i = 0; i<feature_dis_1.size(); i++)
	{
		p1.at<float>(i, 0) = feature_dis_1[i].dx;
		p1.at<float>(i, 1) = feature_dis_1[i].dy;
		p2.at<float>(i, 0) = feature_dis_2[i].dx;
		p2.at<float>(i, 1) = feature_dis_2[i].dy;
	}
    
	cout << "四弟" << endl;
	// 用RANSAC方法计算F
	Mat m_Fundamental;
	// 上面这个变量是基本矩阵
	vector<uchar> m_RANSACStatus;
	// 上面这个变量已经定义过，用于存储RANSAC后每个点的状态

	//一开始使用findFundamentalMat函数，发现可以消除错误匹配，实现很好的效果，但是
	//就是函数返回值不是变换矩阵，而是没有什么用的基础矩阵
	m_Fundamental = findFundamentalMat(p1, p2, m_RANSACStatus, CV_FM_RANSAC);

	//这里使用findHomography函数，这个函数的返回值才是真正的变换矩阵
	Mat m_homography;
	vector<uchar> m;
	m_homography = findHomography(p1, p2, CV_RANSAC, 3, m);

	//由变换矩阵，求得变换后的物体边界四个点
	std::vector<Point2f> scene_corners(4);
	perspectiveTransform(obj_corners, scene_corners, m_homography);
	cout << "五哥" << endl;
	//cout << "scene_corners[0].x:" << scene_corners[0].x << endl;
	//cout << "scene_corners[0].y:" << scene_corners[0].y << endl;
	//line(src2, scene_corners[0], scene_corners[1], Scalar(0, 0, 255), 2);
	//line(src2, scene_corners[1], scene_corners[2], Scalar(0, 0, 255), 2);
	//line(src2, scene_corners[2], scene_corners[3], Scalar(0, 0, 255), 2);
	//line(src2, scene_corners[3], scene_corners[0], Scalar(0, 0, 255), 2);
	//imshow("src2", src2);

	//cvWaitKey(0);
	Rect targetRoi;
	targetRoi.x = m1.x;
	targetRoi.y = m1.y;
	targetRoi.width = m3.x - m1.x;
	targetRoi.height = m3.y - m1.y;
	return targetRoi;
}
