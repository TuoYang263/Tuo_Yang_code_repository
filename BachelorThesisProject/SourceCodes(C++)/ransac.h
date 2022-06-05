#include <iostream>
using namespace std;
#include <cv.h>
#include <cxcore.h>
#include <highgui.h>
using namespace cv;
#include "sift.h"
#include "my_function.h"
#include<opencv2/imgproc/imgproc.hpp>
#include "opencv2/calib3d/calib3d.hpp"	//findHomography����ͷ�ļ�

Rect siftMatch(Mat srcImgObject, Mat srcImgScene,Rect lastRoi)
{
	//��������ͼƬ
	Mat src1 = srcImgObject;
	Mat src2 = srcImgScene;

	//���ĸ�������ģ��ͼ������ɫ������ĸ�����
	Point2f m1(lastRoi.x,lastRoi.y), m2(lastRoi.x, lastRoi.y+lastRoi.height),
		m3(lastRoi.x+lastRoi.width, lastRoi.y+lastRoi.height), m4(lastRoi.x + lastRoi.width, lastRoi.y);
	std::vector<Point2f> obj_corners(4);
	obj_corners[0] = m1;
	obj_corners[1] = m2;
	obj_corners[2] = m3;
	obj_corners[3] = m4;

	//����ͼ�������������
	vector<Keypoint> feature_1, feature_2;
	
	//����sift�㷨���������������У����SIFT��������������ļ���д�õ�
	Sift(src1, feature_1, 1.6);
	Sift(src2, feature_2, 1.6);

	//feature_disΪ���о����������ṹ������
	vector<Keypoint> feature_dis_1;  
	vector<Keypoint> feature_dis_2;
	vector<Keypoint> result;
	KeyPoint x;

	cout << "���" << endl;
	//�����������ƥ�䣬���Match_feature�����Լ�д�ģ����ǲ�������ȴν�С��0.8��Ϊ���ʵ�ƥ�䣬����ƥ�䷽ʽ
	//openCV�в�û�У������Ҿ��Լ�д��
	Match_feature(feature_1, feature_2, feature_dis_1, feature_dis_2);
	cout << "����" << endl;

	//�����￪ʼʹ��RANSAC������������
	//����ĳ��򶼺����Σ����еĽṹ��ֻ��ת����openCV�����Ͳ�����openCV�ĺ�������
	Ptr<DescriptorMatcher> descriptor_matcher = DescriptorMatcher::create("BruteForce");//��������ƥ����  
	int count = feature_dis_1.size();

	//������������ת����openCV�ܹ�ʹ�õ�����
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

	cout << "����" << endl;

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
    
	cout << "�ĵ�" << endl;
	// ��RANSAC��������F
	Mat m_Fundamental;
	// ������������ǻ�������
	vector<uchar> m_RANSACStatus;
	// ������������Ѿ�����������ڴ洢RANSAC��ÿ�����״̬

	//һ��ʼʹ��findFundamentalMat���������ֿ�����������ƥ�䣬ʵ�ֺܺõ�Ч��������
	//���Ǻ�������ֵ���Ǳ任���󣬶���û��ʲô�õĻ�������
	m_Fundamental = findFundamentalMat(p1, p2, m_RANSACStatus, CV_FM_RANSAC);

	//����ʹ��findHomography��������������ķ���ֵ���������ı任����
	Mat m_homography;
	vector<uchar> m;
	m_homography = findHomography(p1, p2, CV_RANSAC, 3, m);

	//�ɱ任������ñ任�������߽��ĸ���
	std::vector<Point2f> scene_corners(4);
	perspectiveTransform(obj_corners, scene_corners, m_homography);
	cout << "���" << endl;
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
