#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

const int HISTSIZE = 8;

Mat bgrHistogram(const Mat& src)
{
	//∑÷¿ÎB°¢G°¢RÕ®µ¿
	vector<Mat> bgr_planes;
	split(src, bgr_planes);


	float range[] = { 0, 256 };
	const float* histRange = { range };

	bool uniform = true; bool accumulate = false;

	Mat hist1d, normHist1d, hist;

	for (int i = 0; i < 3; i++)
	{
		calcHist(&bgr_planes[i], 1, 0, Mat(), hist1d, 1, &HISTSIZE, &histRange, uniform, accumulate);
		normalize(hist1d, hist1d, 1.0, 0.0, CV_L1);
		hist.push_back(hist1d);
	}
	return hist;
}
