#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <iostream>
#include <fstream>
using namespace std;

class Tracker
{
public:
	Tracker() {}
	virtual  ~Tracker() { }

	virtual void init(const cv::Rect &roi, cv::Mat image) = 0;
	virtual cv::Rect  update(cv::Mat image) = 0;


protected:
	cv::Rect_<float> _roi;								//responding box
};
