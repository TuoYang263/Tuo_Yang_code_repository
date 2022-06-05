#include <fstream>
#include <stdio.h>
#include <sstream>
#include <iostream>
#include <math.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "kcftracker.hpp"
#include "ffttools.hpp"
#include "recttools.hpp"
#include "fhog.hpp"
#include "labdata.hpp"
#include "ransac.h"
#include "RGBHistogram.h"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video/tracking.hpp"
using namespace std;
using namespace cv;
static int frameNumber1 = 0;
static int frameNumber2 = 0;
static Mat lastFrame;
static Rect lastRoi;
static Mat lastSecondFrame;
static Rect lastSecondRoi;
static Mat lastThirdFrame;
static Rect lastThirdRoi;
int stateNum = 4;
int measureNum = 2;
static KalmanFilter KF(stateNum, measureNum, 0);
static Mat measurement;
static Rect kalmanRoi;
static bool isShelter = false;
static Mat beforeShelter;
static Rect beforeRoi;
static int index = 0;
static int _except_count = 0;
static float lastPSR = 0;


RNG rng(12345);
struct Target
{
	Point leftTop;
	float width = 0;
	float height = 0;
};

Target preTarget;

void initKalmanTracker()
{
	//kalman参数设置
	measurement = Mat::zeros(measureNum, 1, CV_32F);
	KF.transitionMatrix = (Mat_<float>(stateNum, stateNum) << 1, 0, 1, 0,
		0, 1, 0, 1,
		0, 0, 1, 0,
		0, 0, 0, 1);
	//A 状态转移矩阵
	//这里没有设置控制矩阵B，默认为零
	setIdentity(KF.measurementMatrix);//H=[1,0,0,0;0,1,0,0] 测量矩阵
	setIdentity(KF.processNoiseCov, Scalar::all(1e-5));//Q高斯白噪声，单位阵
	setIdentity(KF.measurementNoiseCov, Scalar::all(1e-1));//R高斯白噪声，单位阵
	setIdentity(KF.errorCovPost, Scalar::all(1));//P后验误差估计协方差矩阵，初始化为单位阵
	randn(KF.statePost, Scalar::all(0), Scalar::all(0.1));//初始化状态为随机值
}

Rect updateKalmanFilter(const cv::Mat currentFrame,const cv::Rect currentRoi)
{
	Mat frame1, frame2;
	frame1 = currentFrame;
	resize(frame1, frame2, Size(), 1, 1, INTER_LINEAR);
	Mat prediction = KF.predict();
	Point predict_pt = Point((int)prediction.at<float>(0), (int)prediction.at<float>(1));
	preTarget.leftTop.x = currentRoi.x;
	preTarget.leftTop.y = currentRoi.y;
	preTarget.width = currentRoi.width;
	preTarget.height = currentRoi.height;
	measurement.at<float>(0) = (float)preTarget.leftTop.x;
	measurement.at<float>(1) = (float)preTarget.leftTop.y;
	KF.correct(measurement);
	//画框
	//Point center(predict_pt.x + preTarget.width*0.5, predict_pt.y + preTarget.height*0.5);
	//ellipse(frame2, center, Size(preTarget.width*0.3, preTarget.height*0.3), 0, 0, 360, Scalar(0, 0, 255), 4, 8, 0);
	//circle(frame2, center, 3, Scalar(0, 0, 255), -1);
	//imshow("kalman", frame2);
	//cvWaitKey(0);
	cout << "kalman's x" << predict_pt.x << endl;
	cout << "kalman's y" << predict_pt.y << endl;
	Rect rect1;
	rect1.x = predict_pt.x;
	rect1.y = predict_pt.y;
	rect1.width = currentRoi.width;
	rect1.height = currentRoi.height;
	return rect1;
}

void KCFTracker::getCentralErrorAndOverlapRate(const cv::Rect &roi,const cv::Rect &base,float *centralError,float *overlapRate)
{
	float param1, param2;
	param1 = (roi.x + roi.width / 2.0f)-(base.x + base.width / 2.0f);
	param2 = (base.y + base.height / 2.0f) - (roi.y + roi.height / 2.0f);
	(*centralError) = sqrt(pow(param1, 2) + pow(param2, 2));
	float x0 = std::max(roi.x, base.x);
	float x1 = std::min(roi.x + roi.width, base.x + base.width);
	float y0 = std::max(roi.y, base.y);
	float y1 = std::min(roi.y + roi.height, base.y + base.height);
	if (x0 >= x1 || y0 >= y1)
		(*overlapRate) = 0.f;
	else
	{
		float areaInt = (x1 - x0)*(y1 - y0);
		(*overlapRate) = areaInt / ((float)roi.width*roi.height + (float)base.width*base.height - areaInt);
	}
}

// Constructor
// 初始化KCF类参数
KCFTracker::KCFTracker(bool hog, bool fixed_window, bool multiscale, bool lab)
{

	// Parameters equal in all cases
	lambda = 0.0001;
	padding = 2.5;
	//output_sigma_factor = 0.1;
	output_sigma_factor = 0.125;


	if (hog) {    // HOG
				  // VOT
		interp_factor = 0.012;
		sigma = 0.6;
		// TPAMI
		//interp_factor = 0.02;
		//sigma = 0.5; 
		cell_size = 4;
		_hogfeatures = true;

		if (lab) {
			interp_factor = 0.005;
			sigma = 0.4;
			//output_sigma_factor = 0.025;
			output_sigma_factor = 0.1;

			_labfeatures = true;
			_labCentroids = cv::Mat(nClusters, 3, CV_32FC1, &dataset);
			cell_sizeQ = cell_size * cell_size;
		}
		else {
			_labfeatures = false;
		}
	}
	else {   // RAW
		interp_factor = 0.075;
		sigma = 0.2;
		cell_size = 1;
		_hogfeatures = false;

		if (lab) {
			printf("Lab features are only used with HOG features.\n");
			_labfeatures = false;
		}
	}


	if (multiscale) { // multiscale
		template_size = 96;
		//template_size = 100;
		scale_step = 1.20;//1.05;
		scale_weight = 0.95;
		if (!fixed_window) {
			//printf("Multiscale does not support non-fixed window.\n");
			fixed_window = true;
		}
	}
	else if (fixed_window) {  // fit correction without multiscale
		template_size = 96;
		//template_size = 100;
		scale_step = 1;
	}
	else {
		template_size = 1;
		scale_step = 1;
	}
}

// Initialize tracker 
// 使用第一帧和它的跟踪框，初始化KCF跟踪器 不用的参数,float dataset1[],float dataset2[],fstream &filepointer
void KCFTracker::init(const cv::Rect &roi, cv::Mat image)
{
	Mat img;
	if (image.channels() == 1)
		img = image;
	else
		cvtColor(image, img, COLOR_BGR2GRAY);
	//imshow("gray image", img);
	_roi = roi;
	assert(roi.width >= 0 && roi.height >= 0);
	_tmpl = getFeatures(image, 1);																				// 获取特征，在train里面每帧修改
	//namedWindow("melt feature", 0);
	//imshow("melt feature", _tmpl);
	_prob = createGaussianPeak(size_patch[0], size_patch[1]);							// 这个不修改了，只初始化一次
	_alphaf = cv::Mat(size_patch[0], size_patch[1], CV_32FC2, float(0));	// 获取特征，在train里面每帧修改
																			//_num = cv::Mat(size_patch[0], size_patch[1], CV_32FC2, float(0));
																			//_den = cv::Mat(size_patch[0], size_patch[1], CV_32FC2, float(0));
	train(_tmpl, 1.0); // train with initial frame
	
	/*char cstring[50];	//从这里开始计算中心误差
	float centralError, overlapRate;	//中心误差和覆盖率变量
	int baseX = 0, baseY = 0, baseWidth = 0, baseHeight = 0;
	filepointer.getline(cstring, sizeof(cstring));
	sscanf(cstring, "%d,%d,%d,%d", &baseX, &baseY, &baseWidth, &baseHeight);	//作用与scanf类似
	getCentralErrorAndOverlapRate(roi,Rect(baseX,baseY,baseWidth,baseHeight),&centralError,&overlapRate);
	//判定centralError和overlapRate是否符合要求
	if (centralError < 20)
		dataset1[frameNumber1++] = (int)(centralError*100)/100.0;
	if (overlapRate >= 0.6)
		dataset2[frameNumber2++] = (int)(overlapRate * 100)/100.0;			//保留小数点后两位，不四舍五入*/
	lastFrame = img;
	lastRoi = roi;
	initKalmanTracker();
}

bool KCFTracker::_update(Mat &image, Rect2f &objRect,float peakValue)
{
	if (peakValue < _peak_val_threshold)
	{
		_except_count++;
		if (_except_count > ALLOWED_EXCEPT_COUNT)
			return false;
		else
			return true;
	}
	else
	{
		_except_count = 0;
		return true;
	}
}

// Update position based on the new frame
// 基于当前帧更新目标位置 不用的参数,float dataset1[],float dataset2[],fstream &filepointer
cv::Rect KCFTracker::update(cv::Mat image)
{
	/*char cstring[50];	//从这里开始计算中心误差
	float centralError, overlapRate;	//中心误差和覆盖率变量
	int baseX = 0, baseY = 0, baseWidth = 0, baseHeight = 0;
	filepointer.getline(cstring, sizeof(cstring));
	sscanf(cstring, "%d,%d,%d,%d", &baseX, &baseY, &baseWidth, &baseHeight);	//作用与scanf类似
	float centerX = baseX + baseWidth / 2.0f;
	float centerY = baseY + baseHeight / 2.0f;*/
	// 修正边界
	if (_roi.x + _roi.width <= 0) _roi.x = -_roi.width + 1;
	if (_roi.y + _roi.height <= 0) _roi.y = -_roi.height + 1;
	if (_roi.x >= image.cols - 1) _roi.x = image.cols - 2;
	if (_roi.y >= image.rows - 1) _roi.y = image.rows - 2;

	// 跟踪框中心
	float cx = _roi.x + _roi.width / 2.0f;
	float cy = _roi.y + _roi.height / 2.0f;

	// 尺度不变时检测峰值结果
	float peak_value;
	cv::Point2f res = detect(_tmpl, getFeatures(image, 0, 1.0f), peak_value);
	//namedWindow("_tmpl", 0);
	//imshow("_tmpl",_tmpl);
	//cvWaitKey(0);
	//cv::rectangle(image, _roi, cv::Scalar(255, 0, 0), 2, 8);
	//imshow("scare1", image);
	//cout << "scare1's width and height:"<<_roi.x<<" "<<_roi.y<< endl;

	// 略大尺度和略小尺度进行检测
	if (scale_step != 1) {
		// Test at a smaller _scale
		// 使用一个小点的尺度测试
		float new_peak_value;
		cv::Point2f new_res = detect(_tmpl, getFeatures(image, 0, 1.0f / scale_step), new_peak_value);

		// 做减益还比同尺度大就认为是目标
		if (scale_weight * new_peak_value > peak_value) {
			res = new_res;
			peak_value = new_peak_value;
			_scale /= scale_step;
			_roi.width /= scale_step;
			_roi.height /= scale_step;
		}
		new_res = detect(_tmpl, getFeatures(image, 0, scale_step), new_peak_value);
		//cv::rectangle(image, _roi, cv::Scalar(0, 255, 0), 2, 8);
		//imshow("scare2", image);
		//cout << "scare2's width and height:" << _roi.x << " " << _roi.y << endl;
		// Test at a bigger _scale
		

		if (scale_weight * new_peak_value > peak_value) {
			res = new_res;
			peak_value = new_peak_value;
			_scale *= scale_step;
			_roi.width *= scale_step;
			_roi.height *= scale_step;
		}
		new_res = detect(_tmpl, getFeatures(image, 0, scale_step), new_peak_value);
		//cv::rectangle(image, _roi, cv::Scalar(0, 0, 255), 2, 8);
		//imshow("scare3", image);
		//cout << "scare3's width and height:" << _roi.x << " " << _roi.y << endl;
	}

	// Adjust by cell size and _scale
	// 因为返回的只有中心坐标，使用尺度和中心坐标调整目标框
	_roi.x = cx - _roi.width / 2.0f + ((float)res.x * cell_size * _scale);
	_roi.y = cy - _roi.height / 2.0f + ((float)res.y * cell_size * _scale);

	if (_roi.x >= image.cols - 1) _roi.x = image.cols - 1;
	if (_roi.y >= image.rows - 1) _roi.y = image.rows - 1;
	if (_roi.x + _roi.width <= 0) _roi.x = -_roi.width + 2;
	if (_roi.y + _roi.height <= 0) _roi.y = -_roi.height + 2;

	assert(_roi.width >= 0 && _roi.height >= 0);
	kalmanRoi = updateKalmanFilter(lastFrame, lastRoi);
	cout << "peak_value:" << peak_value<< endl;
	if (!_update(image, _roi, peak_value)&& (frameNumber1 + 1)>2)	//PSR <(PSR+lastPSR)/2.0
	{
		cout << "kalman" << endl;
		cout << "出现遮挡" << endl;
		if (index != 0)
			_roi = kalmanRoi;
		index++;
		cout << "_except_count:" << _except_count << endl;
		cout << "ALLOWED_EXCEPT_COUNT:" << ALLOWED_EXCEPT_COUNT << endl;
	}
	else
	{
		cv::Mat x = getFeatures(image, 0);
		train(x, interp_factor);
		cout << "kcf" << endl;
	}
	frameNumber1++;
	cout << "tracking value:"<<peak_value << endl;
	/*getCentralErrorAndOverlapRate(_roi, Rect(baseX, baseY, baseWidth, baseHeight), &centralError, &overlapRate);
	//判定centralError和overlapRate是否符合要求
	if (centralError < 20)
		dataset1[frameNumber1++] = (int)(centralError * 100) / 100.0;
	if (overlapRate >= 0.6)
		dataset2[frameNumber2++] = (int)(overlapRate * 100) / 100.0;//保留小数点后两位，不四舍五入*/
	lastRoi = _roi;
	lastFrame = image;
	return _roi;		//返回检测框
}

// Detect object in the current frame.
// z为前一帧样本
// x为当前帧图像
// peak_value为输出的峰值
cv::Point2f KCFTracker::detect(cv::Mat z, cv::Mat x, float &peak_value)
{
	using namespace FFTTools;

	// 做变换得到计算结果res
	cv::Mat k = gaussianCorrelation(x, z);
	//namedWindow("x", 0);
	//imshow("x", x);		    //response map
	//cvWaitKey(0);
	cv::Mat res = (real(fftd(complexMultiplication(_alphaf, fftd(k)), true)));
	//namedWindow("res", 0);
	//imshow("res", res);		    //response map
	//cvWaitKey(0);

	//minMaxLoc only accepts doubles for the peak, and integer points for the coordinates
	// 使用opencv的minMaxLoc来定位峰值坐标位置
	cv::Point2i pi;
	double pv;
	cv::minMaxLoc(res, NULL, &pv, NULL, &pi);
	peak_value = (float)pv;

	//subpixel peak estimation, coordinates will be non-integer
	// 子像素峰值检测，坐标是非整形的
	cv::Point2f p((float)pi.x, (float)pi.y);

	if (pi.x > 0 && pi.x < res.cols - 1) {
		p.x += subPixelPeak(res.at<float>(pi.y, pi.x - 1), peak_value, res.at<float>(pi.y, pi.x + 1));
	}

	if (pi.y > 0 && pi.y < res.rows - 1) {
		p.y += subPixelPeak(res.at<float>(pi.y - 1, pi.x), peak_value, res.at<float>(pi.y + 1, pi.x));
	}

	p.x -= (res.cols) / 2;
	p.y -= (res.rows) / 2;

	return p;
}

// train tracker with a single image
// 使用图像进行训练，得到当前帧的_tmpl，_alphaf
void KCFTracker::train(cv::Mat x, float train_interp_factor)
{
	using namespace FFTTools;

	cv::Mat k = gaussianCorrelation(x, x);
	cv::Mat alphaf = complexDivision(_prob, (fftd(k) + lambda));

	_tmpl = (1 - train_interp_factor) * _tmpl + (train_interp_factor)* x;
	_alphaf = (1 - train_interp_factor) * _alphaf + (train_interp_factor)* alphaf;

	/*cv::Mat kf = fftd(gaussianCorrelation(x, x));
	cv::Mat num = complexMultiplication(kf, _prob);
	cv::Mat den = complexMultiplication(kf, kf + lambda);

	_tmpl = (1 - train_interp_factor) * _tmpl + (train_interp_factor) * x;
	_num = (1 - train_interp_factor) * _num + (train_interp_factor) * num;
	_den = (1 - train_interp_factor) * _den + (train_interp_factor) * den;
	_alphaf = complexDivision(_num, _den);*/

}

// Evaluates a Gaussian kernel with bandwidth SIGMA for all relative shifts between input images X and Y,
// which must both be MxN. They must    also be periodic (ie., pre-processed with a cosine window).
// 使用带宽SIGMA计算高斯卷积核以用于所有图像X和Y之间的相对位移
// 必须都是MxN大小。二者必须都是周期的（即，通过一个cos窗口进行预处理）
cv::Mat KCFTracker::gaussianCorrelation(cv::Mat x1, cv::Mat x2)
{
	using namespace FFTTools;
	cv::Mat c = cv::Mat(cv::Size(size_patch[1], size_patch[0]), CV_32F, cv::Scalar(0));
	// HOG features
	if (_hogfeatures) {
		cv::Mat caux;
		cv::Mat x1aux;
		cv::Mat x2aux;
		for (int i = 0; i < size_patch[2]; i++) {
			x1aux = x1.row(i);   // Procedure do deal with cv::Mat multichannel bug
			x1aux = x1aux.reshape(1, size_patch[0]);
			x2aux = x2.row(i).reshape(1, size_patch[0]);
			cv::mulSpectrums(fftd(x1aux), fftd(x2aux), caux, 0, true);
			caux = fftd(caux, true);
			rearrange(caux);
			caux.convertTo(caux, CV_32F);
			c = c + real(caux);
		}
	}
	// Gray features
	else {
		cv::mulSpectrums(fftd(x1), fftd(x2), c, 0, true);
		c = fftd(c, true);
		rearrange(c);
		c = real(c);
	}
	cv::Mat d;
	cv::max(((cv::sum(x1.mul(x1))[0] + cv::sum(x2.mul(x2))[0]) - 2. * c) / (size_patch[0] * size_patch[1] * size_patch[2]), 0, d);

	cv::Mat k;
	cv::exp((-d / (sigma * sigma)), k);
	return k;
}

// Create Gaussian Peak. Function called only in the first frame.
// 创建高斯峰函数，函数只在第一帧的时候执行
cv::Mat KCFTracker::createGaussianPeak(int sizey, int sizex)
{
	cv::Mat_<float> res(sizey, sizex);

	int syh = (sizey) / 2;
	int sxh = (sizex) / 2;

	float output_sigma = std::sqrt((float)sizex * sizey) / padding * output_sigma_factor;
	float mult = -0.5 / (output_sigma * output_sigma);

	for (int i = 0; i < sizey; i++)
		for (int j = 0; j < sizex; j++)
		{
			int ih = i - syh;
			int jh = j - sxh;
			res(i, j) = std::exp(mult * (float)(ih * ih + jh * jh));
		}
	return FFTTools::fftd(res);
}

// Obtain sub-window from image, with replication-padding and extract features
// 从图像得到子窗口，通过赋值填充并检测特征
cv::Mat KCFTracker::getFeatures(const cv::Mat &image, bool inithann, float scale_adjust)
{
	cv::Rect extracted_roi;
	Mat img;
	image.copyTo(img);

	float cx = _roi.x + _roi.width / 2;
	float cy = _roi.y + _roi.height / 2;
	// 初始化hanning窗，	其实只执行一次，只在第一帧的时候inithann=1
	if (inithann) {
		int padded_w = _roi.width * padding;
		int padded_h = _roi.height * padding;
        

		// 按照长宽比例修改长宽大小，保证比较大的边为template_size大小
		if (template_size > 1) {  // Fit largest dimension to the given template size
			if (padded_w >= padded_h)  //fit to width
				_scale = padded_w / (float)template_size;
			else
				_scale = padded_h / (float)template_size;

			_tmpl_sz.width = padded_w / _scale;
			_tmpl_sz.height = padded_h / _scale;
		}
		else {  //No template size given, use ROI size
			_tmpl_sz.width = padded_w;
			_tmpl_sz.height = padded_h;
			_scale = 1;
			// original code from paper:
			/*if (sqrt(padded_w * padded_h) >= 100) {   //Normal size
			_tmpl_sz.width = padded_w;
			_tmpl_sz.height = padded_h;
			_scale = 1;
			}
			else {   //ROI is too big, track at half size
			_tmpl_sz.width = padded_w / 2;
			_tmpl_sz.height = padded_h / 2;
			_scale = 2;
			}*/
		}

		// 设置_tmpl_sz的长宽：向上取原来长宽的最小2*cell_size倍
		// 其中，较大边长为104
		if (_hogfeatures) {
			// Round to cell size and also make it even
			_tmpl_sz.width = (((int)(_tmpl_sz.width / (2 * cell_size))) * 2 * cell_size) + cell_size * 2;
			_tmpl_sz.height = (((int)(_tmpl_sz.height / (2 * cell_size))) * 2 * cell_size) + cell_size * 2;
		}
		else {  //Make number of pixels even (helps with some logic involving half-dimensions)
			_tmpl_sz.width = (_tmpl_sz.width / 2) * 2;
			_tmpl_sz.height = (_tmpl_sz.height / 2) * 2;
		}
	}

	// 检测区域大小
	extracted_roi.width = scale_adjust * _scale * _tmpl_sz.width;
	extracted_roi.height = scale_adjust * _scale * _tmpl_sz.height;

	// center roi with new size
	// 检测区域坐上角坐标
	extracted_roi.x = cx - extracted_roi.width / 2;
	extracted_roi.y = cy - extracted_roi.height / 2;

	//cv::rectangle(img, extracted_roi, cv::Scalar(255, 0, 0), 2, 8);
	//namedWindow("scare1", 0);
	//imshow("scare1", img);
	
	// 提取目标区域像素，超边界则做填充
	cv::Mat FeaturesMap;
	cv::Mat z = RectTools::subwindow(image, extracted_roi, cv::BORDER_REPLICATE);
	//cv::rectangle(z, extracted_roi, cv::Scalar(255, 0, 0), 2, 8);
	//namedWindow("scare1", 0);
	//imshow("scare1", z);
	

	// 按照比例缩小边界大小
	if (z.cols != _tmpl_sz.width || z.rows != _tmpl_sz.height) {
		cv::resize(z, z, _tmpl_sz);
	}

	// HOG features
	// 提取HOG特征点
	if (_hogfeatures) {
		IplImage z_ipl = z;
		CvLSVMFeatureMapCaskade *map;									// 申请指针
		getFeatureMaps(&z_ipl, cell_size, &map);			// 给map进行赋值
		normalizeAndTruncate(map, 0.2f);								// 归一化
		PCAFeatureMaps(map);													// 由HOG特征变为PCA-HOG
		size_patch[0] = map->sizeY;
		size_patch[1] = map->sizeX;
		size_patch[2] = map->numFeatures;

		FeaturesMap = cv::Mat(cv::Size(map->numFeatures, map->sizeX*map->sizeY), CV_32F, map->map);  // Procedure do deal with cv::Mat multichannel bug
		FeaturesMap = FeaturesMap.t();
		freeFeatureMapObject(&map);
		//namedWindow("HOG feature", 0);
		//cvShowImage("HOG feature",&z_ipl);
		//imshow("HOG feature map", FeaturesMap);
		//cvWaitKey(0);

		// Lab features
		// 我测试结果，带有Lab特征在一些跟踪环节效果并不好
		if (_labfeatures) {
			cv::Mat imgLab;
			cvtColor(z, imgLab, CV_BGR2Lab);
			//namedWindow("lab feature", 0);
			//imshow("lab feature", imgLab);
			//cvWaitKey(0);
			unsigned char *input = (unsigned char*)(imgLab.data);

			// Sparse output vector
			cv::Mat outputLab = cv::Mat(_labCentroids.rows, size_patch[0] * size_patch[1], CV_32F, float(0));

			int cntCell = 0;
			// Iterate through each cell
			for (int cY = cell_size; cY < z.rows - cell_size; cY += cell_size) {
				for (int cX = cell_size; cX < z.cols - cell_size; cX += cell_size) {
					// Iterate through each pixel of cell (cX,cY)
					for (int y = cY; y < cY + cell_size; ++y) {
						for (int x = cX; x < cX + cell_size; ++x) {
							// Lab components for each pixel
							float l = (float)input[(z.cols * y + x) * 3];
							float a = (float)input[(z.cols * y + x) * 3 + 1];
							float b = (float)input[(z.cols * y + x) * 3 + 2];

							// Iterate trough each centroid
							float minDist = FLT_MAX;
							int minIdx = 0;
							float *inputCentroid = (float*)(_labCentroids.data);
							for (int k = 0; k < _labCentroids.rows; ++k) {
								float dist = ((l - inputCentroid[3 * k]) * (l - inputCentroid[3 * k]))
									+ ((a - inputCentroid[3 * k + 1]) * (a - inputCentroid[3 * k + 1]))
									+ ((b - inputCentroid[3 * k + 2]) * (b - inputCentroid[3 * k + 2]));
								if (dist < minDist) {
									minDist = dist;
									minIdx = k;
								}
							}
							// Store result at output
							outputLab.at<float>(minIdx, cntCell) += 1.0 / cell_sizeQ;
							//((float*) outputLab.data)[minIdx * (size_patch[0]*size_patch[1]) + cntCell] += 1.0 / cell_sizeQ; 
						}
					}
					cntCell++;
				}
			}
			// Update size_patch[2] and add features to FeaturesMap
			size_patch[2] += _labCentroids.rows;
			FeaturesMap.push_back(outputLab);
			//namedWindow("Lab feature", 0);
			//imshow("Lab feature", outputLab);
			//cvWaitKey(0);
		}
	}
	else {
		FeaturesMap = RectTools::getGrayImage(z);
		FeaturesMap -= (float) 0.5; // In Paper;
		size_patch[0] = z.rows;
		size_patch[1] = z.cols;
		size_patch[2] = 1;
	}
	
	if (inithann) {
		createHanningMats();
	}
	FeaturesMap = hann.mul(FeaturesMap);
	return FeaturesMap;
}

// Initialize Hanning window. Function called only in the first frame.
// 初始化hanning窗，只执行一次，使用opencv函数做的
void KCFTracker::createHanningMats()
{
	cv::Mat hann1t = cv::Mat(cv::Size(size_patch[1], 1), CV_32F, cv::Scalar(0));
	cv::Mat hann2t = cv::Mat(cv::Size(1, size_patch[0]), CV_32F, cv::Scalar(0));

	for (int i = 0; i < hann1t.cols; i++)
		hann1t.at<float>(0, i) = 0.5 * (1 - std::cos(2 * 3.14159265358979323846 * i / (hann1t.cols - 1)));
	for (int i = 0; i < hann2t.rows; i++)
		hann2t.at<float>(i, 0) = 0.5 * (1 - std::cos(2 * 3.14159265358979323846 * i / (hann2t.rows - 1)));

	cv::Mat hann2d = hann2t * hann1t;
	// HOG features
	if (_hogfeatures) {
		cv::Mat hann1d = hann2d.reshape(1, 1); // Procedure do deal with cv::Mat multichannel bug

		hann = cv::Mat(cv::Size(size_patch[0] * size_patch[1], size_patch[2]), CV_32F, cv::Scalar(0));
		for (int i = 0; i < size_patch[2]; i++) {
			for (int j = 0; j<size_patch[0] * size_patch[1]; j++) {
				hann.at<float>(i, j) = hann1d.at<float>(0, j);
			}
		}
	}
	// Gray features
	else {
		hann = hann2d;
	}
	//namedWindow("hann2d", 0);
	//imshow("hann2d", hann2d);			//hann2d
	//imshow("hann",hann);
	cvWaitKey(0);

}

// Calculate sub-pixel peak for one dimension
// 使用幅值做差来定位峰值的位置，返回的是需要改变的偏移量大小
float KCFTracker::subPixelPeak(float left, float center, float right)
{
	float divisor = 2 * center - right - left;

	if (divisor == 0)
		return 0;

	return 0.5 * (right - left) / divisor;
}