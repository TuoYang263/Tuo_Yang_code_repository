#include <iostream>
#include <fstream>
#include <algorithm>
#include <malloc.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/ml.hpp"
#include "dirent.h"
#include "selectrackingarea.h"
#include "Imagetxt.h"
#include "DrawGraph.h"
#include <sys/timeb.h>
#include<WInSock.h>
#include<string>
#include "kcftracker.hpp"
#include "TrackTask.h"
#include <windows.h>

using namespace cv;
using namespace std;
static fstream filepointer;

int getFileCount(string startFilePath)
{
	string strtemp;
	string token = ".";
	string tokend = "..";
	int frameCount = 0;
	HANDLE hfile;
	WIN32_FIND_DATA fileDate;//WIN32_FIND_DATA结构描述了一个由FindFirstFile, FindFirstFileEx, 或FindNextFile函数查找到的文件信息
	DWORD errorcode = 0;
	hfile = FindFirstFileA((startFilePath + "\\*.*").c_str(), &fileDate);
	//通过FindFirstFileA（）函数,根据当前的文件存放路径查找该文件来把待操作文件的相关属性读取到WIN32_FIND_DATA结构中去
	while (hfile != INVALID_HANDLE_VALUE && errorcode != ERROR_NO_MORE_FILES)
	{

		strtemp = fileDate.cFileName;
		bool flag = false;
		if ((fileDate.dwFileAttributes == FILE_ATTRIBUTE_DIRECTORY) && (strtemp != token) && (strtemp != tokend))
		{
			//如果当前文件是目录文件，则递归调用findFile
			flag = true;
			cout << strtemp << " is a direcotry" << endl;
			getFileCount(startFilePath + "\\" + strtemp);
		}
		else if (fileDate.dwFileAttributes&FILE_ATTRIBUTE_ARCHIVE)
			frameCount++;
		bool isNextFile = FindNextFileA(hfile, &fileDate);//判断该目录下是否还有文件
		if (flag == true && isNextFile == true)//如果还有文件，则调用SetLastError，设为NO_ERROR，这样才能继续遍历后面的文件
			SetLastError(NO_ERROR);
		else
			errorcode = GetLastError();
	}
	return frameCount;
}

void drawCentralErrorGraph(float *centralError,float *frames,int nFrames)
{
	CPlot plotDemo;
	plotDemo.y_max = 20;
	plotDemo.y_min = 0;
	plotDemo.x_max = nFrames;
	plotDemo.x_min = 0;
	plotDemo.plot<float>(frames, centralError, nFrames, CvScalar(0, 0, 0), '.', true);
	plotDemo.title("Central Position Error Graph");
	plotDemo.xlabel("Frames", CvScalar(0, 0, 0));
	plotDemo.ylabel("CentralPositionError(pt)", CvScalar(0, 0, 0));
	cvNamedWindow("CentralError");
	cvShowImage("CentralError",plotDemo.Figure);
}

void drawOverlapRateGraph(float *overlapRate, float *frames, int nFrames)
{
	CPlot plotDemo;
	plotDemo.y_max = 1;
	plotDemo.y_min = 0;
	plotDemo.x_max = nFrames;
	plotDemo.x_min = 0;
	plotDemo.plot<float>(frames, overlapRate, nFrames, CvScalar(0, 0, 0), '.', true);
	plotDemo.title("Overlap Rate Graph");
	plotDemo.xlabel("Frames", CvScalar(0, 0, 0));
	plotDemo.ylabel("OverlapRate(0-1)", CvScalar(0, 0, 0));
	cvNamedWindow("OverlapRate");
	cvShowImage("OverlapRate", plotDemo.Figure);
}

int main(int argc, char* argv[]) {
	struct timeval tv, tz, tv0, tz0; 
	TrackTask conf;
	conf.SetArgs(argc,argv);
	for (int i = 0; i < argc; i++)
		cout << "argv[" << i << "]=" << argv[i] << endl;

	//if (argc > 5) return -1;            // 输入大于5个参数,这里实际上丢进来的有11个参数

	bool HOG = true;                    // 是否使用hog特征
	bool FIXEDWINDOW = true;           // 是否使用修正窗口
	bool MULTISCALE = true;             // 是否使用多尺度
	bool SILENT = false;                 // 是否不做显示
	bool LAB = true;                   // 是否使用LAB颜色

	for (int i = 0; i < argc; i++) {
		if (strcmp(argv[i], "hog") == 0)
			HOG = true;
		if (strcmp(argv[i], "fixed_window") == 0)
			FIXEDWINDOW = true;
		if (strcmp(argv[i], "singlescale") == 0)
			MULTISCALE = false;
		if (strcmp(argv[i], "show") == 0)
			SILENT = false;
		if (strcmp(argv[i], "lab") == 0) {
			LAB = true;
			HOG = true;
		}
		if (strcmp(argv[i], "gray") == 0)
			HOG = false;
	}

	// Create KCFTracker object
	// 创建KCF跟踪器
	KCFTracker tracker(HOG, FIXEDWINDOW, MULTISCALE, LAB);

	// Frame readed
	// 当前帧
	Mat frame;

	// Tracker results
	// 跟踪结果目标框
	Rect result;

	//Using min and max of X and Y for groundtruth rectangle
	float xMin = conf.Bbox.x;
	float yMin = conf.Bbox.y;
	float width = conf.Bbox.width;
	float height = conf.Bbox.height;
	float tha = 0.40;
	static int fps = 0;
	static int lastTime = getTickCount();
	static int endTime;
	int keyValue;
	int delay = 30;
	for (int frameId = conf.StartFrmId, i = 1; frameId <= conf.EndFrmId; ++frameId, ++i)
	{
		//Read each frame from the list
		frame = conf.GetFrm(frameId);
		//First frame,give the groundtruth to the tracker
		if (i == 1)
		{
			result = Rect(xMin, yMin, width, height);
			tracker.init(result, frame);
		}
		else
		{
			result = tracker.update(frame);
		}
		keyValue = waitKey(delay);
		if (keyValue == 27)
			break;
		if (delay >= 0 && keyValue == 32)
		{
			delay = 0;
		}
		conf.PushResult(result);
	}
	endTime = getTickCount();
	cout << "FPS:" << ((endTime - lastTime) / 1000)/conf.EndFrmId<<endl;
	conf.SaveResults();
	system("pause");
	// Path to list.txt
	// images.txt的路径，用于读取图像
	/*ifstream listFile;
	string fileName = "images.txt";
	listFile.open(fileName);

	string tmpPath;
	string imgFilePath = generateImageTxt();
	tmpPath += imgFilePath+"0001.jpg";
	//cout << "tmpPath:" << tmpPath << endl;
	//Mat startFrame = imread(tmpPath);
	/*
	cv::Rect2d *rect(new cv::Rect2d);
	cv::setMouseCallback("Multi-Scale KCF Tracking", mouseClickCallback, reinterpret_cast<void*>(rect));
	cvWaitKey(0);*/
	//cv::imshow("Multi-Scale KCF Tracking", startFrame);

	// Using min and max of X and Y for groundtruth rectangle
	// 使用四个顶点计算出目标框
	/*float xMin = rect->x;
	float yMin = rect->y;
	float width = rect->width;
	float height = rect->height;
	float xMin, yMin, width, height;

	//cout << xMin << yMin << width << height << endl;
	//system("pause");


	// Read Images
	// 读图像
    ifstream listFramesFile;
	string listFrames = "images.txt";
	listFramesFile.open(listFrames);
	string frameName;

	// Write Results
	// 将结果写入output.txt
	ofstream resultsFile;
	string resultsPath = "output.txt";
	resultsFile.open(resultsPath);

	string a = imgFilePath;	//指定要替换的主串
	string b = "img\\";	//指定要替换的子串
	string c = "";	//要替换的内容
	int pos = a.find(b);
	if (pos != -1)
	{
		a.replace(a.find(b), b.length(), c);
	}
	a += "groundtruth_rect.txt";
	char *d = (char*)a.c_str();
	cout << "d:" << d << endl;
	//system("pause");
	fstream initalPos;
	char info[50];
	initalPos.open(d, fstream::in | fstream::out | fstream::app);
	initalPos.getline(info, sizeof(info));
	sscanf(info, "%f,%f,%f,%f", &xMin, &yMin, &width, &height);
	initalPos.close();
	filepointer.open(d ,fstream::in|fstream::out|fstream::app);
	// Frame counter
	// 帧号计数
	int nFrames = 0;
	int totalFrame = getFileCount(imgFilePath);
	float * frames = (float*)malloc(totalFrame * sizeof(float));
	float * centralError = (float*)malloc(totalFrame * sizeof(float));
	float * overlapRate = (float*)malloc(totalFrame * sizeof(float));
	int index = 0;
	char name_write[15] = {};

	int delay = 30;
	int keyValue;
	double fps;
	char fpsInfo[10];
	double t = 0;
	while (getline(listFramesFile, frameName)) {
		t = (double)cv::getTickCount();
		frameName = frameName;

		// Read each frame from the list
		// 读取列表上面的帧
		frame = imread(frameName, CV_LOAD_IMAGE_COLOR);

		// First frame, give the groundtruth to the tracker
		// 使用第一帧和目标框来初始化跟踪器
		if (nFrames == 0) {
			tracker.init(Rect(xMin, yMin, width, height), frame, centralError,overlapRate,filepointer);
			//cout << "hello" << endl;
			rectangle(frame, Rect(xMin, yMin, width, height), Scalar(0, 255, 255), 1, 8);
			resultsFile << xMin << "," << yMin << "," << width << "," << height << endl;
			frames[index] = index++;
		}
		// Update
		// 更新当前帧的结果
		else {
			result = tracker.update(frame, centralError, overlapRate,filepointer);
			rectangle(frame, Point(result.x, result.y), Point(result.x + result.width, result.y + result.height), Scalar(0, 255, 255), 1, 8);
			resultsFile << result.x << "," << result.y << "," << result.width << "," << result.height << endl;
			frames[index] = index++;
		}
		nFrames++;

		// 显示并保存
		if (!SILENT) {
			t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
			fps = 1.0 / t;
			sprintf(fpsInfo, "%.2f", fps);
			std::string fpstring("FPS:");
			fpstring += fpsInfo;
			putText(frame, fpstring, cv::Point(5, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 200, 0));
			imshow("Image", frame);
			cout << "current frame:" << nFrames << endl;
			/*if (nFrames == 200)
			{
				cvWaitKey(0);
			}
			waitKey(1);
			//sprintf(name_write, "%d.jpg", nFrames);
			//imwrite(name_write, frame);
		}
		keyValue = waitKey(delay);
		if (keyValue == 27)
			break;
		if (delay >= 0 && keyValue == 32)
		{
			delay = 0;
		}
	}
	for (int i = 0; i < totalFrame; i++)
	{
		cout << "centralError[" << i << "]=" << centralError[i] << endl;
		cout << "overlapRate[" << i << "]=" << overlapRate[i] << endl;
	}

	drawCentralErrorGraph(centralError,frames,nFrames);
	drawOverlapRateGraph(overlapRate,frames,nFrames);
	waitKey(0);
	
	system("pause");
	// 关闭文件
	resultsFile.close();

	listFile.close();
	filepointer.close();*/
}