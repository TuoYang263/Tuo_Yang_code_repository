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
	WIN32_FIND_DATA fileDate;//WIN32_FIND_DATA�ṹ������һ����FindFirstFile, FindFirstFileEx, ��FindNextFile�������ҵ����ļ���Ϣ
	DWORD errorcode = 0;
	hfile = FindFirstFileA((startFilePath + "\\*.*").c_str(), &fileDate);
	//ͨ��FindFirstFileA��������,���ݵ�ǰ���ļ����·�����Ҹ��ļ����Ѵ������ļ���������Զ�ȡ��WIN32_FIND_DATA�ṹ��ȥ
	while (hfile != INVALID_HANDLE_VALUE && errorcode != ERROR_NO_MORE_FILES)
	{

		strtemp = fileDate.cFileName;
		bool flag = false;
		if ((fileDate.dwFileAttributes == FILE_ATTRIBUTE_DIRECTORY) && (strtemp != token) && (strtemp != tokend))
		{
			//�����ǰ�ļ���Ŀ¼�ļ�����ݹ����findFile
			flag = true;
			cout << strtemp << " is a direcotry" << endl;
			getFileCount(startFilePath + "\\" + strtemp);
		}
		else if (fileDate.dwFileAttributes&FILE_ATTRIBUTE_ARCHIVE)
			frameCount++;
		bool isNextFile = FindNextFileA(hfile, &fileDate);//�жϸ�Ŀ¼���Ƿ����ļ�
		if (flag == true && isNextFile == true)//��������ļ��������SetLastError����ΪNO_ERROR���������ܼ�������������ļ�
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

	//if (argc > 5) return -1;            // �������5������,����ʵ���϶���������11������

	bool HOG = true;                    // �Ƿ�ʹ��hog����
	bool FIXEDWINDOW = true;           // �Ƿ�ʹ����������
	bool MULTISCALE = true;             // �Ƿ�ʹ�ö�߶�
	bool SILENT = false;                 // �Ƿ�����ʾ
	bool LAB = true;                   // �Ƿ�ʹ��LAB��ɫ

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
	// ����KCF������
	KCFTracker tracker(HOG, FIXEDWINDOW, MULTISCALE, LAB);

	// Frame readed
	// ��ǰ֡
	Mat frame;

	// Tracker results
	// ���ٽ��Ŀ���
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
	// images.txt��·�������ڶ�ȡͼ��
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
	// ʹ���ĸ���������Ŀ���
	/*float xMin = rect->x;
	float yMin = rect->y;
	float width = rect->width;
	float height = rect->height;
	float xMin, yMin, width, height;

	//cout << xMin << yMin << width << height << endl;
	//system("pause");


	// Read Images
	// ��ͼ��
    ifstream listFramesFile;
	string listFrames = "images.txt";
	listFramesFile.open(listFrames);
	string frameName;

	// Write Results
	// �����д��output.txt
	ofstream resultsFile;
	string resultsPath = "output.txt";
	resultsFile.open(resultsPath);

	string a = imgFilePath;	//ָ��Ҫ�滻������
	string b = "img\\";	//ָ��Ҫ�滻���Ӵ�
	string c = "";	//Ҫ�滻������
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
	// ֡�ż���
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
		// ��ȡ�б������֡
		frame = imread(frameName, CV_LOAD_IMAGE_COLOR);

		// First frame, give the groundtruth to the tracker
		// ʹ�õ�һ֡��Ŀ�������ʼ��������
		if (nFrames == 0) {
			tracker.init(Rect(xMin, yMin, width, height), frame, centralError,overlapRate,filepointer);
			//cout << "hello" << endl;
			rectangle(frame, Rect(xMin, yMin, width, height), Scalar(0, 255, 255), 1, 8);
			resultsFile << xMin << "," << yMin << "," << width << "," << height << endl;
			frames[index] = index++;
		}
		// Update
		// ���µ�ǰ֡�Ľ��
		else {
			result = tracker.update(frame, centralError, overlapRate,filepointer);
			rectangle(frame, Point(result.x, result.y), Point(result.x + result.width, result.y + result.height), Scalar(0, 255, 255), 1, 8);
			resultsFile << result.x << "," << result.y << "," << result.width << "," << result.height << endl;
			frames[index] = index++;
		}
		nFrames++;

		// ��ʾ������
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
	// �ر��ļ�
	resultsFile.close();

	listFile.close();
	filepointer.close();*/
}