#pragma once
// 功能:计算选定目标位置
using namespace cv;
// 移动鼠标 选取矩形框
void mouseClickCallback(int event,
	int x, int y, int flags, void* userdata)
{
	// 矩形数据返回
	cv::Rect2d * pRect =
		reinterpret_cast<cv::Rect2d*>(userdata);
	// 鼠标按下操作
	if (event == cv::EVENT_LBUTTONDOWN)
	{
		std::cout << "LBUTTONDOWN ("
			<< x << ", " << y << ")" << std::endl;
		// 获取x，y坐标
		pRect->x = x;
		pRect->y = y;
	}
	// 鼠标抬起操作
	else if (event == cv::EVENT_LBUTTONUP)
	{
		std::cout << "LBUTTONUP ("
			<< x << ", " << y << ")" << std::endl;
		// 获取矩形宽高
		pRect->width = std::abs(x - pRect->x);
		pRect->height = std::abs(y - pRect->y);
	}
}
