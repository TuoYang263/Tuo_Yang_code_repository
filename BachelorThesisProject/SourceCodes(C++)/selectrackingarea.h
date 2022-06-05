#pragma once
// ����:����ѡ��Ŀ��λ��
using namespace cv;
// �ƶ���� ѡȡ���ο�
void mouseClickCallback(int event,
	int x, int y, int flags, void* userdata)
{
	// �������ݷ���
	cv::Rect2d * pRect =
		reinterpret_cast<cv::Rect2d*>(userdata);
	// ��갴�²���
	if (event == cv::EVENT_LBUTTONDOWN)
	{
		std::cout << "LBUTTONDOWN ("
			<< x << ", " << y << ")" << std::endl;
		// ��ȡx��y����
		pRect->x = x;
		pRect->y = y;
	}
	// ���̧�����
	else if (event == cv::EVENT_LBUTTONUP)
	{
		std::cout << "LBUTTONUP ("
			<< x << ", " << y << ")" << std::endl;
		// ��ȡ���ο��
		pRect->width = std::abs(x - pRect->x);
		pRect->height = std::abs(y - pRect->y);
	}
}
