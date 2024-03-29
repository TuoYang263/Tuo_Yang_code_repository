#ifndef SIFT_H
#define SIFT_H

#include <cv.h>
#include <cxcore.h>
#include <highgui.h>
#include <vector>
#include <fstream>
#include <iostream>
#include<opencv2/opencv.hpp>
using namespace cv;
using namespace std;

typedef double pixel_t;//像素类型

					   //初始sigma
#define INIT_SIGMA 0.5

					   //sigma 
#define SIGMA 1.6

					   //intervals
#define INTERVALS 3

					   //r
#define RATIO 10

#define MAX_INTERPOLATION_STEPS 5 

					   //|D(x^)| < 0.03   0.03极值点太多
#define DXTHRESHOLD 0.04

					   //bins = 36
#define ORI_HIST_BINS 36    

#define ORI_SIGMA_TIMES 1.5

#define ORI_WINDOW_RADIUS 3.0 * ORI_SIGMA_TIMES 

#define ORI_SMOOTH_TIMES 2

#define ORI_PEAK_RATIO 0.8

#define FEATURE_ELEMENT_LENGTH 128

#define DESCR_HIST_BINS 8

#define DESCR_WINDOW_WIDTH 4

#define DESCR_SCALE_ADJUST 3

#define DESCR_MAG_THR 0.2

#define INT_DESCR_FCTR 512.0


struct Keypoint
{
	int octave; //关键点所在组
	int interval;// 关键点所在层

	double offset_interval;//调整后的层的增量

	int x; //x,y坐标,根据octave和interval可取的层内图像
	int y;

	//scale = sigma0*pow(2.0, o+s/S)
	double scale; //空间尺度坐标

	double dx; //特征点坐标，该坐标被缩放成原图像大小 
	double dy;

	double offset_x;
	double offset_y;

	//高斯金字塔组内各层尺度坐标，不同组的相同层的sigma值相同
	//关键点所在组的组内尺度
	double octave_scale; //offset_i;

	double ori;//方向

	int descr_length;//描述子的维度，在这里为128维
	double descriptor[FEATURE_ELEMENT_LENGTH]; //

	double val;//极值
};

//转换为灰度图像
void ConvertToGray(const Mat& src, Mat& dst)
{
	Size size = src.size();
	if (dst.empty())
		dst.create(size, CV_64F);
	//cout << "type: "<< src.type() << " " << dst.type()<<endl;

	uchar* srcData = src.data;
	pixel_t* dstData = (pixel_t*)dst.data;
	int dstStep = dst.step / sizeof(dstData[0]);

	for (int j = 0; j < src.cols; j++)
	{
		for (int i = 0; i < src.rows; i++)
		{

			double b = *(srcData + src.step * i + src.channels() * j + 0) / 255.0;
			double g = *(srcData + src.step * i + src.channels() * j + 1) / 255.0;
			double r = *(srcData + src.step * i + src.channels() * j + 2) / 255.0;

			*((dstData + dstStep * i + dst.channels() * j)) = (b + g + r) / 3.0;
		}
	}
}

//隔点采样
void DownSample(const Mat& src, Mat& dst)
{
	if (src.channels() != 1)
		return;

	if (src.cols <= 1 || src.rows <= 1)
	{
		src.copyTo(dst);
		return;
	}

	dst.create((int)(src.rows / 2), (int)(src.cols / 2), src.type());
	//cout<<"-- "<<dst.rows<<" " <<dst.cols << " --"<<endl;

	pixel_t* srcData = (pixel_t*)src.data;
	pixel_t* dstData = (pixel_t*)dst.data;

	int srcStep = src.step / sizeof(srcData[0]);
	int dstStep = dst.step / sizeof(dstData[0]);

	int m = 0, n = 0;
	for (int j = 0; j < src.cols; j += 2, n++)
	{
		m = 0;
		for (int i = 0; i < src.rows; i += 2, m++)
		{
			pixel_t sample = *(srcData + srcStep * i + src.channels() * j);

			//防止当图像长宽不一致时，长宽为奇数时，m,n越界
			if (m < dst.rows && n < dst.cols)
			{
				*(dstData + dstStep * m + dst.channels() * n) = sample;

			}
		}
	}

}

//线性插值放大
void UpSample(const Mat &src, Mat &dst)
{
	if (src.channels() != 1)
		return;
	dst.create(src.rows * 2, src.cols * 2, src.type());

	pixel_t* srcData = (pixel_t*)src.data;
	pixel_t* dstData = (pixel_t*)dst.data;

	int srcStep = src.step / sizeof(srcData[0]);
	int dstStep = dst.step / sizeof(dstData[0]);

	int m = 0, n = 0;
	for (int j = 0; j < src.cols - 1; j++, n += 2)
	{
		m = 0;
		for (int i = 0; i < src.rows - 1; i++, m += 2)
		{
			double sample = *(srcData + srcStep * i + src.channels() * j);
			*(dstData + dstStep * m + dst.channels() * n) = sample;

			double rs = *(srcData + srcStep * (i)+src.channels()*j) + (*(srcData + srcStep * (i + 1) + src.channels()*j));
			*(dstData + dstStep * (m + 1) + dst.channels() * n) = rs / 2;
			double cs = *(srcData + srcStep * i + src.channels()*(j)) + (*(srcData + srcStep * i + src.channels()*(j + 1)));
			*(dstData + dstStep * m + dst.channels() * (n + 1)) = cs / 2;

			double center = (*(srcData + srcStep * (i + 1) + src.channels() * j))
				+ (*(srcData + srcStep * i + src.channels() * j))
				+ (*(srcData + srcStep * (i + 1) + src.channels() * (j + 1)))
				+ (*(srcData + srcStep * i + src.channels() * (j + 1)));

			*(dstData + dstStep * (m + 1) + dst.channels() * (n + 1)) = center / 4;

		}

	}



	if (dst.rows < 3 || dst.cols < 3)
		return;

	//最后两行两列
	for (int k = dst.rows - 1; k >= 0; k--)
	{
		*(dstData + dstStep * (k)+dst.channels()*(dst.cols - 2)) = *(dstData + dstStep * (k)+dst.channels()*(dst.cols - 3));
		*(dstData + dstStep * (k)+dst.channels()*(dst.cols - 1)) = *(dstData + dstStep * (k)+dst.channels()*(dst.cols - 3));
	}
	for (int k = dst.cols - 1; k >= 0; k--)
	{
		*(dstData + dstStep * (dst.rows - 2) + dst.channels()*(k)) = *(dstData + dstStep * (dst.rows - 3) + dst.channels()*(k));
		*(dstData + dstStep * (dst.rows - 1) + dst.channels()*(k)) = *(dstData + dstStep * (dst.rows - 3) + dst.channels()*(k));
	}

}

//高斯平滑
//未使用sigma，边缘无处理
void GaussianTemplateSmooth(const Mat &src, Mat &dst, double sigma)
{
	//高斯模板(7*7)，sigma = 0.84089642，归一化后得到
	static const double gaussianTemplate[7][7] =
	{
		{ 0.00000067, 0.00002292, 0.00019117, 0.00038771, 0.00019117, 0.00002292, 0.00000067 },
	{ 0.00002292, 0.00078633, 0.00655965, 0.01330373, 0.00655965, 0.00078633, 0.00002292 },
	{ 0.00019117, 0.00655965, 0.05472157, 0.11098164, 0.05472157, 0.00655965, 0.00019117 },
	{ 0.00038771, 0.01330373, 0.11098164, 0.22508352, 0.11098164, 0.01330373, 0.00038771 },
	{ 0.00019117, 0.00655965, 0.05472157, 0.11098164, 0.05472157, 0.00655965, 0.00019117 },
	{ 0.00002292, 0.00078633, 0.00655965, 0.01330373, 0.00655965, 0.00078633, 0.00002292 },
	{ 0.00000067, 0.00002292, 0.00019117, 0.00038771, 0.00019117, 0.00002292, 0.00000067 }
	};

	dst.create(src.size(), src.type());
	uchar* srcData = src.data;
	uchar* dstData = dst.data;

	for (int j = 0; j < src.cols - 7; j++)
	{
		for (int i = 0; i < src.rows - 7; i++)
		{
			double acc = 0;
			double accb = 0, accg = 0, accr = 0;
			for (int m = 0; m < 7; m++)
			{
				for (int n = 0; n < 7; n++)
				{
					if (src.channels() == 1)
						acc += *(srcData + src.step * (i + n) + src.channels() * (j + m)) * gaussianTemplate[m][n];
					else
					{
						accb += *(srcData + src.step * (i + n) + src.channels() * (j + m) + 0) * gaussianTemplate[m][n];
						accg += *(srcData + src.step * (i + n) + src.channels() * (j + m) + 1) * gaussianTemplate[m][n];
						accr += *(srcData + src.step * (i + n) + src.channels() * (j + m) + 2) * gaussianTemplate[m][n];
					}
				}
			}
			if (src.channels() == 1)
				*(dstData + dst.step * (i + 3) + dst.channels() * (j + 3)) = (int)acc;
			else
			{
				*(dstData + dst.step * (i + 3) + dst.channels() * (j + 3) + 0) = (int)accb;
				*(dstData + dst.step * (i + 3) + dst.channels() * (j + 3) + 1) = (int)accg;
				*(dstData + dst.step * (i + 3) + dst.channels() * (j + 3) + 2) = (int)accr;
			}
		}
	}

}




void GaussianSmooth2D(const Mat &src, Mat &dst, double sigma)
{
	if (src.channels() != 1)
		return;

	//确保sigma为正数 
	sigma = sigma > 0 ? sigma : 0;
	//高斯核矩阵的大小为(6*sigma+1)*(6*sigma+1)
	//ksize为奇数
	int ksize = cvRound(sigma * 3) * 2 + 1;

	//cout << "ksize=" <<ksize<<endl;
	//	dst.create(src.size(), src.type());
	if (ksize == 1)
	{
		src.copyTo(dst);
		return;
	}

	dst.create(src.size(), src.type());

	//计算高斯核矩阵
	double *kernel = new double[ksize*ksize];

	double scale = -0.5 / (sigma*sigma);
	const double PIValue = 3.141592653;
	double cons = -scale / PIValue;

	double sum = 0;

	for (int i = 0; i < ksize; i++)
	{
		for (int j = 0; j < ksize; j++)
		{
			int x = i - (ksize - 1) / 2;
			int y = j - (ksize - 1) / 2;
			kernel[i*ksize + j] = cons * exp(scale * (x*x + y * y));

			sum += kernel[i*ksize + j];
			//			cout << " " << kernel[i*ksize + j];
		}
		//		cout <<endl;
	}
	//归一化
	for (int i = ksize * ksize - 1; i >= 0; i--)
	{
		*(kernel + i) /= sum;
	}
	/*
	ofstream out("output.txt");
	for(int i = 0; i < ksize; i++)
	{
	for(int j = 0; j < ksize; j++)
	{
	//		cout << " " << kernel[i*ksize + j];
	out << " " << kernel[i*ksize + j];
	}
	//	cout <<endl;
	out <<endl;
	}
	*/
	uchar* srcData = src.data;
	uchar* dstData = dst.data;

	//图像卷积运算
	for (int j = 0; j < src.cols - ksize; j++)
	{
		for (int i = 0; i < src.rows - ksize; i++)
		{
			double acc = 0;

			for (int m = 0; m < ksize; m++)
			{
				for (int n = 0; n < ksize; n++)
				{
					acc += *(srcData + src.step * (i + n) + src.channels() * (j + m)) * kernel[m*ksize + n];
				}
			}

			/*
			for(int l = 0; l < ksize * ksize; l++)
			acc +=  *(srcData + src.step * (i+(int)l/ksize) + src.channels() * (j+(int)l%ksize)) * kernel[l];
			*/
			*(dstData + dst.step * (i + (ksize - 1) / 2) + (j + (ksize - 1) / 2)) = (int)acc;
		}
	}

	//模板边缘用原象素填充
	/*
	for(int j = 0; j < src.cols; j++)
	{
	for(int i = src.rows - ksize; i < src.rows; i++)
	{
	*(dstData + dst.step * i + j) = *(srcData + src.step * i + j);
	*(dstData + dst.step * j + i) = *(srcData + src.step * j + i);
	}

	for(int i = 0; i < ksize; i++)
	{
	*(dstData + dst.step * i + j) = *(srcData + src.step * i + j);
	*(dstData + dst.step * j + i) = *(srcData + src.step * j + i);
	}
	}
	*/
	delete[]kernel;
}

void GaussianSmooth(const Mat &src, Mat &dst, double sigma)
{
	GaussianBlur(src, dst, Size(0, 0), sigma);
	/*
	if(src.channels() != 1 && src.channels() != 3)
	return;

	//
	sigma = sigma > 0 ? sigma : -sigma;
	//高斯核矩阵的大小为(6*sigma+1)*(6*sigma+1)
	//ksize为奇数
	int ksize = cvRound(sigma * 3) * 2 + 1;

	//cout << "ksize=" <<ksize<<endl;
	//	dst.create(src.size(), src.type());
	if(ksize == 1)
	{
	src.copyTo(dst);
	return;
	}

	//计算一维高斯核
	double *kernel = new double[ksize];

	double scale = -0.5/(sigma*sigma);
	const double PI = 3.141592653;
	double cons = 1/sqrt(-scale / PI);

	double sum = 0;
	int kcenter = ksize/2;
	int i = 0, j = 0;
	for(i = 0; i < ksize; i++)
	{
	int x = i - kcenter;
	*(kernel+i) = cons * exp(x * x * scale);//一维高斯函数
	sum += *(kernel+i);

	//		cout << " " << *(kernel+i);
	}
	//	cout << endl;
	//归一化,确保高斯权值在[0,1]之间
	for(i = 0; i < ksize; i++)
	{
	*(kernel+i) /= sum;
	//		cout << " " << *(kernel+i);
	}
	//	cout << endl;

	dst.create(src.size(), src.type());
	Mat temp;
	temp.create(src.size(), src.type());

	pixel_t* srcData = (pixel_t*)src.data;
	pixel_t* dstData = (pixel_t*)dst.data;
	pixel_t* tempData = (pixel_t*)temp.data;

	int srcStep = src.step/sizeof(srcData[0]);
	int dstStep = src.step/sizeof(dstData[0]);
	int tempStep = src.step/sizeof(tempData[0]);


	//x方向一维高斯模糊
	for(int y = 0; y < src.rows; y++)
	{
	for(int x = 0; x < src.cols; x++)
	{
	double mul = 0;
	sum = 0;
	double bmul = 0, gmul = 0, rmul = 0;
	for(i = -kcenter; i <= kcenter; i++)
	{
	if((x+i) >= 0 && (x+i) < src.cols)
	{
	if(src.channels() == 1)
	{
	mul += *(srcData+y*srcStep+(x+i))*(*(kernel+kcenter+i));
	}
	else
	{
	bmul += *(srcData+y*srcStep+(x+i)*src.channels() + 0)*(*(kernel+kcenter+i));
	gmul += *(srcData+y*srcStep+(x+i)*src.channels() + 1)*(*(kernel+kcenter+i));
	rmul += *(srcData+y*srcStep+(x+i)*src.channels() + 2)*(*(kernel+kcenter+i));
	}
	sum += (*(kernel+kcenter+i));
	}
	}
	if(src.channels() == 1)
	{
	*(tempData+y*tempStep+x) = mul/sum;
	}
	else
	{
	*(tempData+y*tempStep+x*temp.channels()+0) = bmul/sum;
	*(tempData+y*tempStep+x*temp.channels()+1) = gmul/sum;
	*(tempData+y*tempStep+x*temp.channels()+2) = rmul/sum;
	}
	}
	}


	//y方向一维高斯模糊
	for(int x = 0; x < temp.cols; x++)
	{
	for(int y = 0; y < temp.rows; y++)
	{
	double mul = 0;
	sum = 0;
	double bmul = 0, gmul = 0, rmul = 0;
	for(i = -kcenter; i <= kcenter; i++)
	{
	if((y+i) >= 0 && (y+i) < temp.rows)
	{
	if(temp.channels() == 1)
	{
	mul += *(tempData+(y+i)*tempStep+x)*(*(kernel+kcenter+i));
	}
	else
	{
	bmul += *(tempData+(y+i)*tempStep+x*temp.channels() + 0)*(*(kernel+kcenter+i));
	gmul += *(tempData+(y+i)*tempStep+x*temp.channels() + 1)*(*(kernel+kcenter+i));
	rmul += *(tempData+(y+i)*tempStep+x*temp.channels() + 2)*(*(kernel+kcenter+i));
	}
	sum += (*(kernel+kcenter+i));
	}
	}
	if(temp.channels() == 1)
	{
	*(dstData+y*dstStep+x) = mul/sum;
	}
	else
	{
	*(dstData+y*dstStep+x*dst.channels()+0) = bmul/sum;
	*(dstData+y*dstStep+x*dst.channels()+1) = gmul/sum;
	*(dstData+y*dstStep+x*dst.channels()+2) = rmul/sum;
	}

	}
	}

	delete[] kernel;
	*/
}

//创建初始灰度图像
//初始图像先将原图像灰度化，再扩大一倍后，使用高斯模糊平滑
void CreateInitSmoothGray(const Mat &src, Mat &dst, double sigma = SIGMA)
{
	Mat gray, up;

	ConvertToGray(src, gray);
	//imshow("gray", gray);
	UpSample(gray, up);

	//-1层的sigma
	double  sigma_init = sqrt(sigma * sigma - (INIT_SIGMA * 2) * (INIT_SIGMA * 2));

	GaussianSmooth(up, dst, sigma_init);
}


//高斯金字塔
void GaussianPyramid(const Mat &src, vector<Mat>&gauss_pyr, int octaves, int intervals = INTERVALS, double sigma = SIGMA)
{
	//
	double *sigmas = new double[intervals + 3];
	double k = pow(2.0, 1.0 / intervals);    //x的y次幂

											 //cout <<"k=" <<k<<endl;
	sigmas[0] = sigma;
	/*
	for(int i = 1; i < intervals+3; i++)
	{
	sigmas[i] = k*sigmas[i-1];
	//cout << " "<<sigmas[i] ;
	}
	*/
	double sig_prev, sig_total;
	for (int i = 1; i < intervals + 3; i++)
	{
		sig_prev = pow(k, i - 1) * sigma;                                 //这里求得每一层的模糊稀疏sigma的方法，不太懂。。。
		sig_total = sig_prev * k;
		sigmas[i] = sqrt(sig_total * sig_total - sig_prev * sig_prev);
	}

	for (int o = 0; o < octaves; o++)
	{
		//每组多三层
		for (int i = 0; i < intervals + 3; i++)
		{
			Mat mat;
			if (o == 0 && i == 0)
			{
				src.copyTo(mat);
			}
			else if (o != 0 && i == 0)
			{
				//前一组的倒数第二层
				DownSample(gauss_pyr[o*(intervals + 3) - 2], mat);
				//				DownSample(gauss_pyr[(o-1)*(intervals+3)+intervals], mat);
			}
			else
			{
				//每组中下一层由上一层高斯模糊得到
				GaussianSmooth(gauss_pyr[o * (intervals + 3) + i - 1], mat, sigmas[i]);
			}
			gauss_pyr.push_back(mat);
		}
	}

	delete[] sigmas;
}

//c = a - b,两个图像做差分，结果存储在第三个Mat对象中
void Sub(const Mat& a, const Mat& b, Mat & c)
{
	if (a.rows != b.rows || a.cols != b.cols || a.type() != b.type())
		return;
	if (!c.empty())
		return;
	c.create(a.size(), a.type());

	pixel_t* ap = (pixel_t*)a.data;
	pixel_t* ap_end = (pixel_t*)a.dataend;
	pixel_t* bp = (pixel_t*)b.data;
	pixel_t* cp = (pixel_t*)c.data;
	int step = a.step / sizeof(pixel_t);

	while (ap != ap_end)
	{
		*cp++ = *ap++ - *bp++;
	}
	/*
	for(int i = 0; i <a.cols; i++ )
	{
	for(int j = 0; j < a.rows; j++)
	{
	*(cp+j*step+i)=*(ap+j*step+i)-(*(bp+j*step+i));
	}
	}
	*/
}

//高斯差分金字塔
void DogPyramid(const vector<Mat>& gauss_pyr, vector<Mat>& dog_pyr, int octaves, int intervals = INTERVALS)
{
	for (int o = 0; o < octaves; o++)
	{
		for (int i = 1; i < intervals + 3; i++)
		{
			Mat mat;
			Sub(gauss_pyr[o*(intervals + 3) + i], gauss_pyr[o*(intervals + 3) + i - 1], mat);
			dog_pyr.push_back(mat);
		}
	}
}


//
bool isExtremum(int x, int y, const vector<Mat>& dog_pyr, int index)
{
	pixel_t * data = (pixel_t *)dog_pyr[index].data;
	int step = dog_pyr[index].step / sizeof(data[0]);
	pixel_t val = *(data + y * step + x);

	if (val > 0)                                                                   //这里检验极值点的代码不太明白。。。
	{
		for (int i = -1; i <= 1; i++)
		{
			int stp = dog_pyr[index + i].step / sizeof(pixel_t);
			for (int j = -1; j <= 1; j++)
			{
				for (int k = -1; k <= 1; k++)
				{
					//检查最大极值
					if (val < *((pixel_t*)dog_pyr[index + i].data + stp * (y + j) + (x + k)))
					{
						return false;
					}
				}
			}
		}
	}
	else
	{
		for (int i = -1; i <= 1; i++)
		{
			int stp = dog_pyr[index + i].step / sizeof(pixel_t);
			for (int j = -1; j <= 1; j++)
			{
				for (int k = -1; k <= 1; k++)
				{
					//检查最小极值
					if (val > *((pixel_t*)dog_pyr[index + i].data + stp * (y + j) + (x + k)))
					{
						return false;
					}
				}
			}
		}
	}

	return true;
}

//4.1 eliminating edge responses
//hessian矩阵，排除边缘点
#define DAt(x, y) (*(data+(y)*step+(x))) 

bool passEdgeResponse(int x, int y, const vector<Mat>& dog_pyr, int index, double r = RATIO)
{
	pixel_t *data = (pixel_t *)dog_pyr[index].data;
	int step = dog_pyr[index].step / sizeof(data[0]);
	pixel_t val = *(data + y * step + x);

	double Dxx, Dyy, Dxy;
	double Tr_h, Det_h;

	//hessian矩阵                                        矩阵这一块，还是看不懂。。。。
	//	   _ 	    _
	//    | Dxx  Dxy |
	// H =|			 |
	//	  |_Dxy  Dyy_|	
	//	  
	Dxx = DAt(x + 1, y) + DAt(x - 1, y) - 2 * val;
	Dyy = DAt(x, y + 1) + DAt(x, y - 1) - 2 * val;
	Dxy = (DAt(x + 1, y + 1) + DAt(x - 1, y - 1) - DAt(x - 1, y + 1) - DAt(x + 1, y - 1)) / 4.0;

	Tr_h = Dxx + Dyy;
	Det_h = Dxx * Dyy - Dxy * Dxy;

	if (Det_h <= 0)
		return false;

	if (Tr_h * Tr_h / Det_h < (r + 1) * (r + 1) / r)
		return true;

	return false;
}

#define Hat(i, j) (*(H+(i)*3 + (j)))
//#define At(index, x, y) (*((pixel_t*)dog_pyr[(index)].data+(y)*((int)(dog_pyr[(index)].step/sizeof((pixel_t*)dog_pyr[index].data[0])))+(x)))

double PyrAt(const vector<Mat>& pyr, int index, int x, int y)
{
	pixel_t *data = (pixel_t*)pyr[index].data;
	int step = pyr[index].step / sizeof(data[0]);
	pixel_t val = *(data + y * step + x);

	return val;
}

#define At(index, x, y) (PyrAt(dog_pyr, (index), (x), (y)))

//3维D(x)一阶偏导,dx列向量
void DerivativeOf3D(int x, int y, const vector<Mat>& dog_pyr, int index, double *dx)
{
	double Dx = (At(index, x + 1, y) - At(index, x - 1, y)) / 2.0;
	double Dy = (At(index, x, y + 1) - At(index, x, y - 1)) / 2.0;
	double Ds = (At(index + 1, x, y) - At(index - 1, x, y)) / 2.0;

	dx[0] = Dx;
	dx[1] = Dy;
	dx[2] = Ds;
}

//3维D(x)二阶偏导，即Hessian矩阵
void Hessian3D(int x, int y, const vector<Mat>& dog_pyr, int index, double *H)
{
	double val, Dxx, Dyy, Dss, Dxy, Dxs, Dys;

	val = At(index, x, y);

	Dxx = At(index, x + 1, y) + At(index, x - 1, y) - 2 * val;
	Dyy = At(index, x, y + 1) + At(index, x, y - 1) - 2 * val;
	Dss = At(index + 1, x, y) + At(index - 1, x, y) - 2 * val;

	Dxy = (At(index, x + 1, y + 1) + At(index, x - 1, y - 1)
		- At(index, x + 1, y - 1) - At(index, x - 1, y + 1)) / 4.0;

	Dxs = (At(index + 1, x + 1, y) + At(index - 1, x - 1, y)
		- At(index - 1, x + 1, y) - At(index + 1, x - 1, y)) / 4.0;

	Dys = (At(index + 1, x, y + 1) + At(index - 1, x, y - 1)
		- At(index + 1, x, y - 1) - At(index - 1, x, y + 1)) / 4.0;

	Hat(0, 0) = Dxx;      //#define Hat(i, j) (*(H+(i)*3 + (j)))   上面有这句
	Hat(1, 1) = Dyy;
	Hat(2, 2) = Dss;

	Hat(1, 0) = Hat(0, 1) = Dxy;
	Hat(2, 0) = Hat(0, 2) = Dxs;
	Hat(2, 1) = Hat(1, 2) = Dys;
}

#define HIat(i, j) (*(H_inve+(i)*3 + (j)))
//3*3阶矩阵求逆
bool Inverse3D(const double *H, double *H_inve)
{
	//A=|H|
	//		 / A00 A01 A02 \				   
	//若H =  | A10 A11 A12 |   
	//		 \ A20 A21 A22 /	
	//则 行列式|H|=A00*A11*A22+A01*A12*A20+A02*A10*A21
	//	    -A00*A12*A21-A01*A10*A22-A02*A11*A20
	//

	double A = Hat(0, 0)*Hat(1, 1)*Hat(2, 2)
		+ Hat(0, 1)*Hat(1, 2)*Hat(2, 0)
		+ Hat(0, 2)*Hat(1, 0)*Hat(2, 1)
		- Hat(0, 0)*Hat(1, 2)*Hat(2, 1)
		- Hat(0, 1)*Hat(1, 0)*Hat(2, 2)
		- Hat(0, 2)*Hat(1, 1)*Hat(2, 0);
	//cout<<A<<endl;
	//没有逆矩阵
	if (fabs(A) < 1e-10)
		return false;



	//三阶逆矩阵运算公式：
	//		 / a b c \				    / ei-hf -(bi-ch) bf-ce\
			//若A =  | d e f |   则A(-1) =1/|H|*| fg-id -(cg-ia) cd-af |
//		 \ g h i /				    \ dh-ge -(ah-gb) ae-bd/



	HIat(0, 0) = Hat(1, 1) * Hat(2, 2) - Hat(2, 1)*Hat(1, 2);
	HIat(0, 1) = -(Hat(0, 1) * Hat(2, 2) - Hat(2, 1) * Hat(0, 2));
	HIat(0, 2) = Hat(0, 1) * Hat(1, 2) - Hat(0, 2)*Hat(1, 1);

	HIat(1, 0) = Hat(1, 2) * Hat(2, 0) - Hat(2, 2)*Hat(1, 0);
	HIat(1, 1) = -(Hat(0, 2) * Hat(2, 0) - Hat(0, 0) * Hat(2, 2));
	HIat(1, 2) = Hat(0, 2) * Hat(1, 0) - Hat(0, 0)*Hat(1, 2);

	HIat(2, 0) = Hat(1, 0) * Hat(2, 1) - Hat(1, 1)*Hat(2, 0);
	HIat(2, 1) = -(Hat(0, 0) * Hat(2, 1) - Hat(0, 1) * Hat(2, 0));
	HIat(2, 2) = Hat(0, 0) * Hat(1, 1) - Hat(0, 1)*Hat(1, 0);

	for (int i = 0; i < 9; i++)
	{
		*(H_inve + i) /= A;
	}
	return true;
}

//计算x^
void GetOffsetX(int x, int y, const vector<Mat>& dog_pyr, int index, double *offset_x)
{
	//x^ = -H^(-1) * dx; dx = (Dx, Dy, Ds)^T
	double H[9], H_inve[9] = { 0 };
	Hessian3D(x, y, dog_pyr, index, H);
	Inverse3D(H, H_inve);
	double dx[3];
	DerivativeOf3D(x, y, dog_pyr, index, dx);

	for (int i = 0; i < 3; i++)
	{
		offset_x[i] = 0.0;
		for (int j = 0; j < 3; j++)
		{
			offset_x[i] += H_inve[i * 3 + j] * dx[j];
		}
		offset_x[i] = -offset_x[i];
	}
}

//计算|D(x^)|
double GetFabsDx(int x, int y, const vector<Mat>& dog_pyr, int index, const double* offset_x)
{
	//|D(x^)|=D + 0.5 * dx * offset_x; dx=(Dx, Dy, Ds)^T
	double dx[3];
	DerivativeOf3D(x, y, dog_pyr, index, dx);

	double term = 0.0;
	for (int i = 0; i < 3; i++)
		term += dx[i] * offset_x[i];

	pixel_t *data = (pixel_t *)dog_pyr[index].data;
	int step = dog_pyr[index].step / sizeof(data[0]);
	pixel_t val = *(data + y * step + x);

	return fabs(val + 0.5 * term);
}

//修正极值点，删除不稳定点
// |D(x)| < 0.03 Lowe 2004
Keypoint* InterploationExtremum(int x, int y, const vector<Mat>& dog_pyr, int index, int octave, int interval, double dxthreshold = DXTHRESHOLD)
{
	//计算x=(x,y,sigma)^T
	//x^ = -H^(-1) * dx; dx = (Dx, Dy, Ds)^T
	double offset_x[3] = { 0 };
	//
	const Mat &mat = dog_pyr[index];
	int idx = index;
	int intvl = interval;
	int i = 0;
	while (i < MAX_INTERPOLATION_STEPS)
	{
		GetOffsetX(x, y, dog_pyr, idx, offset_x);
		//4. Accurate keypoint localization.  Lowe
		//
		//如果offset_x 的任一维度大于0.5，it means that the extremum lies closer to a different sample point.
		if (fabs(offset_x[0]) < 0.5 && fabs(offset_x[1]) < 0.5 && fabs(offset_x[2]) < 0.5)
			break;

		//用周围的点代替
		//
		x += cvRound(offset_x[0]);
		y += cvRound(offset_x[1]);
		interval += cvRound(offset_x[2]);

		idx = index - intvl + interval;
		//		idx = octave*(INTERVALS+2)+interval;

		if (interval < 1 || interval > INTERVALS ||
			x >= mat.cols - 1 || x < 2 ||
			y >= mat.rows - 1 || y < 2)  //此处保证检测边时 x+1,y+1和x-1, y-1有效
		{
			return NULL;
		}

		i++;
	}

	//窜改失败
	if (i >= MAX_INTERPOLATION_STEPS)
		return NULL;

	//rejecting unstable extrema
	//|D(x^)| < 0.03取经验值
	if (GetFabsDx(x, y, dog_pyr, idx, offset_x) < dxthreshold / INTERVALS)
	{
		return NULL;
	}

	Keypoint *keypoint = new Keypoint;


	keypoint->x = x;
	keypoint->y = y;

	keypoint->offset_x = offset_x[0];
	keypoint->offset_y = offset_x[1];

	keypoint->interval = interval;
	keypoint->offset_interval = offset_x[2];

	keypoint->octave = octave;




	keypoint->dx = (x + offset_x[0])*pow(2.0, octave);
	keypoint->dy = (y + offset_x[1])*pow(2.0, octave);

	return keypoint;
}

//检测当地极值点
void DetectionLocalExtrema(const vector<Mat>& dog_pyr, vector<Keypoint>& extrema, int octaves, int intervals = INTERVALS)
{
	long int dd = 0, cc1 = 0, cc2 = 0, cc3 = 0, cc0 = 0, cc00 = 0;

	double thresh = 0.5 * DXTHRESHOLD / intervals;
	for (int o = 0; o < octaves; o++)
	{
		//第一层和最后一层极值忽略
		for (int i = 1; i < (intervals + 2) - 1; i++)
		{
			int index = o * (intervals + 2) + i;
			pixel_t *data = (pixel_t *)dog_pyr[index].data;
			int step = dog_pyr[index].step / sizeof(data[0]);


			for (int y = 1; y < dog_pyr[index].rows - 2; y++)
			{
				for (int x = 1; x < dog_pyr[index].cols - 2; x++)
				{
					cc00++;
					//
					pixel_t val = *(data + y * step + x);
					if (fabs(val) > thresh) //排除阈值过小的点
					{
						cc0++;
						if (isExtremum(x, y, dog_pyr, index))
						{
							cc1++;
							Keypoint *extrmum = InterploationExtremum(x, y, dog_pyr, index, o, i);

							if (extrmum)
							{
								cc2++;

								if (passEdgeResponse(extrmum->x, extrmum->y, dog_pyr, index))
								{
									extrmum->val = *(data + extrmum->y*step + extrmum->x);
									cc3++;
									extrema.push_back(*extrmum);
								}

								delete extrmum;
							}
						}
					}
				}
			}

		}
	}
	cout << "-- " << "cc00: " << cc00 << ", cc0: " << cc0 << ", cc1: " << cc1 << ", cc2: " << cc2 << ", cc3: " << cc3 << " " << thresh << " --" << endl;
}

void CalculateScale(vector<Keypoint>& features, double sigma = SIGMA, int intervals = INTERVALS)
{
	double intvl = 0;
	for (int i = 0; i < features.size(); i++)
	{
		intvl = features[i].interval + features[i].offset_interval;
		features[i].scale = sigma * pow(2.0, features[i].octave + intvl / intervals);
		features[i].octave_scale = sigma * pow(2.0, intvl / intervals);
	}

}

//对扩大的图像特征缩放
void HalfFeatures(vector<Keypoint>& features)
{
	for (int i = 0; i < features.size(); i++)
	{
		features[i].dx /= 2;
		features[i].dy /= 2;

		features[i].scale /= 2;
	}
}

bool CalcGradMagOri(const Mat& gauss, int x, int y, double& mag, double& ori)
{
	if (x > 0 && x < gauss.cols - 1 && y > 0 && y < gauss.rows - 1)
	{
		pixel_t *data = (pixel_t*)gauss.data;
		int step = gauss.step / sizeof(*data);

		double dx = *(data + step * y + (x + 1)) - (*(data + step * y + (x - 1)));
		double dy = *(data + step * (y + 1) + x) - (*(data + step * (y - 1) + x));

		mag = sqrt(dx*dx + dy * dy);

		//atan2返回[-Pi, -Pi]的弧度值
		ori = atan2(dy, dx);
		return true;
	}
	else
		return false;
}
//每10度为一个方向，这个10度在哪里体现的，没找到。。。
double* CalculateOrientationHistogram(const Mat& gauss, int x, int y, int bins, int radius, double sigma)
{
	double *hist = new double[bins];

	for (int i = 0; i < bins; i++)
		*(hist + i) = 0.0;

	double mag, ori;

	double weight;

	int bin;
	const double PI2 = 2.0*CV_PI;

	double econs = -1.0 / (2.0*sigma*sigma);

	for (int i = -radius; i <= radius; i++)
	{
		for (int j = -radius; j <= radius; j++)
		{
			if (CalcGradMagOri(gauss, x + i, y + j, mag, ori))
			{
				weight = exp((i*i + j * j)*econs);

				//使用Pi-ori将ori转换到[0,2*PI]之间
				bin = cvRound(bins * (CV_PI - ori) / PI2);
				bin = bin < bins ? bin : 0;

				hist[bin] += mag * weight;
			}
		}
	}

	return hist;
}

//高斯平滑，模板为{0.25, 0.5, 0.25}
void GaussSmoothOriHist(double *hist, int n)
{
	double prev = hist[n - 1], temp, h0 = hist[0];


	for (int i = 0; i < n; i++)
	{
		temp = hist[i];
		hist[i] = 0.25 * prev + 0.5 * hist[i] +
			0.25 * (i + 1 >= n ? h0 : hist[i + 1]);
		prev = temp;
	}
}

//计算方向直方图中的主方向
double DominantDirection(double *hist, int n)
{
	double maxd = hist[0];
	for (int i = 1; i < n; i++)
	{
		if (hist[i] > maxd)
			maxd = hist[i];
	}
	return maxd;
}


void CopyKeypoint(const Keypoint& src, Keypoint& dst)
{
	dst.dx = src.dx;
	dst.dy = src.dy;

	dst.interval = src.interval;
	dst.octave = src.octave;
	dst.octave_scale = src.octave_scale;
	dst.offset_interval = src.offset_interval;

	dst.offset_x = src.offset_x;
	dst.offset_y = src.offset_y;

	dst.ori = src.ori;
	dst.scale = src.scale;
	dst.val = src.val;
	dst.x = src.x;
	dst.y = src.y;
}

//抛物插值
#define Parabola_Interpolate(l, c, r) (0.5*((l)-(r))/((l)-2.0*(c)+(r))) 


void CalcOriFeatures(const Keypoint& keypoint, vector<Keypoint>& features, const double *hist, int n, double mag_thr)
{
	double bin, PI2 = CV_PI * 2.0;
	int l, r;
	for (int i = 0; i < n; i++)
	{
		l = (i == 0) ? n - 1 : i - 1;
		r = (i + 1) % n;

		//hist[i]是极值
		if (hist[i] > hist[l] && hist[i] > hist[r] && hist[i] >= mag_thr)
		{
			bin = i + Parabola_Interpolate(hist[l], hist[i], hist[r]);
			bin = (bin < 0) ? (bin + n) : (bin >= n ? (bin - n) : bin);
			Keypoint new_key;
			CopyKeypoint(keypoint, new_key);
			new_key.ori = ((PI2 * bin) / n) - CV_PI;
			features.push_back(new_key);
		}
	}
}

//关键点方向分配
void OrientationAssignment(vector<Keypoint>& extrema, vector<Keypoint>& features, const vector<Mat>& gauss_pyr)
{
	int n = extrema.size();
	double *hist;

	for (int i = 0; i < n; i++)
	{

		hist = CalculateOrientationHistogram(gauss_pyr[extrema[i].octave*(INTERVALS + 3) + extrema[i].interval],
			extrema[i].x, extrema[i].y, ORI_HIST_BINS, cvRound(ORI_WINDOW_RADIUS*extrema[i].octave_scale),
			ORI_SIGMA_TIMES*extrema[i].octave_scale);

		for (int j = 0; j < ORI_SMOOTH_TIMES; j++)
			GaussSmoothOriHist(hist, ORI_HIST_BINS);
		double highest_peak = DominantDirection(hist, ORI_HIST_BINS);

		CalcOriFeatures(extrema[i], features, hist, ORI_HIST_BINS, highest_peak*ORI_PEAK_RATIO);

		delete[] hist;

	}
}

void InterpHistEntry(double ***hist, double xbin, double ybin, double obin, double mag, int bins, int d)
{
	double d_r, d_c, d_o, v_r, v_c, v_o;
	double** row, *h;
	int r0, c0, o0, rb, cb, ob, r, c, o;

	r0 = cvFloor(ybin);
	c0 = cvFloor(xbin);
	o0 = cvFloor(obin);
	d_r = ybin - r0;
	d_c = xbin - c0;
	d_o = obin - o0;

	/*
	做插值：
	xbin,ybin,obin:种子点所在子窗口的位置和方向
	所有种子点都将落在4*4的窗口中
	r0,c0取不大于xbin，ybin的正整数
	r0,c0只能取到0,1,2
	xbin,ybin在(-1, 2)

	r0取不大于xbin的正整数时。
	r0+0 <= xbin <= r0+1
	mag在区间[r0,r1]上做插值

	obin同理
	*/

	for (r = 0; r <= 1; r++)
	{
		rb = r0 + r;
		if (rb >= 0 && rb < d)
		{
			v_r = mag * ((r == 0) ? 1.0 - d_r : d_r);
			row = hist[rb];
			for (c = 0; c <= 1; c++)
			{
				cb = c0 + c;
				if (cb >= 0 && cb < d)
				{
					v_c = v_r * ((c == 0) ? 1.0 - d_c : d_c);
					h = row[cb];
					for (o = 0; o <= 1; o++)
					{
						ob = (o0 + o) % bins;
						v_o = v_c * ((o == 0) ? 1.0 - d_o : d_o);
						h[ob] += v_o;
					}
				}
			}
		}
	}


}

double*** CalculateDescrHist(const Mat& gauss, int x, int y, double octave_scale, double ori, int bins, int width)
{
	double ***hist = new double**[width];

	//申请空间并初始化
	for (int i = 0; i < width; i++)
	{
		hist[i] = new double*[width];
		for (int j = 0; j < width; j++)
		{
			hist[i][j] = new double[bins];
		}
	}

	for (int r = 0; r < width; r++)
		for (int c = 0; c < width; c++)
			for (int o = 0; o < bins; o++)
				hist[r][c][o] = 0.0;


	double cos_ori = cos(ori);
	double sin_ori = sin(ori);

	//6.1高斯权值，sigma等于描述字窗口宽度的一半
	double sigma = 0.5 * width;
	double conste = -1.0 / (2 * sigma*sigma);

	double PI2 = CV_PI * 2;

	double sub_hist_width = DESCR_SCALE_ADJUST * octave_scale;

	//领域半径
	int radius = (sub_hist_width*sqrt(2.0)*(width + 1)) / 2.0 + 0.5; //+0.5取四舍五入

	double grad_ori, grad_mag;
	for (int i = -radius; i <= radius; i++)
	{
		for (int j = -radius; j <= radius; j++)
		{
			double rot_x = (cos_ori * j - sin_ori * i) / sub_hist_width;
			double rot_y = (sin_ori * j + cos_ori * i) / sub_hist_width;


			//xbin,ybin为落在4*4窗口中的下标值
			double xbin = rot_x + width / 2 - 0.5;
			double ybin = rot_y + width / 2 - 0.5;

			//
			if (xbin > -1.0 && xbin < width && ybin > -1.0 && ybin < width)
			{
				if (CalcGradMagOri(gauss, x + j, y + i, grad_mag, grad_ori))
				{
					grad_ori = (CV_PI - grad_ori) - ori;
					while (grad_ori < 0.0)
						grad_ori += PI2;
					while (grad_ori >= PI2)
						grad_ori -= PI2;

					double obin = grad_ori * (bins / PI2);

					double weight = exp(conste*(rot_x*rot_x + rot_y * rot_y));

					InterpHistEntry(hist, xbin, ybin, obin, grad_mag*weight, bins, width);

				}
			}
		}
	}

	return hist;
}

void NormalizeDescr(Keypoint& feat)
{
	double cur, len_inv, len_sq = 0.0;
	int i, d = feat.descr_length;

	for (i = 0; i < d; i++)
	{
		cur = feat.descriptor[i];
		len_sq += cur * cur;
	}
	len_inv = 1.0 / sqrt(len_sq);
	for (i = 0; i < d; i++)
		feat.descriptor[i] *= len_inv;
}

void HistToDescriptor(double ***hist, int width, int bins, Keypoint& feature)
{
	int int_val, i, r, c, o, k = 0;

	for (r = 0; r < width; r++)
		for (c = 0; c < width; c++)
			for (o = 0; o < bins; o++)
			{
				feature.descriptor[k++] = hist[r][c][o];
			}

	feature.descr_length = k;
	NormalizeDescr(feature);
	for (i = 0; i < k; i++)
		if (feature.descriptor[i] > DESCR_MAG_THR)
			feature.descriptor[i] = DESCR_MAG_THR;
	NormalizeDescr(feature);

	/* convert floating-point descriptor to integer valued descriptor */
	for (i = 0; i < k; i++)
	{
		int_val = INT_DESCR_FCTR * feature.descriptor[i];
		feature.descriptor[i] = min(255, int_val);
	}
}

//计算描述符
void DescriptorRepresentation(vector<Keypoint>& features, const vector<Mat>& gauss_pyr, int bins, int width)
{
	double ***hist;

	for (int i = 0; i < features.size(); i++)
	{
		hist = CalculateDescrHist(gauss_pyr[features[i].octave*(INTERVALS + 3) + features[i].interval],
			features[i].x, features[i].y, features[i].octave_scale, features[i].ori, bins, width);


		HistToDescriptor(hist, width, bins, features[i]);

		for (int j = 0; j < width; j++)
		{

			for (int k = 0; k < width; k++)
			{
				delete[] hist[j][k];
			}
			delete[] hist[j];
		}
		delete[] hist;
	}
}

bool FeatureCmp(Keypoint& f1, Keypoint& f2)
{
	return f1.scale < f2.scale;
}

//sift 算法
void Sift(const Mat &src, vector<Keypoint>& features, double sigma, int intervals=INTERVALS)
{
	Mat init_gray;
	CreateInitSmoothGray(src, init_gray, sigma);   //初始图像先将原图像灰度化，再扩大一倍后，使用高斯模糊平滑
												   //计算组数
												   //int octaves = log((double)min(init_gray.rows, init_gray.cols))/log(2.0) - 3;
	int octaves = log((double)min(init_gray.rows, init_gray.cols)) / log(2.0) - 2;
	cout << "rows=" << init_gray.rows << " cols=" << init_gray.cols << " octaves=" << octaves << endl;  //输出行数，列数，组数


	cout << "building gaussian pyramid ..." << endl;
	vector<Mat> gauss_pyr;
	GaussianPyramid(init_gray, gauss_pyr, octaves, intervals, sigma);

	//	write_pyr(gauss_pyr, "gausspyramid");

	cout << "building difference of gaussian pyramid..." << endl;
	vector<Mat> dog_pyr;
	DogPyramid(gauss_pyr, dog_pyr, octaves, intervals);

	//	write_pyr(dog_pyr, "dogpyramid");

	cout << "deatecting local extrema..." << endl;

	vector<Keypoint> extrema;
	DetectionLocalExtrema(dog_pyr, extrema, octaves, intervals);

	cout << "--keypoints cout: " << extrema.size() << " --" << endl;

	cout << "extrema detection finished." << endl << "--please look dir gausspyramid, dogpyramid and extrema.txt.--" << endl;
	//计算尺度
	CalculateScale(extrema, sigma, intervals);

	HalfFeatures(extrema);

	cout << "orientation assignment..." << endl;
	OrientationAssignment(extrema, features, gauss_pyr);
	cout << "--features count: " << features.size() << " --" << endl;

	DescriptorRepresentation(features, gauss_pyr, DESCR_HIST_BINS, DESCR_WINDOW_WIDTH);

	sort(features.begin(), features.end(), FeatureCmp);

	cout << "finished." << endl;
}

#endif
