#include <fstream>
#include <cstdlib>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <io.h>
#include <string>

using namespace std;

int GetDesignateFilesNumber(char *fileNamePath)
{
	int fileCounter = 0;  //用来统计特定文件个数的文件计数器；

	_finddata_t fileInfo;

	long long handle = _findfirst(fileNamePath, &fileInfo);

	if (handle == -1L)
	{
		cerr << "failed to transfer files" << endl;
		return false;
	}

	cout << "正在处理中，请稍后........" << endl;
	do
	{
		fileCounter++;  //计数器加1;
		//cout << fileInfo.name << endl;  //输出文件名；

	} while (_findnext(handle, &fileInfo) == 0);
	return fileCounter;
}

//void main()
string generateImageTxt()
{
	int fileCounter; //用来统计特定文件个数的文件计数器；			
	char * fileNamePath = (char*)".\\images\\Walking2\\img\\*.jpg";	 //需要统计特定的文件类型，例如jpg文件;

	fileCounter = GetDesignateFilesNumber(fileNamePath);		//调用函数GetDesignateFilesNumber，统计该文件夹中指定的文件个数,输出统计结果
	cout << "该文件夹中，jpg文件个数为：" << fileCounter << endl;

	string result_name = "images.txt";        
	ofstream result("images.txt");
	if (!result)
		cout << "error!" << endl;

	int img_num = fileCounter;
	for (int i = 1; i <= img_num; i++)
	{
		char img_name[80];
		result << ".\\images\\Walking2\\img\\";
		sprintf(img_name, "000%d.jpg\n", i);
		string tmp_name = img_name;
		strcpy(img_name,tmp_name.substr(tmp_name.length() - 9).c_str());
		//cout << "img_name:" << img_name << endl;
		result << img_name;
	}

	string a = fileNamePath;	//指定要替换的主串
	string b = "*.jpg";	//指定要替换的子串
	string c = "";	//要替换的内容
	int pos = a.find(b);
	if (pos != -1)
	{
		a.replace(a.find(b), b.length(), c);
	}
	cout <<"被替换的字符串:"<< a.c_str() << endl;
	//system("pause");
	return a;
}
