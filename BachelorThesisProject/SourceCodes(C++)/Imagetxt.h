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
	int fileCounter = 0;  //����ͳ���ض��ļ��������ļ���������

	_finddata_t fileInfo;

	long long handle = _findfirst(fileNamePath, &fileInfo);

	if (handle == -1L)
	{
		cerr << "failed to transfer files" << endl;
		return false;
	}

	cout << "���ڴ����У����Ժ�........" << endl;
	do
	{
		fileCounter++;  //��������1;
		//cout << fileInfo.name << endl;  //����ļ�����

	} while (_findnext(handle, &fileInfo) == 0);
	return fileCounter;
}

//void main()
string generateImageTxt()
{
	int fileCounter; //����ͳ���ض��ļ��������ļ���������			
	char * fileNamePath = (char*)".\\images\\Walking2\\img\\*.jpg";	 //��Ҫͳ���ض����ļ����ͣ�����jpg�ļ�;

	fileCounter = GetDesignateFilesNumber(fileNamePath);		//���ú���GetDesignateFilesNumber��ͳ�Ƹ��ļ�����ָ�����ļ�����,���ͳ�ƽ��
	cout << "���ļ����У�jpg�ļ�����Ϊ��" << fileCounter << endl;

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

	string a = fileNamePath;	//ָ��Ҫ�滻������
	string b = "*.jpg";	//ָ��Ҫ�滻���Ӵ�
	string c = "";	//Ҫ�滻������
	int pos = a.find(b);
	if (pos != -1)
	{
		a.replace(a.find(b), b.length(), c);
	}
	cout <<"���滻���ַ���:"<< a.c_str() << endl;
	//system("pause");
	return a;
}
