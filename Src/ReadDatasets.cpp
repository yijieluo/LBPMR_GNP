#include "ReadDatasets.h"
#include <fstream>
#include <algorithm>

bool readOuluDataset(const string& path_, const string& subset, LabeledData& data)
{
	string path;
	if (subset == "TC10")
		path = path_ + "/000/";
	else if (subset == "TC12_0")
		path = path_ + "/000/";
	else if (subset == "TC12_1")
		path = path_ + "/001/";
	else
		;

	ifstream classesTxt(path + "classes.txt");
	uint32_t groupedNum;
	classesTxt >> groupedNum;
	classesTxt.close();

	data.TrainData.resize(groupedNum);
	data.PredictData.resize(groupedNum);

	ifstream trainImgTxt(path + "train.txt");
	uint32_t trainImgAllNum; 
	trainImgTxt >> trainImgAllNum;
	data.TrainLabel = Mat(trainImgAllNum, 1, CV_32SC1);

	vector<vector<string>> trainImgGroupedName(groupedNum);

	for (size_t i = 0; i < trainImgAllNum; i++)
	{
		string trainImgName;
		int label;
		trainImgTxt >> trainImgName >> label;
		trainImgGroupedName[label].push_back(trainImgName);
	}
	trainImgTxt.close();

	int t = 0;
	for (size_t i = 0; i < groupedNum; i++)
	{
		uint32_t sizeTemp = trainImgGroupedName[i].size();
		for (size_t j = 0; j < sizeTemp; ++j){
			Mat imgTemp1 = imread(path + "../images/" + trainImgGroupedName[i][j]);
			if(imgTemp1.empty()){
				return false;
			}
			Mat imgTemp2;
			cvtColor(imgTemp1, imgTemp2, COLOR_RGB2GRAY);
			data.TrainData[i].push_back(imgTemp2);	
			data.TrainLabel.ptr<int>(t++)[0] = i;
		}
	}

	/*-------------------------------------------------------------------------------------*/

	ifstream testImgTxt(path + "test.txt");
	uint32_t testImgAllNum;
	testImgTxt >> testImgAllNum;
	data.PredictLabel = Mat(testImgAllNum, 1, CV_32SC1);

	vector<vector<string>> testImgGroupedName(groupedNum);
	for (size_t i = 0; i < testImgAllNum; i++)
	{
		string testImgName;
		int label;
		testImgTxt >> testImgName >> label;
		testImgGroupedName[label].push_back(testImgName);
	}
	testImgTxt.close();

	t = 0;
	for (size_t i = 0; i < groupedNum; i++){
		uint32_t sizeTemp = testImgGroupedName[i].size();
		for (size_t j = 0; j < sizeTemp; ++j){
			Mat imgTemp1 = imread(path + "../images/" + testImgGroupedName[i][j]);
			if(imgTemp1.empty()){
				return false;
			}
			Mat imgTemp2;
			cvtColor(imgTemp1, imgTemp2, COLOR_RGB2GRAY);
			data.PredictData[i].push_back(imgTemp2);	
			data.PredictLabel.ptr<int>(t++)[0] = i;
		}
	}
	return true;
}

bool readCUReTDataset(const string& path, uint8_t N, LabeledData& data)
{
	data.TrainData.resize(61);
	data.TrainLabel = Mat(61*N, 1, CV_32SC1);

	data.PredictData.resize(61);
	data.PredictLabel = Mat(61*(92-N), 1, CV_32SC1);
	
	ifstream classNames(path + "/classes.txt");

	uint32_t imgAllNum = 61 * 92;
	uint32_t trainImgNum = 61 * N, testImgNum = 61 * (92 - N);
	int t1 = 0, t2 = 0;
	for (size_t i = 0; i < 61; i++)
	{
		string className;
		classNames >> className;
		ifstream imgNames(path +"/"+ className + "/images.txt");

		array<string, 92> imgName;
		for (auto& i: imgName)
		{
			imgNames >> i;
		}
		array<int, 92> imgNameIndex;
		int t = 0;
		for (auto &i : imgNameIndex){
			i = t++;
		}
		random_shuffle(imgNameIndex.begin(), imgNameIndex.end());

		for (size_t j = 0; j < N; j++){
			Mat tmp = imread(path + "/" + className + "/" + imgName[imgNameIndex[j]], IMREAD_GRAYSCALE);
			if(tmp.empty()){
				return false;
			}
			data.TrainData[i].push_back(tmp);
			data.TrainLabel.ptr<int>(t1++)[0] = i;
		}

		for (size_t j = N; j < 92; j++){
			Mat tmp = imread(path + "/" + className + "/" + imgName[imgNameIndex[j]], IMREAD_GRAYSCALE);
			if(tmp.empty()){
				return false;
			}
			data.PredictData[i].push_back(tmp);
			data.PredictLabel.ptr<int>(t2++)[0] = i;
		}
		imgNames.close();
	}
	classNames.close();
	return true;
}

bool readUIUCDataset(const string& path, uint8_t N, LabeledData& data)
{
	data.TrainData.resize(25);
	data.TrainLabel = Mat(25*N, 1, CV_32SC1);

	data.PredictData.resize(25);
	data.PredictLabel = Mat(25*(40-N), 1, CV_32SC1);

	ifstream file(path + "/names.txt");

	uint32_t imgAllNum = 25 * 40;
	uint32_t trainImgNum = 25 * N, testImgNum = 25 * (40 - N);
	int t1 = 0, t2 = 0;
	for (size_t i = 0; i < 25; i++){
		array<string, 40> imgName;
		for (auto& i: imgName)
		{
			string groupInfo, arrow, name;
			file >> groupInfo >> arrow >> name;
			i = name;
		}

		array<int, 40> imgNameIndex;
		int t = 0;
		for (auto &i : imgNameIndex)
		{
			i = t++;
		}
		random_shuffle(imgNameIndex.begin(), imgNameIndex.end());

		for (size_t j = 0; j < N; j++){
			Mat tmp = imread(path + "/" + imgName[imgNameIndex[j]], IMREAD_GRAYSCALE);
			if(tmp.empty()){
				return false;
			}
			data.TrainData[i].push_back(tmp);
			data.TrainLabel.ptr<int>(t1++)[0] = i;
		}
		for (size_t j = N; j < 40; j++){
			Mat tmp = imread(path + "/" + imgName[imgNameIndex[j]], IMREAD_GRAYSCALE);
			if(tmp.empty()){
				return false;
			}
			data.PredictData[i].push_back(tmp);
			data.PredictLabel.ptr<int>(t2++)[0] = i;
		}
	}
	file.close();
	return true;
}

bool readKTHTIPS2bDataset(const string& path, uint8_t k, LabeledData& data)
{
	const int classNum = 11, trainImgNum = 108*k, testImgNum = 108*(4-k);
	array<string, classNum> className = {"aluminium_foil", "brown_bread", "corduroy", "cork", "cotton", "cracker", "lettuce_leaf", "linen", "white_bread", "wood", "wool"};
	array<string, 4> classSubName = {"sample_a", "sample_b", "sample_c", "sample_d"};

	data.TrainData.resize(classNum);
	data.TrainLabel = Mat(classNum*trainImgNum, 1, CV_32SC1);

	data.PredictData.resize(classNum);
	data.PredictLabel = Mat(classNum*testImgNum, 1, CV_32SC1);

	int t1 = 0, t2 = 0;
	for (size_t i = 0; i < className.size(); i++)
	{
		for (size_t j = 0; j < classSubName.size(); j++)
		{
			if (j != k)
			{	
				string path_ = path + "/" + className[i] + "/" + classSubName[j];
				ifstream trainImgTxt(path_ + "/name.txt");
				for (size_t t = 0; t < 108; t++)
				{	
					string imgName;
					trainImgTxt >> imgName;
					Mat tmp = imread(path_ + "/" + imgName, IMREAD_GRAYSCALE);
					if (tmp.empty()){
						return false;
					}
					data.TrainData[i].push_back(tmp);
					data.TrainLabel.ptr<int>(t1++)[0] = i;
				}
				trainImgTxt.close();
			}
			else
			{
				string path_ = path + "/" + className[i] + "/" + classSubName[j];
				ifstream testImgTxt(path_ + "/name.txt");
				for (size_t t = 0; t < 108; t++)
				{
					string imgName;
					testImgTxt >> imgName;
					Mat tmp = imread(path_ + "/" + imgName, IMREAD_GRAYSCALE);
					if (tmp.empty()){
						return false;
					}
					data.PredictData[i].push_back(tmp);
					data.PredictLabel.ptr<int>(t2++)[0] = i;

				}
				testImgTxt.close();
			}
		}
	}
	return true;
}


bool readUMDDataset(const string& path, uint8_t N, LabeledData& data)
{
	data.TrainData.resize(25);
	data.TrainLabel = Mat(25*N, 1, CV_32SC1);

	data.PredictData.resize(25);
	data.PredictLabel = Mat(25*(40-N), 1, CV_32SC1);

	int t1 = 0, t2 = 0;

	uint32_t imgAllNum = 25 * 40;
	uint32_t trainImgNum = 25 * N, testImgNum = 25 * (40 - N);
	for (size_t i = 0; i < 25; i++)
	{
		ifstream imgNameTxt(path + "/" +to_string(i+1)+ "/name.txt");
		array<string, 40> imgName;
		for (auto& i : imgName)
		{
			imgNameTxt >> i;
		}

		array<int, 40> imgNameIndex;
		int t = 0;
		for (auto &i : imgNameIndex)
		{
			i = t++;
		}
		random_shuffle(imgNameIndex.begin(), imgNameIndex.end());

		for (size_t j = 0; j < N; j++)
		{
			Mat src = imread(path + "/" + to_string(i+1) + "/" + imgName[imgNameIndex[j]], IMREAD_GRAYSCALE);
			if(src.empty()){
				return false;
			}
			resize(src, src, Size(320, 240));
			data.TrainData[i].push_back(src);
			data.TrainLabel.ptr<int>(t1++)[0] = i;
		}

		for (size_t j = N; j < 40; j++)
		{
			Mat src = imread(path + "/" + to_string(i + 1) + "/" + imgName[imgNameIndex[j]], IMREAD_GRAYSCALE);
			if(src.empty()){
				return false;
			}
			resize(src, src, Size(320, 240));
			data.PredictData[i].push_back(src);
			data.PredictLabel.ptr<int>(t2++)[0] = i;
		}
	}
	return true;
}