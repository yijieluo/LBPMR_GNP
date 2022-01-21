#ifndef OPERATOR_H
#define OPERATOR_H

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

typedef vector<vector<Mat>> Data;

class Operator{
public:
	virtual void process(const Mat& src, Mat& dst) = 0;
	virtual int getPN() const = 0;//get Pattern Number
	virtual bool preTrain(const Data& imgs){return false;}
};

#endif
