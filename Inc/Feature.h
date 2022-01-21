#ifndef FEATURE_H
#define FEATURE_H

#include <opencv2/opencv.hpp>
#include <math.h>
#include <iostream>
#include <vector>
#include <memory>
#include <numeric>
using namespace cv;

template<typename T>
void matToHist(const Mat& src, uint32_t maxGray, T* pHist)
{
	for (int i = 0; i < src.rows; ++i){
		int const* pPixValue = src.ptr<int>(i);
		for (int j = 0; j < src.cols; ++j){
			pHist[pPixValue[j]]++;
		}
	}
	uint32_t num = src.rows*src.cols;
	for (size_t i = 0; i < maxGray; i++){
		pHist[i] = pHist[i]/num;
	}
}

template<typename T>
void matTo2DHist(const Mat & src0, uint32_t maxGray0, const Mat & src1, uint32_t maxGray1, T* pHist)
{
	int margin = (src1.rows - src0.rows) / 2;

	for (int i = 0; i < src0.rows; ++i)
	{
		int const* pPixValue0 = src0.ptr<int>(i);
		int const* pPixValue1 = src1.ptr<int>(i + margin);

		for (int j = 0; j < src0.cols; ++j)
		{
			pHist[pPixValue1[j + margin] * maxGray0 + pPixValue0[j]]++;
		}
	}
	uint32_t num = src0.rows*src0.cols;
	for (size_t i = 0; i < maxGray0*maxGray1; i++){
		pHist[i] = pHist[i]/num;
	}	
}

template<typename T>
void normalize(T* pData, uint32_t length)
{
	double mean = 0;
	for (size_t i = 0; i < length; i++)
	{
		mean += pData[i];
	}
	mean /= length;

	double accum = 0;

	for (size_t i = 0; i < length; i++)
	{
		accum += pow((pData[i] - mean), 2);
	}
	accum = sqrt(accum / (length - 1));

	for (size_t i = 0; i < length; i++)
	{
		pData[i] = (pData[i] - mean) / accum;
	}
}

inline void matNormalize(Mat& input){
	for (size_t i = 0; i < input.rows; i++){
		float* pData = input.ptr<float>(i);
		normalize(pData, input.cols);
	}
}

#endif