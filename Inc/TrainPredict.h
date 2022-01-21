#ifndef TRAINPREDICT_H
#define TRAINPREDICT_H
#include <opencv2/opencv.hpp>

uint8_t svmTraining(const cv::Mat & featureMat, const cv::Mat & labelMat);
uint8_t knnTraining(const cv::Mat & featureMat, const cv::Mat & labelMat);

double svmPrediction(const cv::Mat&, const cv::Mat&);
double knnPrediction(const cv::Mat&, const cv::Mat&);
#endif
