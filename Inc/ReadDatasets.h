#ifndef READDATASETS_H
#define READDATASETS_H

#include <iostream>
#include <vector>
#include <array>
#include <utility>
#include <memory>

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

typedef vector<vector<Mat>> Data;

class LabeledData{
public:
	Mat TrainLabel;
	Data TrainData;

	Mat PredictLabel;
	Data PredictData;
};

bool readOuluDataset(const string& path_, const string& subset, LabeledData& data);
bool readCUReTDataset(const string& path, uint8_t N, LabeledData& data);
bool readUIUCDataset(const string& path, uint8_t N, LabeledData& data);
bool readKTHTIPS2bDataset(const string& path, uint8_t k, LabeledData& data);
bool readUMDDataset(const string& path, uint8_t N, LabeledData& data);

#endif