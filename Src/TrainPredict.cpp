#include "TrainPredict.h"

using namespace cv::ml;

uint8_t svmTraining(const cv::Mat & featureMat, const cv::Mat & labelMat)
{
	cv::Ptr<SVM> svm = SVM::create();

	svm->setType(SVM::C_SVC);
	svm->setKernel(SVM::RBF);
	svm->setGamma(0.01);//0.01
	svm->setC(1000);
	svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::EPS, 1000, FLT_EPSILON));
	svm->train(TrainData::create(featureMat, ROW_SAMPLE, labelMat));
	svm->save("./svmModel.xml");

	return 0;
}

uint8_t knnTraining(const cv::Mat & featureMat, const cv::Mat & labelMat)
{
	cv::Ptr<KNearest> knn = KNearest::create();
	cv::Ptr<TrainData> trainData;
	trainData = TrainData::create(featureMat, SampleTypes::ROW_SAMPLE, labelMat);
	knn->setIsClassifier(true);
	knn->setAlgorithmType(KNearest::Types::BRUTE_FORCE);
	knn->setDefaultK(1);
	knn->train(trainData);
	knn->save("./knnModel.txt");
	return 0;
}

double svmPrediction(const cv::Mat& featureMat, const cv::Mat & labelMat)
{
	auto svm = cv::Algorithm::load<SVM>("./svmModel.xml");

	uint32_t q0 = 0, q1 = 0;
	for (size_t i = 0; i < featureMat.rows; i++)
	{
		cv::Mat sampleMat = featureMat.row(i).clone();
		uint8_t result = round(svm->predict(sampleMat));

		if (static_cast<bool>(result == labelMat.at<int>(i,0)))
		{
			q0++;
		}
		q1++;
	}
	double result = 100.0*q0 / q1;
	printf("\nPrediction Accuracy: %.2f%%\n", result);

	return result;
}
double knnPrediction(const cv::Mat& featureMat, const cv::Mat& labelMat)
{
	cv::Ptr<KNearest> knn = StatModel::load<KNearest>("./knnModel.txt");

	uint32_t q0 = 0, q1 = 0;
	for (size_t i = 0; i < featureMat.rows; i++)
	{
		cv::Mat sampleMat = featureMat.row(i).clone();

		uint8_t result = round(knn->predict(sampleMat));

		if (static_cast<bool>(result == labelMat.at<int>(i, 0)))
		{
			q0++;
		}
		q1++;
	}
	double result = 100.0*q0 / q1;
	printf("\nPrediction Accuracy: %.2f%%\n", result);

	return result;
}


