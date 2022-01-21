#include "AppThread.h"
#include "LBPMR.h"
#include "Feature.h"

#include "numeric"

SpinLock appThreadLock;
extern atomic_uint8_t appAllThreadEndFlag;


void lbpmrPreTrainThread(uint8_t threadNum, uint8_t threadIndex,
						LBPMR* lbpmr,
						const Data& imgs)
{
	for (size_t i = 0, index = threadIndex; index < imgs.size(); ++i, index = i * threadNum + threadIndex){
		for (size_t j = 0; j < imgs[index].size(); j++){
			lbpmr->countHistogram(imgs[index][j]);
		}
	}
	appAllThreadEndFlag++;
}
void appThread1(uint8_t threadNum, uint8_t threadIndex,
			   vector<Operator*> op1, Operator* op2,
			   const Data& imgs, Mat& features)
{
	uint32_t fl = 0; //feature length
	uint32_t op2PN;
	if(op2 == nullptr){
		op2PN = 1;
		for (auto &pOp : op1)
			fl += pOp->getPN();
	}else{
		op2PN = op2->getPN();
		for (auto &pOp : op1)
			fl += pOp->getPN()*op2->getPN();
	}
	
	appThreadLock.lock();
	if (features.rows == 0)
	{
		int t = 0;
		for(auto& i: imgs){
			t += i.size();
		}
		features = Mat(t, fl, CV_32FC1);
	}
	appThreadLock.unlock();

	vector<uint32_t> indexBias(imgs.size(), 0);
	for (size_t i = 1; i < imgs.size(); i++){
		indexBias[i] = indexBias[i - 1] + imgs[i - 1].size();
	}

	for (size_t i = 0, index = threadIndex; index < imgs.size(); ++i, index = i * threadNum + threadIndex)
	{
		for (size_t j = 0; j < imgs[index].size(); j++)
		{
			Mat feature = Mat::zeros(1, fl, CV_32FC1);
			float *pFeature = feature.ptr<float>(0);

			int k = 0;
			for (size_t t = 0; t < op1.size(); t++)
			{
				Mat dst_S, dst_C;
				op1[t]->process(imgs[index][j], dst_S);
				t == 0? k = 0 : k += op1[t-1]->getPN() * op2PN;
				if(op2 != nullptr){
					op2->process(imgs[index][j], dst_C);
					matTo2DHist(dst_S, op1[t]->getPN(), dst_C, op2->getPN(), &pFeature[k]);
				}else{
					matToHist(dst_S, op1[t]->getPN(), &pFeature[k]);
				}
			}
			matNormalize(feature);
			Mat temp = features.row(j + indexBias[index]);
			feature.copyTo(temp);
		}
	}
	appAllThreadEndFlag++;
}