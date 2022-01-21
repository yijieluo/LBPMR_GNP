#include "LBPMR.h"

SharedMemory LBPMR::sm;

LBPMR::LBPMR(LOCAL_SAMPLE_SHAPE shape_, LOCAL_SAMPLE_RP rp_, double ratio_):lp(shape_, rp_), ratio(ratio_){
    PN = pow(2, lp.getP());

    table = new int32_t[PN];
    for (size_t i = 0; i < PN; i++){
        table[i] = i;
    }    
    sm.histogram.push_back(nullptr);
    sm.lookuptable.push_back(nullptr);
    ID = sm.histogram.size()-1;
}

void LBPMR::process(const Mat& src, Mat& dst){
    const int r = lp.getR();
    const int p = lp.getP();

    uint32_t *lbm = new uint32_t[p];
    uint32_t *lbw = new uint32_t[p];
    bool *lbs = new bool[p];
    dst = Mat(src.rows - 2 * r, src.cols - 2 * r, CV_32SC1);
    int p1 = 0, p2 = p/4, p3 = p/2, p4 = 3*p/4;
    for (size_t i = r; i < src.rows - r; ++i){
        uint8_t const * pSrcRow  = src.ptr<uint8_t>(i);
        int* pDstRow  = dst.ptr<int>(i-r);
        for (size_t j = r; j < src.cols - r; ++j){
            uint8_t c = pSrcRow[j];
            uint32_t lbpmr = 0;
            for(size_t t = 0; t < p; t++){
                int tmp;
                if(t == p1 || t == p2 || t == p3 || t == p4){
                    tmp = src.ptr<uint8_t>(i + lp.getRY()[t])[int(j+lp.getRX()[t])] - c;
                }else{
                    tmp = round(biLinearInterpolation(src, Point2d(j+lp.getRX()[t], i+lp.getRY()[t])) - c);
                }
                lbm[t] = abs(tmp);
                lbs[t] = (tmp > 0) ? 1 : 0;
            }
            getBinWeight(lbm, lbw, p);
            for (size_t t = 0; t < p; ++t){
                lbpmr += (lbs[t] << lbw[t]); //+= cannot be replaced by |=
            }
            pDstRow[j-r] = table[lbpmr];
        }
    }
    delete[] lbm;
    delete[] lbw;
    delete[] lbs;
}

extern atomic_uint8_t appAllThreadEndFlag;
extern void lbpmrPreTrainThread(uint8_t threadNum, uint8_t threadIndex,
						LBPMR* lbpmr, 
						const Data& img);

bool LBPMR::preTrain(const Data& imgs){
    if(!(ratio > 0 && ratio < 1)){
        return false;
    }
    vector<thread*> preTrainThread;
    uint8_t threadNum = 16;
    appAllThreadEndFlag = 0;

    for (size_t i = 0; i < threadNum; i++){
		preTrainThread.push_back(new thread(lbpmrPreTrainThread, \
			threadNum, i, \
			this, \
			ref(imgs)));
		preTrainThread[i]->detach();
	}

    while (appAllThreadEndFlag != threadNum){
		this_thread::sleep_for(chrono::milliseconds(100));
	}
	for (auto& i : preTrainThread)
		delete i;
	preTrainThread.clear();

    countDP();

    return true;
}

void LBPMR::countHistogram(const Mat& img){
    if(!(ratio > 0 && ratio < 1.0)){
        return;
    }
    Mat tmp;
    process(img, tmp);
    sm.lock.lock();
    if(sm.histogram[ID] == nullptr){
        sm.histogram[ID] = new int32_t[PN]{0};
    }
    sm.lock.unlock();

    for (size_t i = 0; i < tmp.rows; i++){
        int* pRow = tmp.ptr<int>(i);
        for (size_t j = 0; j < tmp.cols; j++){
            sm.lock.lock();
            sm.histogram[ID][pRow[j]]++;
            sm.lock.unlock();
        }
    }
}
void LBPMR::countDP(){
    if(!(ratio > 0 && ratio < 1.0)){
        return;
    }

    double* histogramNorm = new double[PN]{0};
	uint64_t histogramAdd = 0;

	for (uint32_t i = 0; i < PN; i++){
		histogramAdd += sm.histogram[ID][i];
	}
	for (uint32_t i = 0; i < PN; i++){
		histogramNorm[i] = sm.histogram[ID][i] / static_cast<double>(histogramAdd);
	}

    std::vector<int32_t> DP; //effective pattern
	DP.reserve(PN);

	double ratioAdd = 0;
	double max;

	for (uint32_t i = 0; i < PN; i++)//外层循环累加概率 判断是否大于给定ratio，然后跳出循环
	{
		DP.push_back(0);
		max = 0;
		for (uint32_t j = 0; j < PN; j++)//内层循环，依次找到最大概率，第二大的概率，，，，不能直接用sort排序，因为排序之后找不到对应的索引值
		{
			if (max < histogramNorm[j])
			{
				DP[i] = j;
				max = histogramNorm[j];
			}
		}
		ratioAdd += histogramNorm[DP[i]];
		histogramNorm[DP[i]] = 0;//清零，下次循环还是找最大值
		if (ratioAdd >= ratio)
			break;
	}
	delete[] histogramNorm;

    int oldPN = PN;
	PN = DP.size();//update the value of PN
	std::sort(DP.begin(), DP.end(), std::less<int>());

	sm.lookuptable[ID] = new int32_t[oldPN]{};
	//uint32_t margin = param->margin;
	for (uint32_t i = 0; i < oldPN; i++)//查表法，此处填表
	{
		uint32_t diff, minDiff = 2 * oldPN;
		sm.lookuptable[ID][i] = PN - 1;
		for (uint32_t j = 0; j < DP.size(); j++)
		{
			diff = abs(DP[j] - i);
			if (diff < minDiff)
			{
				minDiff = diff;
				sm.lookuptable[ID][i] = j;
				continue;
			}
		}
	}
    for (size_t i = 0; i < oldPN; i++){
        table[i] = sm.lookuptable[ID][i];
    }
}