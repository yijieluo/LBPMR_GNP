#ifndef DP_H
#define DP_H

#include "TFDR.h"

typedef vector<vector<Mat>> Data;

extern atomic_uint8_t appAllThreadEndFlag;
extern SpinLock appThreadLock;

//Dominant Pattern
template<typename T>
class DP: public TFDR{
protected:
    double ratio;
    T* op = nullptr;
    uint32_t* histogram = nullptr;
public:
    DP(const LPS& lps, const Data& imgs, double ratio_);
	DP(): TFDR(0){};
    virtual ~DP(){
        if(histogram != nullptr){
            delete [] histogram;
            histogram = nullptr;
        }
        if(op != nullptr){
            delete op;
            op = nullptr;
        }
    }
	virtual void countDP();
};

template<typename T>
DP<T>::DP(const LPS& lps, const Data& imgs, double ratio_): TFDR(0){
    ratio = ratio_;
    if(ratio <=0 || ratio >= 1.0){
        cout << "param error\n";
        return;
    }
	PN = pow(2, lps.getP());
    op = new T(lps, new TFDR(PN));

    vector<thread*> preTrainThread;
    uint8_t threadNum = 16;
    appAllThreadEndFlag = 0;

    for (size_t i = 0; i < threadNum; i++){
        preTrainThread.push_back(new thread(dpPreTrainThread, \
            threadNum, i, \
            op,
            ref(imgs),
            ref(histogram)));
        preTrainThread[i]->detach();
    }

    while (appAllThreadEndFlag != threadNum){
        this_thread::sleep_for(chrono::milliseconds(100));
    }
    for (auto& i : preTrainThread)
        delete i;
    preTrainThread.clear();

    countDP();
}

template<typename T>
void DP<T>::countDP(){
    double* histogramNorm = new double[PN]{0};
	uint64_t histogramAdd = 0;

	for (uint32_t i = 0; i < PN; i++){
		histogramAdd += histogram[i];
	}
	for (uint32_t i = 0; i < PN; i++){
		histogramNorm[i] = histogram[i] / static_cast<double>(histogramAdd);
	}

    std::vector<int32_t> DP; //dominant pattern
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

	table = new int[oldPN]{};
	//uint32_t margin = param->margin;
	for (uint32_t i = 0; i < oldPN; i++)//查表法，此处填表
	{
		uint32_t diff, minDiff = 2 * oldPN;
		table[i] = PN - 1;
		for (uint32_t j = 0; j < DP.size(); j++)
		{
			diff = abs(DP[j] - i);
			if (diff < minDiff)
			{
				minDiff = diff;
				table[i] = j;
				continue;
			}
		}
	}
}

template<typename T>
class DP2: public DP<T>{
	static int m; //m: the number of different scales
	static double ratio2;
	static vector<DP2*> pDP2;
public:
	static void init(int m_, double ratio_){
		if(ratio_ <=0 || ratio_ >= 1.0){
			cout << "param error\n";
			return;
		}
		m = m_;
		ratio2 = ratio_;
	}
    DP2(const LPS& lps, const Data& imgs);
    virtual ~DP2(){}
	virtual void countDP();
};

template<typename T>
int DP2<T>::m = 0;

template<typename T>
double DP2<T>::ratio2 = 0;


template<typename T>
vector<DP2<T>*> DP2<T>::pDP2(0);

template<typename T>
DP2<T>::DP2(const LPS& lps, const Data& imgs){

	pDP2.push_back(this);

	TFDR::PN = pow(2, lps.getP());
    DP<T>::op = new T(lps, new TFDR(TFDR::PN));

    vector<thread*> preTrainThread;
    uint8_t threadNum = 16;
    appAllThreadEndFlag = 0;

    for (size_t i = 0; i < threadNum; i++){
        preTrainThread.push_back(new thread(dpPreTrainThread, \
            threadNum, i, \
            DP<T>::op,
            ref(imgs),
            ref(DP<T>::histogram)));
        preTrainThread[i]->detach();
    }

    while (appAllThreadEndFlag != threadNum){
        this_thread::sleep_for(chrono::milliseconds(100));
    }
    for (auto& i : preTrainThread)
        delete i;
    preTrainThread.clear();

    countDP();
}

template<typename T>
void DP2<T>::countDP(){
	if(pDP2.size() != m){
		return;
	}
	vector<double*> histogramNorm;

	for (size_t i = 0; i < pDP2.size(); i++){
		histogramNorm.push_back(new double[pDP2[i]->PN]);
		uint64_t histogramAdd = 0;

		for (uint32_t j = 0; j < pDP2[i]->PN; j++) {
			histogramAdd += pDP2[i]->histogram[j];
		}
		for (uint32_t j = 0; j < pDP2[i]->PN; j++) {
			histogramNorm[i][j] = pDP2[i]->histogram[j] / static_cast<double>(histogramAdd);
		}
	}

	vector<vector<int32_t>> DP_(m);
	//DP.reserve(PN);
	double ratio_ = ratio2 * m;

	double ratioAdd = 0;
	double max;
	int maxIndex1, maxIndex2;
	while(1)//外层循环累加概率 判断是否大于给定ratio，然后跳出循环
	{
		max = 0;
		for (size_t t = 0; t < m; t++)
		{
			for (uint32_t j = 0; j < pDP2[t]->PN; j++)//内层循环，依次找到最大概率，第二大的概率
			{
				if (max < histogramNorm[t][j])
				{
					max = histogramNorm[t][j];
					maxIndex1 = t;
					maxIndex2 = j;
				}
			}
		}
		DP_[maxIndex1].push_back(maxIndex2);
		ratioAdd += max;
		histogramNorm[maxIndex1][maxIndex2] = 0;//清零，下次循环还是找最大值
		if (ratioAdd >= ratio_)
			break;
	}
	for (auto& i : histogramNorm)
		delete[] i;

	for (auto& i : DP_){
		if (i.size() <= 1)
			cerr << "ratio is too small\n";
	}

	for (size_t t = 0; t < m; t++)
	{
		int oldPN = pDP2[t]->PN;
		pDP2[t]->PN = DP_[t].size();//update the value of PN

		std::sort(DP_[t].begin(), DP_[t].end(), std::less<int>());

		pDP2[t]->table = new int32_t[oldPN]{};

		for (uint32_t i = 0; i < oldPN; i++)//查表法，此处填表
		{
			uint32_t diff, minDiff = 2 * oldPN;
			pDP2[t]->table[i] = pDP2[t]->PN - 1;
			for (uint32_t j = 0; j < DP_[t].size(); j++)
			{
				diff = abs(DP_[t][j] - i);
				if (diff < minDiff)
				{
					minDiff = diff;
					pDP2[t]->table[i] = j;
					continue;
				}
			}
		}
	}
	pDP2.clear();
}

#endif