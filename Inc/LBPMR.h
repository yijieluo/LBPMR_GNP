#ifndef LBPMR_H
#define LBPMR_H

#include "Operator.h"
#include "LBPVariants.h"
#include <atomic>
#include <thread>

class SpinLock{
	std::atomic<bool> flag_;
public:
	SpinLock() : flag_(false){}
	void lock(){
		bool expect = false;
		while (!flag_.compare_exchange_weak(expect, true)){
			expect = false;
		}
	}
	void unlock(){
		flag_.store(false);
	}
};

inline void getBinWeight(uint32_t *diff, uint32_t *binWeight, uint8_t len){
	memset(binWeight, 0, sizeof(uint32_t)*len);
	for (int i = 0; i < len-1; ++i){
		for (int j = i + 1; j < len; ++j){
			if (diff[i] != diff[j])
				diff[i] > diff[j] ? ++binWeight[i] : ++binWeight[j];
		}
	}
}

class SharedMemory{
public:
    SpinLock lock;
    vector<int32_t*> histogram;
    vector<int32_t*> lookuptable;
    ~SharedMemory(){
        for(auto& i: histogram){
            if(i != nullptr){
                delete[] i;
                i = nullptr;
            }
        }
        for(auto& i: lookuptable){
            if(i != nullptr){
                delete[] i;
                i = nullptr;
            }
        }
    }
};

class LBPMR : public Operator{
    const LocalPixels lp;
    double ratio;
    int32_t* table = nullptr;
    int PN;
    int ID;
public:
    static SharedMemory sm;
	LBPMR(LOCAL_SAMPLE_SHAPE shape_, LOCAL_SAMPLE_RP rp_, double ratio_ = 1);
	~LBPMR(){
        if(table != nullptr){
            delete[] table;
            table = nullptr;
        }
    }
	virtual void process(const Mat& src, Mat& dst);
	virtual int getPN() const{
        return PN;
    }
	virtual bool preTrain(const Data& imgs);
    void countHistogram(const Mat& img);
    void countDP();
};
#endif