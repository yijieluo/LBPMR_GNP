#ifndef APPTHREAD_H
#define APPTHREAD_H

#include <memory>
#include <thread>
#include <atomic>
#include <opencv2/opencv.hpp>
#include <vector>

class Operator;

template<typename T>
class DP;

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

typedef std::vector<std::vector<cv::Mat>> Data;

void dpPreTrainThread(uint8_t threadNum, uint8_t threadIndex, \
						Operator* op, \
						const Data& imgs,\
						uint32_t*& histogram);


void appThread1(uint8_t threadNum, uint8_t threadIndex,
			   std::vector<Operator*> op1, Operator* op2,
			   const Data& imgs, cv::Mat &feature);

void appThread2(uint8_t threadNum, uint8_t threadIndex,
			   Operator* op,
			   const Data& imgs, cv::Mat &dst);

#endif