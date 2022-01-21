#ifndef APPTHREAD_H
#define APPTHREAD_H

#include "Operator.h"
#include "LBPMR.h"

#include "ReadDatasets.h"
#include <memory>
#include <thread>

void lbpmrPreTrainThread(uint8_t threadNum, uint8_t threadIndex,
						vector<LBPMR*> lbpmr,
						const Data& img);


void appThread1(uint8_t threadNum, uint8_t threadIndex,
			   vector<Operator*> op1, Operator* op2,
			   const Data& imgs, Mat &feature);


#endif