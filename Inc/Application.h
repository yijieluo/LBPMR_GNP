#ifndef APPLICATION_H
#define APPLICATION_H

#include "AppThread.h"
#include "LBPVariants.h"
#include "Feature.h"
#include "ReadDatasets.h"
#include "TrainPredict.h"


/*
Experiment#1: ablation study
    param1: dataset should be "TC10", "TC12_0", or "TC12_1"
    param2: op should be "LBPRIU2", "LBPMR", "LBPMRK=0.8", or "LBPMRK=0.8/GNPN=2"
    param3: sch should be "SCH1", "SCH2", "SCH3", "SCH4", or "SCH5"
*/
void experiment1(const string& dataset, const string& op, const string& sch);

/*
Experiment#2: prection accuracy of LBPMRK=0.8_GNPN=2 on different datasets
    param1: dataset should be "TC10", "TC12_0", "TC12_1"， "KTH-TIPS2-b", "CUReT", "UIUC" or "UMD"
*/
void experiment2(const string& dataset);

/*
Experiment#3: feature extraction time comparsion
*/
void experiment3();


/*
Experiment#0: examples of how to combine different operators with differen dimension reduction methods
*/
void experiment0();


/*
H(sp) vs H(mp), Quantitative comparison based on information entropy
*/
void experiment0_1();


/*
A example of the fixed weight problem
A example of discarding absolute gray level problem
*/
void experiment0_2();

/*
The feature image comparison: LBPmr vs LBP
*/
void experiment0_3();


/*
U value comparison under different scale combination schemes
*/
void experiment0_4();


/*
Effect of extracting the dominant pattern of LBPmr feature image(K=0.1)
*/
void experiment0_5();


/*
Number of dominant patterns in LBPmr feature images at different values of K
*/
void experiment0_6();

#endif
