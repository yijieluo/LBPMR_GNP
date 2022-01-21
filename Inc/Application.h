#ifndef APPLICATION_H
#define APPLICATION_H

#include "AppThread.h"
#include "LBPMR.h"
#include "GNP.h"
#include "LBPVariants.h"
#include "Feature.h"
#include "ReadDatasets.h"
#include "TrainPredict.h"


/*
Experiment#1: ablation study
    param1: dataset should be "TC10", "TC12_0", or "TC12_1"
    param2: op should be "LBPRIU2", "LBPMR", "LBPMRK=0.8", or "LBPMRK=0.8_GNPN=2"
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

#endif
