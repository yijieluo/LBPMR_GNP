#include "Application.h"
#include "TFDR.h"
#include "DP.h"

extern atomic_uint8_t appAllThreadEndFlag;

void experiment1(const string& dataset, const string& op, const string& sch)
{
	LabeledData data;
    string subset;
    string path;

	vector<Operator*> op1;
    vector<TFDR*> tfdr;
    Operator* op2 = nullptr;

    if(dataset == "TC10"){
        subset = "TC10";
        path = "../Datasets/Outex_TC_00010";
    }else if(dataset == "TC12_0"){
        subset = "TC12_0";
        path = "../Datasets/Outex_TC_00012";
    }else if(dataset == "TC12_1"){
        subset = "TC12_1";
        path = "../Datasets/Outex_TC_00012";
    }else{
        cout << "param error!\n";
        return;
    }

    if(!readOuluDataset(path, subset, data)){
        cout << "Reading images failed\n";
        return;
    }

    vector<LPS> lps;
    if(sch == "SCH1"){
        lps.push_back(LPS(CIRCLE, R1P8));
    }else if(sch == "SCH2"){
        lps.push_back(LPS(CIRCLE, R1P8));
        lps.push_back(LPS(CIRCLE, R2P8));
    }else if(sch == "SCH3"){
        lps.push_back(LPS(CIRCLE, R1P8));
        lps.push_back(LPS(CIRCLE, R2P8));
        lps.push_back(LPS(CIRCLE, R3P8));
    }else if(sch == "SCH4"){
        lps.push_back(LPS(CIRCLE, R1P8));
        lps.push_back(LPS(CIRCLE, R2P8));
        lps.push_back(LPS(CIRCLE, R3P8));
        lps.push_back(LPS(CIRCLE, R4P8));
    }else if(sch == "SCH5"){
        lps.push_back(LPS(CIRCLE, R1P8));
        lps.push_back(LPS(CIRCLE, R2P8));
        lps.push_back(LPS(CIRCLE, R3P8));
        lps.push_back(LPS(CIRCLE, R4P8));
        lps.push_back(LPS(CIRCLE, R5P8));
    }else if(sch == "SCH6"){
        lps.push_back(LPS(CIRCLE, R1P8));
        lps.push_back(LPS(CIRCLE, R2P8));
        lps.push_back(LPS(CIRCLE, R3P8));
        lps.push_back(LPS(CIRCLE, R4P8));
        lps.push_back(LPS(CIRCLE, R5P8));
        lps.push_back(LPS(CIRCLE, R6P8));
    }else if(sch == "SCH7"){
        lps.push_back(LPS(CIRCLE, R1P8));
        lps.push_back(LPS(CIRCLE, R2P8));
        lps.push_back(LPS(CIRCLE, R3P8));
        lps.push_back(LPS(CIRCLE, R4P8));
        lps.push_back(LPS(CIRCLE, R5P8));
        lps.push_back(LPS(CIRCLE, R6P8));
        lps.push_back(LPS(CIRCLE, R7P8));
    }else if(sch == "SCH8"){
        lps.push_back(LPS(CIRCLE, R1P8));
        lps.push_back(LPS(CIRCLE, R2P8));
        lps.push_back(LPS(CIRCLE, R3P8));
        lps.push_back(LPS(CIRCLE, R4P8));
        lps.push_back(LPS(CIRCLE, R5P8));
        lps.push_back(LPS(CIRCLE, R6P8));
        lps.push_back(LPS(CIRCLE, R7P8));
        lps.push_back(LPS(CIRCLE, R8P8));
    }else if(sch == "SCH9"){
        lps.push_back(LPS(CIRCLE, R1P8));
        lps.push_back(LPS(CIRCLE, R2P8));
        lps.push_back(LPS(CIRCLE, R3P8));
        lps.push_back(LPS(CIRCLE, R4P8));
        lps.push_back(LPS(CIRCLE, R5P8));
        lps.push_back(LPS(CIRCLE, R6P8));
        lps.push_back(LPS(CIRCLE, R7P8));
        lps.push_back(LPS(CIRCLE, R9P8));
    }else{
        cout << "param error!\n";
        return;
    }

    if(op == "LBPRIU2"){
        for(auto i: lps){
            tfdr.push_back(new RIU2(i.getP()));
            op1.push_back(new LBP(i, tfdr.back()));
        }
    }else if(op == "LBPMR"){
        for(auto i: lps){
            op1.push_back(new LBPMR(i));
        }
    }else if(op == "LBPMRK=0.8"){
        for(auto i: lps){
            tfdr.push_back(new DP<LBPMR>(i, data.TrainData, 0.8));
            op1.push_back(new LBPMR(i, tfdr.back()));
        }
    }else if(op == "LBPMRK2=0.8"){
        DP2<LBPMR>::init(lps.size(), 0.8);
        for(auto i: lps){
            tfdr.push_back(new DP2<LBPMR>(i, data.TrainData));
            op1.push_back(new LBPMR(i, tfdr.back()));
        }
    }else if(op == "LBPMR/GNPN=2"){
        for(auto i: lps){
            op1.push_back(new LBPMR(i));
        }
        op2 = new GNP(2);
    }else if(op == "LBPMR/GNPN=3"){
        for(auto i: lps){
            op1.push_back(new LBPMR(i));
        }
        op2 = new GNP(3);
    }else if(op == "LBPMRK=0.8/GNPN=2"){
        for(auto i: lps){
            tfdr.push_back(new DP<LBPMR>(i, data.TrainData, 0.8));
            op1.push_back(new LBPMR(i, tfdr.back()));
        }
        op2 = new GNP(2);
    }else if(op == "LBPMR/GNPN=3"){
        for(auto i: lps){
            tfdr.push_back(new DP<LBPMR>(i, data.TrainData, 0.8));
            op1.push_back(new LBPMR(i, tfdr.back()));
        }
        op2 = new GNP(3);
    }else{
        cout << "param error!\n";
        return;
    }

    uint8_t threadNum = 16;
	vector<thread*> app1Thread;
    /*-------------------------------Training-------------------------------*/
    Mat trainFeatureMat;
	appAllThreadEndFlag = 0;

    for (size_t i = 0; i < threadNum; i++){
		app1Thread.push_back(new thread(appThread1, \
			threadNum, i,\
			op1, op2, \
			ref(data.TrainData), ref(trainFeatureMat)));
		app1Thread[i]->detach();
	}

    while (appAllThreadEndFlag != threadNum){
		this_thread::sleep_for(chrono::milliseconds(100));
	}
	for (auto& i : app1Thread)
		delete i;
	app1Thread.clear();
	
	cout << "feature dimension : " << trainFeatureMat.cols << endl;
	svmTraining(trainFeatureMat, data.TrainLabel);
	//knnTraining(trainFeatureMat, trainLabelMat);

    /*-------------------------------Prediction-------------------------------*/
    Mat predictFeatureMat;
	appAllThreadEndFlag = 0;
	for (size_t i = 0; i < threadNum; i++){
		app1Thread.push_back(new thread(appThread1, \
			threadNum, i,\
			op1, op2, \
			ref(data.PredictData), ref(predictFeatureMat)));

		app1Thread[i]->detach();
	}
	while (appAllThreadEndFlag != threadNum){
		this_thread::sleep_for(chrono::milliseconds(100));
	}
	for (auto& i : app1Thread)
		delete i;
	app1Thread.clear();

	svmPrediction(predictFeatureMat, data.PredictLabel);
	//knnPrediction(testFeatureMat, testLabelMat);
    for(auto& i : op1){
        delete i;
    }
    op1.clear();

    for(auto& i : tfdr){
        delete i;
    }
    tfdr.clear();

    delete op2;

}


void experiment2(const string& dataset)
{
    int cnt = 1;
    
    if(dataset == "TC10" || dataset == "TC12_0" || dataset == "TC12_1" || dataset == "KTH-TIPS2-b"){
        cnt = 1;
    }else if(dataset == "CUReT" || dataset == "UIUC" || dataset == "UMD"){
        cnt = 50;
    }else{
        cout << "param error!\n";
        return;
    }
    vector<double> result;

    for (size_t t = 0; t < cnt; t++){
        LabeledData data;
        if(dataset == "TC10"){
            string subset = "TC10";
            string path = "../Datasets/Outex_TC_00010";
            if(!readOuluDataset(path, subset, data)){
                cerr << "Reading images failed\n";
                return;
            }
        }else if(dataset == "TC12_0"){
            string subset = "TC12_0";
            string path = "../Datasets/Outex_TC_00012";
            if(!readOuluDataset(path, subset, data)){
                cerr << "Reading images failed\n";
                return;
            }
        }else if(dataset == "TC12_1"){
            string subset = "TC12_1";
            string path = "../Datasets/Outex_TC_00012";
            if(!readOuluDataset(path, subset, data)){
                cerr << "Reading images failed\n";
                return;
            }
        }else if(dataset == "KTH-TIPS2-b"){
            string path = "../Datasets/KTH-TIPS2-b";
            if(!readKTHTIPS2bDataset(path, 3, data)){
                cerr << "Reading images failed\n";
                return;
            }
        }else if(dataset == "CUReT"){
            string path = "../Datasets/CUReT";
            if(!readCUReTDataset(path, 92/2, data)){
                cerr << "Reading images failed\n";
                return;
            }
        }else if(dataset == "UIUC"){
            string path = "../Datasets/UIUC";
            if(!readUIUCDataset(path, 40/2, data)){
                cerr << "Reading images failed\n";
                return;
            }
        }else if(dataset == "UMD"){
            string path = "../Datasets/UMD";
            if(!readUMDDataset(path, 40/2, data)){
                cerr << "Reading images failed\n";
                return;
            }
        }else{
            cout << "param error!\n";
            return;
        }

        LPS lps1 = LPS(CIRCLE, R1P8);
        LPS lps2 = LPS(CIRCLE, R2P8);
        LPS lps3 = LPS(CIRCLE, R3P8);
        LPS lps4 = LPS(CIRCLE, R4P8);

        vector<Operator*> op1;

        // DP2<LBPMR>::init(4, 0.8);
        // TFDR* tfdr1 = new DP2<LBPMR>(lps1, data.TrainData);
        // TFDR* tfdr2 = new DP2<LBPMR>(lps2, data.TrainData);
        // TFDR* tfdr3 = new DP2<LBPMR>(lps3, data.TrainData);
        // TFDR* tfdr4 = new DP2<LBPMR>(lps4, data.TrainData);

        TFDR* tfdr1 = new DP<LBPMR>(lps1, data.TrainData, 0.8);
        TFDR* tfdr2 = new DP<LBPMR>(lps2, data.TrainData, 0.8);
        TFDR* tfdr3 = new DP<LBPMR>(lps3, data.TrainData, 0.8);
        TFDR* tfdr4 = new DP<LBPMR>(lps4, data.TrainData, 0.8);

        op1.push_back(new LBPMR(lps1, tfdr1));
        op1.push_back(new LBPMR(lps2, tfdr2));
        op1.push_back(new LBPMR(lps3, tfdr3));
        op1.push_back(new LBPMR(lps4, tfdr4));

        Operator* op2 = new GNP(2);

        uint8_t threadNum = 16;
        vector<thread*> app1Thread;
        /*-------------------------------Training-------------------------------*/
        Mat trainFeatureMat;
        appAllThreadEndFlag = 0;

        for (size_t i = 0; i < threadNum; i++){
            app1Thread.push_back(new thread(appThread1, \
                threadNum, i,\
                op1, op2, \
                ref(data.TrainData), ref(trainFeatureMat)));
            app1Thread[i]->detach();
        }

        while (appAllThreadEndFlag != threadNum){
            this_thread::sleep_for(chrono::milliseconds(100));
        }
        for (auto& i : app1Thread)
            delete i;
        app1Thread.clear();
        
        //cout << "feature dimension : " << trainFeatureMat.cols << endl;
        svmTraining(trainFeatureMat, data.TrainLabel);
        //knnTraining(trainFeatureMat, trainLabelMat);

        /*-------------------------------Prediction-------------------------------*/
        Mat predictFeatureMat;
        appAllThreadEndFlag = 0;
        for (size_t i = 0; i < threadNum; i++){
            app1Thread.push_back(new thread(appThread1, \
                threadNum, i,\
                op1, op2, \
                ref(data.PredictData), ref(predictFeatureMat)));

            app1Thread[i]->detach();
        }
        while (appAllThreadEndFlag != threadNum){
            this_thread::sleep_for(chrono::milliseconds(100));
        }
        for (auto& i : app1Thread)
            delete i;
        app1Thread.clear();

        result.push_back(svmPrediction(predictFeatureMat, data.PredictLabel));
        //knnPrediction(testFeatureMat, testLabelMat);

        for(auto& i : op1){
            delete i;
        }
        op1.clear();
        delete op2;
        if(tfdr1 != nullptr){
            delete tfdr1;
        }
        if(tfdr2 != nullptr){
            delete tfdr2;
        }
        if(tfdr3 != nullptr){
            delete tfdr3;
        }
        if(tfdr4 != nullptr){
            delete tfdr4;
        }
    }
    if(result.size() != 1){
        double avr = std::accumulate(result.begin(), result.end(), 0.0)/result.size();
        printf("\nThe Average Prediction Accuracy is %.2f%%\n", avr);
    }
}

void experiment3()
{
    //Mat testImg = imread("../datasets/UIUC/S1001L01.jpg", IMREAD_GRAYSCALE);
	Mat testImg = imread("../Datasets/Outex_TC_00010/images/000000.ras");
    if(testImg.empty()){
        cout << "Reading image failed!\n";
        return;
    }
	cvtColor(testImg, testImg, COLOR_RGB2GRAY);
	int cnt = 1000;

	clock_t start, finish;
	double  time1, time2;

    TFDR* tfdr1 = new RI(8);

    Operator* op1 = new LBP(LPS(CIRCLE, R1P8), tfdr1); //LBPri
    Operator* op2 = new LBPMR(LPS(CIRCLE, R1P8)); //LBPmr

    //---------------------------------------------------------------------------
	start = clock();
	for (size_t i = 0; i < cnt; i++)
	{
		Mat img;
		op1->process(testImg, img);

		Mat feature = Mat::zeros(1, op1->getPN(), CV_32FC1);
		float* pFeature = feature.ptr<float>(0);
		matToHist(img, op1->getPN(), &pFeature[0]);
	}
	finish = clock();
	time1 = (double)(finish - start) / CLOCKS_PER_SEC;
	printf("LBPRI feature extraction time is %f ms\n", time1); //1000 * time1 / cnt

    //---------------------------------------------------------------------------
	start = clock();
	for (size_t i = 0; i < cnt; i++)
	{
		Mat img;
		op2->process(testImg, img);

		Mat feature = Mat::zeros(1, op2->getPN(), CV_32FC1);
		float* pFeature = feature.ptr<float>(0);
		matToHist(img, op2->getPN(), &pFeature[0]);
	}
	finish = clock();
	time2 = (double)(finish - start) / CLOCKS_PER_SEC;
	printf("LBPMR feature extraction time is %f ms\n", time2);

    printf("ratio is %.2f\n", time2 / time1);

}


void imgCvt(Mat& src){
    Mat dst = Mat(src.rows, src.cols, CV_8UC1);
	for (size_t i = 0; i < src.rows; i++)
	{	
		int* pSrc = src.ptr<int>(i);
		uint8_t* pDst = dst.ptr<uint8_t>(i);
		for (size_t j = 0; j < src.cols; j++)
		{
			pDst[j] = pSrc[j];
		}
	}
	src = dst;
}

void experiment0(){
    LPS lps1 = LPS(CIRCLE, R1P8);
    LPS lps2 = LPS(CIRCLE, R2P8);
    LPS lps3 = LPS(CIRCLE, R3P8);
    LPS lps4 = LPS(CIRCLE, R4P8);

    TFDR* riu2 = new RIU2(lps1.getP());

    Operator* lbp = new LBP(lps1);
    Operator* lbpriu2 = new LBP(lps1, riu2);
    Operator* lbpmr = new LBPMR(lps1);
    Operator* lbpmrriu2 = new LBPMR(lps1, riu2);

    //data is empty, so the following code can't normal run.
    LabeledData data;
    TFDR* dp1 = new DP<LBP>(lps1, data.TrainData, 0.8);
    Operator* lbpdp = new LBP(lps1, dp1);

    TFDR* dp2 = new DP<LBP>(lps1, data.TrainData, 0.8);
    Operator* lbpmrdp = new LBPMR(lps1, dp2);

    DP2<LBP>::init(4, 0.8);
    TFDR* dp2_1 = new DP2<LBP>(lps1, data.TrainData);
    TFDR* dp2_2 = new DP2<LBP>(lps2, data.TrainData);
    TFDR* dp2_3 = new DP2<LBP>(lps3, data.TrainData);
    TFDR* dp2_4 = new DP2<LBP>(lps4, data.TrainData);
    Operator* lbpdp2_1 = new LBP(lps1, dp2_1);
    Operator* lbpdp2_2 = new LBP(lps1, dp2_2);
    Operator* lbpdp2_3 = new LBP(lps1, dp2_3);
    Operator* lbpdp2_4 = new LBP(lps1, dp2_4);

    DP2<LBPMR>::init(4, 0.8);
    TFDR* dp2_5 = new DP2<LBPMR>(lps1, data.TrainData);
    TFDR* dp2_6 = new DP2<LBPMR>(lps2, data.TrainData);
    TFDR* dp2_7 = new DP2<LBPMR>(lps3, data.TrainData);
    TFDR* dp2_8 = new DP2<LBPMR>(lps4, data.TrainData);
    Operator* lbpdp2_5 = new LBPMR(lps1, dp2_5);
    Operator* lbpdp2_6 = new LBPMR(lps1, dp2_6);
    Operator* lbpdp2_7 = new LBPMR(lps1, dp2_7);
    Operator* lbpdp2_8 = new LBPMR(lps1, dp2_8);

    //You can combine the parameters, operators and tfdr method very flexibly.
}


// void experiment0_1(){
//     vector<string> dataset;
//     dataset.push_back("TC10");
//     dataset.push_back("TC12_0");
//     dataset.push_back("TC12_1");
//     dataset.push_back("CUReT");
//     dataset.push_back("UIUC");

//     for (auto& m : dataset)
//     {
//         LabeledData data;
//         string subset;
//         string path;

//         if(m == "TC10"){
//             string subset = "TC10";
//             string path = "../Datasets/Outex_TC_00010";
//             if(!readOuluDataset(path, subset, data)){
//                 cerr << "Reading images failed\n";
//                 return;
//             }
//         }else if(m == "TC12_0"){
//             string subset = "TC12_0";
//             string path = "../Datasets/Outex_TC_00012";
//             if(!readOuluDataset(path, subset, data)){
//                 cerr << "Reading images failed\n";
//                 return;
//             }
//         }else if(m == "TC12_1"){
//             string subset = "TC12_1";
//             string path = "../Datasets/Outex_TC_00012";
//             if(!readOuluDataset(path, subset, data)){
//                 cerr << "Reading images failed\n";
//                 return;
//             }
//         }else if(m == "CUReT"){
//             string path = "../Datasets/CUReT";
//             if(!readCUReTDataset(path, 92/2, data)){
//                 cerr << "Reading images failed\n";
//                 return;
//             }
//         }else if(m == "UIUC"){
//             string path = "../Datasets/UIUC";
//             if(!readUIUCDataset(path, 40/2, data)){
//                 cerr << "Reading images failed\n";
//                 return;
//             }
//         }else{

//         }

//         uint8_t threadNum = 12;
// 		vector<thread*> experiment0Thread;

//         Operator* op = new LBP_Exp0(CIRCLE, R1P8);
//         Mat dst = Mat::zeros(2, 256, CV_32SC1);

// 		appAllThreadEndFlag = 0;
// 		for (size_t i = 0; i < threadNum; i++)
// 		{
// 			experiment0Thread.push_back(new thread(appThread2, threadNum, i, \
// 				ref(op), \
// 				ref(data.TrainData), ref(dst)));
// 			experiment0Thread[i]->detach();
// 		}
// 		while (appAllThreadEndFlag != threadNum)
// 		{
// 			this_thread::sleep_for(chrono::milliseconds(100));
// 		}
// 		appAllThreadEndFlag = 0;
// 		for (auto& i : experiment0Thread)
// 			delete i;
// 		experiment0Thread.clear();

//         for (size_t i = 0; i < threadNum; i++)
// 		{
// 			experiment0Thread.push_back(new thread(appThread2, threadNum, i, \
// 				ref(op), \
// 				ref(data.PredictData), ref(dst)));
// 			experiment0Thread[i]->detach();
// 		}
// 		while (appAllThreadEndFlag != threadNum)
// 		{
// 			this_thread::sleep_for(chrono::milliseconds(100));
// 		}
// 		appAllThreadEndFlag = 0;
// 		for (auto& i : experiment0Thread)
// 			delete i;
// 		experiment0Thread.clear();
        
//         int* pS = dst.ptr<int>(0);
//         int* pM = dst.ptr<int>(1);

//         uint32_t S_Sum = pS[0] + pS[1], M_Sum = 0;

// 		for (size_t i = 0; i < 256; i++)
// 		{
// 			M_Sum += pM[i];
// 		}
// 		double S_Entropy = 0, M_Entropy = 0;

// 		for (size_t i = 0; i < 2; i++)
// 		{
// 			if (pS != 0)
// 			{
// 				double p = pS[i] / double(S_Sum);
// 				S_Entropy += -p * log(p);
// 			}
// 		}

// 		for (size_t i = 0; i < 256; i++)
// 		{
// 			if (pM[i] != 0)
// 			{
// 				double p = pM[i] / double(M_Sum);
// 				M_Entropy += -p * log(p);
// 			}
// 		}
// 		cout << m << " dataset: " << "S_Entropy = " << S_Entropy << " M_Entropy = " << M_Entropy << endl;
//     }
// }

void experiment0_2(){
    Mat src1 = imread("../SampleImgs/checkerboard1.png", IMREAD_GRAYSCALE);
    Mat src2 = imread("../SampleImgs/checkerboard2.png", IMREAD_GRAYSCALE);
    Mat src3 = imread("../SampleImgs/checkerboard3.png", IMREAD_GRAYSCALE);

    Mat dst1_1, dst1_2, dst2, dst3;

    Operator* op1 = new LBP(LPS(CIRCLE, R1P8));
    Operator* op2 = new LBPMR(LPS(CIRCLE, R1P8));

    op1->process(src1, dst1_1);
    op2->process(src1, dst1_2);

    op1->process(src2,  dst2);
    op1->process(src2,  dst3);

    imgCvt(dst1_1);
    imgCvt(dst1_2);
    imgCvt(dst2);
    imgCvt(dst3);

    imshow("cb1", src1);
    imshow("cb1 lbp", dst1_1);
    imshow("cb1 lbpmr", dst1_2);

    waitKey();

    imshow("cb2", src2);
    imshow("cb3", src3);
    

    Mat diff =  (dst2 != dst3);

    if(countNonZero(diff) == 0){
        imshow("dst2_3", dst2);
        cout << "Equal! \n";
    }else{
        cout << "Not equal!";
    }

    waitKey();

    delete op1;
    delete op2;
}

void experiment0_3(){
    Mat src1 = imread("../SampleImgs/lenna.png", IMREAD_GRAYSCALE);
    Mat src2 = imread("../SampleImgs/orange.jpg", IMREAD_GRAYSCALE);

    Mat dst1_1, dst1_2, dst2_1, dst2_2;

    LPS lps(CIRCLE, R1P8);
    TFDR* tfdr = new TFDR(pow(2, lps.getP()));

    Operator* op1 = new LBP(lps, tfdr);
    Operator* op2 = new LBPMR(lps, tfdr);

    op1->process(src1, dst1_1);
    op1->process(src2, dst2_1);

    op2->process(src1, dst1_2);
    op2->process(src2, dst2_2);

    imgCvt(dst1_1);
    imgCvt(dst1_2);
    imgCvt(dst2_1);
    imgCvt(dst2_2);

    imshow("lenna", src1);
    imshow("lenna lbp", dst1_1);
    imshow("lenna lbpmr", dst1_2);

    imshow("orange", src2);
    imshow("orange lbp", dst2_1);
    imshow("orange lbpmr", dst2_2);

    waitKey();

    delete op1;
    delete op2;
    delete tfdr;
}

Mat imgCvt2(const Mat& src, Mat& dst){
    Mat dst2 = Mat(dst.rows, dst.cols, CV_8UC1);
	for (size_t i = 0; i < dst.rows; i++)
	{	
		uint8_t const* pSrc = src.ptr<uint8_t>(i+1);
		int* pDst = dst.ptr<int>(i);
        uint8_t* pDst2 = dst2.ptr<uint8_t>(i);
		for (size_t j = 0; j < dst.cols; j++)
		{
            if(pDst[j] == 0){
                pDst2[j] = 0;
            }else{
                pDst2[j] = pSrc[j+1];
            }
		}
	}
	return dst2;
}

// void experiment0_5(){
//     LabeledData data;
//     string path = "../Datasets/UIUC";
//     if(!readUIUCDataset(path, 40/2, data)){
//         cerr << "Reading images failed\n";
//         return;
//     }

//     Operator* op = new LBPMR_(CIRCLE, R1P8, 0.1);
//     if(!(op->preTrain(data.TrainData))){
//         cerr << "PreTraining failed\n";
//         return;
//     }

//     for(auto& i : data.PredictData){
//         for(auto& j : i){
//             Mat tmp, dst;
//             op->process(j, tmp);
//             dst = imgCvt2(j, tmp);
//             imshow("src", j);
//             imshow("dst", dst);
//             if(waitKey() == 27){
//                 delete op;
//                 return;
//             }
//         }
//     }
// }

// void experiment0_6(){
//     LabeledData data;
//     string subset = "TC10";
//     string path = "../Datasets/Outex_TC_00010";
//     if(!readOuluDataset(path, subset, data)){
//         cerr << "Reading images failed\n";
//         return;
//     }

//     for(double k = 1; k >= 0.5;  k -= 0.1){
//         Operator* op = new LBPMR_(CIRCLE, R1P8, k);
//         if(!(op->preTrain(data.TrainData))){
//             cerr << "PreTraining failed\n";
//             return;
//         }
//         printf("K = %.2f, DP Number: %d\n", k, op->getPN());
//         delete op;
//     }
// }
