#include "Application.h"

atomic_uint8_t appAllThreadEndFlag{0};

void experiment1(const string& dataset, const string& op, const string& sch)
{
	LabeledData data;
    string subset;
    string path;

	vector<Operator*> op1;
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

    vector<LOCAL_SAMPLE_RP> rp;
    if(sch == "SCH1"){
        rp.push_back(R1P8);
    }else if(sch == "SCH2"){
        rp.push_back(R1P8);
        rp.push_back(R2P8);
    }else if(sch == "SCH3"){
        rp.push_back(R1P8);
        rp.push_back(R2P8);
        rp.push_back(R3P8);
    }else if(sch == "SCH4"){
        rp.push_back(R1P8);
        rp.push_back(R2P8);
        rp.push_back(R3P8);
        rp.push_back(R4P8);
    }else if(sch == "SCH5"){
        rp.push_back(R1P8);
        rp.push_back(R2P8);
        rp.push_back(R3P8);
        rp.push_back(R4P8);
        rp.push_back(R5P8);
    }else{
        cout << "param error!\n";
        return;
    }

    if(op == "LBPRIU2"){
        for(auto i: rp){
            op1.push_back(new LBPRIU2(CIRCLE, i));
        }
    }else if(op == "LBPMR"){
        for(auto i: rp){
            op1.push_back(new LBPMR(CIRCLE, i));
        }
    }else if(op == "LBPMRK=0.8"){
        for(auto i: rp){
            op1.push_back(new LBPMR(CIRCLE, i, 0.8));
        }
        for(auto& i: op1){
            if(!(i->preTrain(data.TrainData))){
                cerr << "PreTraining failed\n";
                return;
            }
        }
    }else if(op == "LBPMRK=0.8_GNPN=2"){
        for(auto i: rp){
            op1.push_back(new LBPMR(CIRCLE, i, 0.8));
        }
        for(auto& i: op1){
            if(!(i->preTrain(data.TrainData))){
                cerr << "PreTraining failed\n";
                return;
            }
        }
        op2 = new GNP(2);
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

        vector<Operator*> op1;
        op1.push_back(new LBPMR(CIRCLE, R1P8, 0.8));
        op1.push_back(new LBPMR(CIRCLE, R2P8, 0.8));
        op1.push_back(new LBPMR(CIRCLE, R3P8, 0.8));
        op1.push_back(new LBPMR(CIRCLE, R4P8, 0.8));

        for(auto& i: op1){
            if(!(i->preTrain(data.TrainData))){
                cerr << "PreTraining failed\n";
                return;
            }
        }
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

        for (auto& i : LBPMR::sm.histogram){
            if(i != nullptr){
                delete[] i;
                i = nullptr;
            }
        }
        LBPMR::sm.histogram.clear();
        for (auto& i : LBPMR::sm.lookuptable){
            if(i != nullptr){
                delete[] i;
                i = nullptr;
            }
        }
        LBPMR::sm.lookuptable.clear();
    }
    if(result.size() != 1){
        double avr = std::accumulate(result.begin(), result.end(), 0)/result.size();
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
	double  time;

    Operator* op1 = new LBPRI(CIRCLE, R1P8);
    Operator* op2 = new LBPMR(CIRCLE, R1P8);

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
	time = (double)(finish - start) / CLOCKS_PER_SEC;
	printf("LBPRI feature extraction time is %f s\n", time / cnt);

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
	time = (double)(finish - start) / CLOCKS_PER_SEC;
	printf("LBPMR feature extraction time is %f s\n", time / cnt);
}