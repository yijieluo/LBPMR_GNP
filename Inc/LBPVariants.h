#ifndef LBPVARIANTS_H
#define LBPVARIANTS_H



#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "math.h"
#include <atomic>

using namespace cv;
using namespace std;

typedef vector<vector<Mat>> Data;

class TFDR;

class Operator{
protected:
	TFDR* table = nullptr;
public:
	Operator(TFDR* table_): table(table_){}
	virtual ~Operator(){}
	virtual void process(const Mat& src, Mat& dst) = 0;
	virtual int getPN() const = 0;//get Pattern Number
};

enum LOCAL_SAMPLE_SHAPE
{
	CIRCLE = 0,
	SQUARE
};
enum LOCAL_SAMPLE_RP
{
	R1P8 = 1, R2P8 = 2, R3P8 = 3, R4P8 = 4, R5P8 = 5, R6P8 = 6, R7P8 = 7, R8P8 = 8, R9P8 = 9, 
	R2P16 = 12, R3P24 = 23, R4P24 = 24, R5P24 = 25, R6P24 = 26, R7P24 = 27, R8P24 = 28, R9P24 = 29,
};

//Local Pixels Sample
class LPS{
    const int R, P;
    double* RX;
    double* RY;

    constexpr int setR(LOCAL_SAMPLE_RP rp){
        return rp%10;
    }
    constexpr int setP(LOCAL_SAMPLE_RP rp){
        return (rp/10 + 1) * 8;
    }
public:
	LPS(LOCAL_SAMPLE_SHAPE shape_, LOCAL_SAMPLE_RP rp_): R(setR(rp_)), P(setP(rp_)){
        RX = new double[P];
        RY = new double[P];
        const double PI = 3.14159;
        if(shape_ == CIRCLE){
            for(int i = 0; i < P; i++){
                RX[i] = R * cos(2*PI*i/P);
                RY[i] = R * sin(2*PI*i/P);
            }
            RX[0] = R; RY[0] = 0;
            RX[P/4] = 0; RY[P/4] = R;
            RX[P/2] = -R; RY[P/2] = 0;
            RX[P*3/4] = 0; RY[P*3/4] = -R;
        }else if(shape_ == SQUARE){
            // for(int i = 0; i < P; i++){
            //     if(i < )
            // }waiting for realizing
        }
    }
    LPS(const LPS& lp): R(lp.R), P(lp.P){
        RX = new double[P];
        RY = new double[P];
        for (size_t i = 0; i < P; i++)
        {
            RX[i] = lp.RX[i];
            RY[i] = lp.RY[i];
        }//or memcpy
    }
    ~LPS(){
        delete [] RX;
        delete [] RY;
    }
    const double* const getRX() const{
        return RX;
    }
    const double* const getRY() const{
        return RY;
    }
    constexpr int getR() const{
        return R;
    }
    constexpr int getP() const{
        return P;
    }
};


class LBP : public Operator {
protected:
    const LPS lps;
    bool flag = false;
public:
    LBP(const LPS& lps_, TFDR* tfdr_ = nullptr);
	virtual ~LBP();
	virtual void process(const Mat& src, Mat& dst);
	virtual int getPN() const;
};

// class LBP_Exp0 : public LBP{
// public:
// 	LBP_Exp0(LOCAL_SAMPLE_SHAPE shape_, LOCAL_SAMPLE_RP rp_):LBP(shape_, rp_){}
//     virtual ~LBP_Exp0(){};
//     virtual void process(const Mat& src, Mat& dst);
// };

class LBPMR : public LBP{
public:
	LBPMR(const LPS& lp_, TFDR* tfdr_ = nullptr): LBP(lp_, tfdr_){};
    virtual ~LBPMR(){}
	virtual void process(const Mat& src, Mat& dst);
};

class GNP : public Operator{
int N;
public:
    GNP(int N_);
    virtual ~GNP();
    virtual void process(const Mat& src, Mat& dst);
    virtual int getPN() const {return N;}
};


#endif