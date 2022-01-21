#ifndef LBPVARIANTS_H
#define LBPVARIANTS_H

#include "Operator.h"
#include "math.h"

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
inline double biLinearInterpolation(const Mat &src, const Point2d &dst)
{
    Rect rect(floor(dst.x), floor(dst.y), 1, 1);

    // uint8_t lt = src.at<uint8_t>(rect.y, rect.x),
    //         rt = src.at<uint8_t>(rect.y, rect.x + 1),
    //         lb = src.at<uint8_t>(rect.y + 1, rect.x),
    //         rb = src.at<uint8_t>(rect.y + 1, rect.x + 1);
    uint8_t lt = src.ptr<uint8_t>(rect.y)[rect.x],
            rt = src.ptr<uint8_t>(rect.y)[rect.x + 1],
            lb = src.ptr<uint8_t>(rect.y + 1)[rect.x],
            rb = src.ptr<uint8_t>(rect.y + 1)[rect.x + 1];

    double v1 = lt + (rt-lt)*(dst.x-rect.x);
    double v2 = lb + (rb-lb)*(dst.x-rect.x);
    return v1 + (v2-v1)*(dst.y-rect.y);
}

class LocalPixels{
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
	LocalPixels(LOCAL_SAMPLE_SHAPE shape_, LOCAL_SAMPLE_RP rp_): R(setR(rp_)), P(setP(rp_)){
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
    ~LocalPixels(){
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

class LBP : public Operator{
protected:
    const LocalPixels lp;
    int* table;
public:
	LBP(LOCAL_SAMPLE_SHAPE shape_, LOCAL_SAMPLE_RP rp_):lp(shape_, rp_){
        table = new int[getPN()];
        for (size_t i = 0; i < getPN(); i++){
            table[i] = i;
        };
    }
	~LBP(){
        delete[] table;
    };
	virtual void process(const Mat& src, Mat& dst);
	virtual int getPN() const{
        return pow(2, lp.getP());
    }
};

class LBPRI : public LBP{
public:
	LBPRI(LOCAL_SAMPLE_SHAPE shape_, LOCAL_SAMPLE_RP rp_):LBP(shape_, rp_){
        int p = pow(2, lp.getP());
        for (size_t i = 0; i < p; i++){
            table[i] = 0;//waiting to be realized
        }
    }
	~LBPRI(){};
	virtual int getPN() const{
        if(lp.getP() == 8){
            return 36;
        }else{
            return 0;//waiting to be realized
        }
    }
};

class LBPRIU2 : public LBP{
public:
	LBPRIU2(LOCAL_SAMPLE_SHAPE shape_, LOCAL_SAMPLE_RP rp_):LBP(shape_, rp_){
        int p = pow(2, lp.getP());
        for (size_t i = 0; i < p; i++){
            table[i] = calUniform(i, lp.getP());
        };
    }
	~LBPRIU2(){};
	virtual int getPN() const{
        return lp.getP()+2;
    }
    int calUniform(int t, int enob) const{
		int lowestBit = t & 1;
		int t_ = t >> 1 | (lowestBit << (enob - 1));
		int u2 = 0;
		for(uint8_t i=0; i < enob; i++){
			int tmp = 1 << i;
			u2 += ((t&tmp)!=(t_&tmp));
		}
		int out = 0;
		if(u2 > 2){
			out = enob + 1;
		}else{
			for (size_t i = 0; i < enob; i++)
			{
				if(t & (1 <<i))
					out += 1;
			}
		}
		return out;
	}
};


#endif