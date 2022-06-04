#ifndef TFDR_H
#define TFDR_H

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "AppThread.h"
#include "LBPVariants.h"

using namespace cv;
using namespace std;

class TFDR //texture feature dimension reduction
{
protected:
    int* table = nullptr;
    int PN;
public:
    TFDR(int dim){
        if(dim <= 0){
            return;
        }
        table = new int[dim];
        for (size_t i = 0; i < dim; i++)
        {
            table[i] = i;
        }
        PN = dim;
    }
    virtual ~TFDR(){
        if(table != nullptr){
            delete [] table;
            table = nullptr;
        }
    }
    virtual inline int operator[](int in){
        return table[in];
    }
    virtual int getPN(){
        return PN;
    }
};

class RI: public TFDR{
private:
    uint8_t calRI(int t, int enob) const{
		return 0;
	}
public:
    RI(int p): TFDR(pow(2, p)){
        if(p != 8){
            cout << "param error!\n";
        }
        int dim = pow(2, p);
        for (size_t i = 0; i < dim; i++){
            table[i] = calRI(i, p);
        };
        PN = 36;
    }
    virtual ~RI(){}
};

class RIU2: public TFDR{
private:
    uint8_t calUniform(int t, int enob) const{
		int lowestBit = t & 1;
		int t_ = t >> 1 | (lowestBit << (enob - 1));
		int u2 = 0;
		for(uint8_t i=0; i < enob; i++){
			int tmp = 1 << i;
			u2 += ((t&tmp)!=(t_&tmp));
		}
		uint8_t out = 0;
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
public:
    RIU2(int p): TFDR(pow(2, p)){
        int dim = pow(2, p);
        for (size_t i = 0; i < dim; i++){
            table[i] = calUniform(i, p);
        };
        PN = p+2;
    }
    virtual ~RIU2(){}
};

#endif