#ifndef GNP_H
#define GNP_H
#include "Operator.h"

 
class GNP : public Operator{
    int PN;
public:
	GNP(int N): PN(N){}
	~GNP(){};
	virtual void process(const Mat& src, Mat& dst){
        uint32_t maxGrayLevel = 256;
        uint32_t* cHist = new uint32_t[maxGrayLevel]{ 0 };
        uint8_t* cLookupTable = new uint8_t[maxGrayLevel]{ 0 };
        for (size_t i = 0; i < src.rows; i++)
        {
            const uint8_t* pData = src.ptr<uint8_t>(i);
            for (size_t j = 0; j < src.cols; j++)
            {
                cHist[pData[j]]++;
            }
        }

        uint32_t pointsNum = src.rows*src.cols;

        double* median = new double[PN-1]{ 0 };
        uint32_t tmp1 = 0;
        for (uint16_t i = 0, j = 1; i < maxGrayLevel; i++)
        {
            tmp1 += cHist[i];
            double tmp2 = pointsNum * j / double(PN);
            if (tmp1 > tmp2)
            {
                median[j - 1] = i;
                if (j == (PN-1))
                    break;
                j++;
            }else if(tmp1 == tmp2){
                median[j - 1] = i + 0.5;
                if (j == (PN-1))
                    break;
                j++;
            }
        }
        for (uint16_t i = 0, j = 0; i < maxGrayLevel; i++)
        {
            if (j < PN-1)
            {
                if (i < median[j])
                {
                    cLookupTable[i] = j;
                }
                else
                {
                    j++;
                    cLookupTable[i] = j;
                }
            }
            else
            {
                cLookupTable[i] = j;
            }
        }

        dst = Mat(src.rows, src.cols, CV_32SC1);
        for (size_t i = 0; i < src.rows; i++)
        {
            const uint8_t* pSrcData = src.ptr<uint8_t>(i);
            int32_t* pDstData = dst.ptr<int32_t>(i);
            for (size_t j = 0; j < src.cols; j++)
            {
                pDstData[j] = cLookupTable[pSrcData[j]];
            }
        }
        delete[] median;
        delete[] cHist;
        delete[] cLookupTable;
    }
	virtual int getPN() const{
        return PN;
    }
};

#endif