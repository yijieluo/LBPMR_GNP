#include "LBPVariants.h"
#include "TFDR.h"

static inline void getBinWeight(uint32_t *diff, uint32_t *binWeight, uint8_t len){
	memset(binWeight, 0, sizeof(uint32_t)*len);
	for (int i = 0; i < len-1; ++i){
		for (int j = i + 1; j < len; ++j){
			if (diff[i] != diff[j])
				diff[i] > diff[j] ? ++binWeight[i] : ++binWeight[j];
		}
	}
}

static inline double biLinearInterpolation(const Mat &src, const Point2d &dst)
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

LBP::LBP(const LPS& lps_, TFDR* tfdr_): Operator(tfdr_), lps(lps_){
    int PN = pow(2, lps.getP());
    if(tfdr_ == nullptr){
        flag = true;
        table = new TFDR(PN);
    }
}
LBP::~LBP(){
    if(flag){
        delete table;
        table = nullptr;
    }
}
int LBP::getPN() const{
    return table->getPN();
}
void LBP::process(const Mat& src, Mat& dst){
    const int r = lps.getR();
    const int p = lps.getP();

    dst = Mat(src.rows - 2 * r, src.cols - 2 * r, CV_32SC1);

    int p1 = 0, p2 = p/4, p3 = p/2, p4 = 3*p/4;
    for (int i = r; i < src.rows - r; ++i){
        uint8_t const * pSrcRow  = src.ptr<uint8_t>(i);
        int* pDstRow  = dst.ptr<int>(i-r);
        for (int j = r; j < src.cols - r; ++j){
            int c = pSrcRow[j];
            int lbp = 0;
            for(int t = 0; t < p; ++t){
                int tmp;
                if(t == p1 || t == p2 || t == p3 || t == p4){
                    tmp = src.at<uint8_t>(i + lps.getRY()[t], j + lps.getRX()[t]) - c;
                }else{
                    tmp = round(biLinearInterpolation(src, Point2d(j+lps.getRX()[t], i+lps.getRY()[t])) - c);
                }
                lbp |= ((tmp > 0 ? 1 : 0) << t);
            }
            pDstRow[j-r] = (*table)[lbp];
        }
    }
}

void LBPMR::process(const Mat& src, Mat& dst){
    const int r = lps.getR();
    const int p = lps.getP();

    uint32_t *lbm = new uint32_t[p];
    uint32_t *lbw = new uint32_t[p];
    bool *lbs = new bool[p];
    dst = Mat(src.rows - 2 * r, src.cols - 2 * r, CV_32SC1);
    int p1 = 0, p2 = p/4, p3 = p/2, p4 = 3*p/4;
    for (size_t i = r; i < src.rows - r; ++i){
        uint8_t const * pSrcRow  = src.ptr<uint8_t>(i);
        int* pDstRow  = dst.ptr<int>(i-r);
        for (size_t j = r; j < src.cols - r; ++j){
            uint8_t c = pSrcRow[j];
            uint32_t lbpmr = 0;
            for(size_t t = 0; t < p; t++){
                int tmp;
                if(t == p1 || t == p2 || t == p3 || t == p4){
                    tmp = src.ptr<uint8_t>(i + lps.getRY()[t])[int(j+lps.getRX()[t])] - c;
                }else{
                    tmp = round(biLinearInterpolation(src, Point2d(j+lps.getRX()[t], i+lps.getRY()[t])) - c);
                }
                lbm[t] = abs(tmp);
                lbs[t] = (tmp > 0) ? 1 : 0;
            }
            getBinWeight(lbm, lbw, p);
            for (size_t t = 0; t < p; ++t){
                lbpmr += (lbs[t] << lbw[t]); //+= cannot be replaced by |=
            }
            pDstRow[j-r] = (*table)[lbpmr];
        }
    }
    delete[] lbm;
    delete[] lbw;
    delete[] lbs;
}

GNP::GNP(int N_): Operator(new TFDR(256)), N(N_){}
GNP::~GNP(){
    delete table;
}
void GNP::process(const Mat& src, Mat& dst){
    uint32_t maxGrayLevel = 256;
    uint32_t* cHist = new uint32_t[maxGrayLevel]{ 0 };
    uint8_t* cLookupTable = new uint8_t[maxGrayLevel]{ 0 };
    for (size_t i = 0; i < src.rows; i++){
        const uint8_t* pData = src.ptr<uint8_t>(i);
        for (size_t j = 0; j < src.cols; j++){
            cHist[pData[j]]++;
        }
    }

    uint32_t pointsNum = src.rows*src.cols;

    double* median = new double[N-1]{ 0 };
    uint32_t tmp1 = 0;
    for (uint16_t i = 0, j = 1; i < maxGrayLevel; i++){
        tmp1 += cHist[i];
        double tmp2 = pointsNum * j / double(N);
        if (tmp1 > tmp2){
            median[j - 1] = i;
            if (j == (N-1))
                break;
            j++;
        }else if(tmp1 == tmp2){
            median[j - 1] = i + 0.5;
            if (j == (N-1))
                break;
            j++;
        }
    }
    for (uint16_t i = 0, j = 0; i < maxGrayLevel; i++){
        if (j < N-1){
            if (i < median[j]){
                cLookupTable[i] = j;
            }else{
                j++;
                cLookupTable[i] = j;
            }
        }else{
            cLookupTable[i] = j;
        }
    }

    dst = Mat(src.rows, src.cols, CV_32SC1);
    for (size_t i = 0; i < src.rows; i++){
        const uint8_t* pSrcData = src.ptr<uint8_t>(i);
        int32_t* pDstData = dst.ptr<int32_t>(i);
        for (size_t j = 0; j < src.cols; j++){
            pDstData[j] = cLookupTable[pSrcData[j]];
        }
    }
    delete[] median;
    delete[] cHist;
    delete[] cLookupTable;
}


// mutex dstLock;
// void LBP_Exp0::process(const Mat& src, Mat& dst){
//     const int r = lp.getR();
//     const int p = lp.getP();

//     int p1 = 0, p2 = p/4, p3 = p/2, p4 = 3*p/4;
//     int* pS = dst.ptr<int>(0);
//     int* pM = dst.ptr<int>(1);

//     for (int i = r; i < src.rows - r; ++i){
//         uint8_t const * pSrcRow  = src.ptr<uint8_t>(i);
//         for (int j = r; j < src.cols - r; ++j){
//             int c = pSrcRow[j];
//             for(int t = 0; t < p; ++t){
//                 int tmp;
//                 if(t == p1 || t == p2 || t == p3 || t == p4){
//                     tmp = src.at<uint8_t>(i + lp.getRY()[t], j + lp.getRX()[t]) - c;
//                 }else{
//                     tmp = round(biLinearInterpolation(src, Point2d(j+lp.getRX()[t], i+lp.getRY()[t])) - c);
//                 }
//                 dstLock.lock();
//                 tmp > 0 ? pS[0] ++ : pS[1] ++;
//                 pM[abs(tmp)]++;
//                 dstLock.unlock();
//             }
//         }
//     }

// }