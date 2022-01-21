#include "LBPVariants.h"

void LBP::process(const Mat& src, Mat& dst){
    const int r = lp.getR();
    const int p = lp.getP();

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
                    tmp = src.at<uint8_t>(i + lp.getRY()[t], j + lp.getRX()[t]) - c;
                }else{
                    tmp = round(biLinearInterpolation(src, Point2d(j+lp.getRX()[t], i+lp.getRY()[t])) - c);
                }
                lbp |= ((tmp >= 0 ? 1 : 0) << t);
            }
            pDstRow[j-r] = table[lbp];
        }
    }
}