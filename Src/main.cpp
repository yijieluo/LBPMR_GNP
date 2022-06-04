#include "Application.h"

int main()
{
    // vector<string> dataset{"TC10"/*,"TC12_0","TC12_1"*/};
    // vector<string> op{"LBPMRK=0.8/GNPN=3"};
    // vector<string> op{"LBPMR/GNPN=2", "LBPMR/GNPN=3"};
    // vector<string> sch{"SCH1","SCH2","SCH3","SCH4", "SCH5"};
    // for(auto& d : dataset){
    //     for(auto& o : op){
    //         for(auto& s : sch){
    //             cout << "\n" << d << "   " << o << "   " << s << "\n";
    //             experiment1(d, o, s);
    //             cout << "-----------------------------\n";
    //         }
    //     }
    // }

    experiment1("TC10", "LBPMRK=0.8/GNPN=2", "SCH4");

    // cout << "-------------KTH-TIPS2-b----------------\n";
    //experiment2("KTH-TIPS2-b");//Need to modify the C (SVM parameter).
    
    // cout << "-------------CUReT----------------\n";
    // experiment2("CUReT");

    // // cout << "-------------UIUC----------------\n";
    // experiment2("UIUC");

    // // cout << "-------------UMD----------------\n";
    // experiment2("UMD");

    //experiment3();//Compile this function under the RELEASE mode.


    //experiment0(); //Example of how to combine various parameters, opertors and TFDR methods.

    //experiment0_1();
    //experiment0_2();
    //experiment0_3();
    //experiment0_5();
    //experiment0_6();
    return 0;
}
