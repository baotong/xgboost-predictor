#include "predictor.h"
#include "tree_model.h"

using namespace xgboost;
using namespace std;

int main() {
    //std::unique_ptr<dmlc::Stream> fi(dmlc::Stream::Create("data/0002.model", "r"));
    
    Predictor*  pred = new Predictor();
    pred->Load("data/0002.model");
    pred->DumpModel();
    cout << "predict test: " << std::endl;
    unordered_map<size_t, float> inst = {{3,1},  {9,1}, {19,1}, {21,1}, {24,1}, {34,1}, {36,1}, {39,1}, {51,1},
                                           {53,1}, {56,1}, {65,1}, {69,1}, {77,1}, {86,1}, {88,1}, {92,1}, {95,1},
                                           {102,1}, {106,1}, {116,1}, {122,1}};
	unordered_map<size_t, float> inst1 = {{56,0}};
    float pred_val1 = pred->Predict(&inst, false, 0);
    cout << "pred_value : " << pred_val1 << endl;
    delete pred;

    return 0;
}
