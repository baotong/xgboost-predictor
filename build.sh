g++ -std=c++11 -ggdb -I include/ -I ../dmlc-core/include/ test/predict_test.cc -L../dmlc-core -ldmlc -lpthread -o gbdt_predict
