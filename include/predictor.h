/*!
 * Copyright 2014 by Contributors
 * \file learner.cc
 * \brief Implementation of learning algorithm.
 * \author Tianqi Chen
 */
#ifndef XGBOOST_PREDICTOR_H
#define XGBOOST_PREDICTOR_H

#include <algorithm>
#include <memory>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include <unordered_map>
#include <fstream>
#include "gbtree_model.h"
#include "tree_model.h"

namespace xgboost {
/*! \brief training parameter for regression */
    struct LearnerModelParam {
        /* \brief global bias */
        bst_float base_score;
        /* \brief number of features  */
        unsigned num_feature;
        /* \brief number of classes, if it is multi-class classification  */
        int num_class;
        /*! \brief Model contain additional properties */
        int contain_extra_attrs;
        /*! \brief Model contain eval metrics */
        int contain_eval_metrics;
        /*! \brief reserved field */
        int reserved[29];
        /*! \brief constructor */
        LearnerModelParam() {
            std::memset(this, 0, sizeof(LearnerModelParam));
            base_score = 0.5f;
        }
    };


/*!
 * \brief learner that performs gradient boosting for a specific objective
 * function. It does training and prediction.
 */
    class Predictor {
    public:

        void InitModel() {}

		int Load(const std::string& model_path) {
            // TODO: add exception handling
            std::ifstream ifile(model_path, std::ios::binary|std::ios::in);
            if (!ifile) {
                std::cout << "read file error: " << model_path << std::endl;
            }
            {
                ifile.read((char*)&mparam, sizeof(mparam));
                uint64_t len;
                ifile.read((char*)&len, sizeof(len));
                if (len >= std::numeric_limits<unsigned>::max()) {
                    std::cerr << "len is too large: " << len << std::endl;
                    int gap;
                    ifile.read((char*)&gap, sizeof(gap));
                    len = len >> static_cast<uint64_t>(32UL);
                }
                
                if (len != 0) {
                    std::cout << "name_obj len: " << len << std::endl;
                    name_obj_.resize(len);
                    ifile.read((char*)&name_obj_[0], len);
                    std::cout << "name_obj: " << name_obj_ << std::endl;
                }
                // TODO: check size               
                ifile.read((char*)&len, sizeof(len));
                std::cout << "name gbm length: " << len << std::endl;
                name_gbm_.resize(len);
                ifile.read((char*)&name_gbm_[0], len);
                std::cout << "gbm name: " << name_gbm_ << std::endl;
                gbm_.reset(new gbm::GBTreeModel(mparam.base_score));
                gbm_->Load(ifile);
                
            }
            ifile.close();
        }
        


        inline float Sigmoid(float x) const {
            return 1.0f / (1.0f + std::exp(-x));
        }
		
		float Predict(const std::unordered_map<uint64_t, bst_float>* feats,
				bool output_margin, unsigned ntree_limit) const {
			FVec fvec;
			fvec.Set(feats);
			return PredictFVec(fvec, output_margin, ntree_limit);
		}

        inline float PredictFVec(FVec &feats,
                      bool output_margin,
                      unsigned ntree_limit) const {
            if (ntree_limit == 0 || ntree_limit > gbm_->trees.size()) {
                ntree_limit = static_cast<unsigned>(gbm_->trees.size());
            }

            float predict_val = gbm_->PredictInstanceRaw(feats, 0, ntree_limit);
            if (!output_margin) {
                return Sigmoid(predict_val);
            } else {
                return predict_val;
            }
        }

        void DumpModel() {
            std::cout << "base_score: " << mparam.base_score << std::endl;
            std::cout << "number_feature: " << mparam.num_feature << std::endl;
            std::cout << "number_class: " << mparam.num_class << std::endl;
            std::cout << "number_trees: " << gbm_->param.num_trees << std::endl;
        }

    protected:
        // return whether model is already initialized.
        inline bool ModelInitialized() const { return gbm_.get() != nullptr; }

        // model parameter
        LearnerModelParam mparam;
        // temporal storages for prediction
        // std::vector<bst_float> preds_;
        std::unique_ptr<gbm::GBTreeModel> gbm_;
        // name of gbm
        std::string name_gbm_;
        // name of objective function
        std::string name_obj_;

    private:
        /*! \brief random number transformation seed. */
        static const int kRandSeedMagic = 127;
    };

}  // namespace xgboost

#endif
