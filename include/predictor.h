/*!
 * Copyright 2014 by Contributors
 * \file learner.cc
 * \brief Implementation of learning algorithm.
 * \author Tianqi Chen
 */
#ifndef XGBOOST_PREDICTOR_H
#define XGBOOST_PREDICTOR_H

#include <dmlc/io.h>
#include <dmlc/timer.h>
#include <algorithm>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include "io.h"
#include "gbtree_model.h"
#include "tree_model.h"

namespace xgboost {
/*! \brief training parameter for regression */
    struct LearnerModelParam : public dmlc::Parameter<LearnerModelParam> {
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

        void Load(dmlc::Stream *fi) {
            // TODO(tqchen) mark deprecation of old format.
            common::PeekableInStream fp(fi);
            // backward compatible header check.
            std::string header;
            header.resize(4);
            if (fp.PeekRead(&header[0], 4) == 4) {
                CHECK_NE(header, "bs64")
                        << "Base64 format is no longer supported in brick.";
                if (header == "binf") {
                    CHECK_EQ(fp.Read(&header[0], 4), 4U);
                }
            }
            // use the peekable reader.
            fi = &fp;
            // read parameter
            CHECK_EQ(fi->Read(&mparam, sizeof(mparam)), sizeof(mparam))
                    << "BoostLearner: wrong model format";
            {
                // backward compatibility code for compatible with old model type
                // for new model, Read(&name_obj_) is suffice
                uint64_t len;
                CHECK_EQ(fi->Read(&len, sizeof(len)), sizeof(len));
                if (len >= std::numeric_limits<unsigned>::max()) {
                    int gap;
                    CHECK_EQ(fi->Read(&gap, sizeof(gap)), sizeof(gap))
                            << "BoostLearner: wrong model format";
                    len = len >> static_cast<uint64_t>(32UL);
                }
                if (len != 0) {
                    name_obj_.resize(len);
                    CHECK_EQ(fi->Read(&name_obj_[0], len), len)
                            << "BoostLearner: wrong model format";
                }
            }
            CHECK(fi->Read(&name_gbm_)) << "BoostLearner: wrong model format";
            // duplicated code with LazyInitModel
            //obj_.reset(ObjFunction::Create(name_obj_));
            gbm_.reset(new gbm::GBTreeModel(mparam.base_score));
            gbm_->Load(fi);
            /*if (mparam.contain_extra_attrs != 0) {
              std::vector<std::pair<std::string, std::string> > attr;
              fi->Read(&attr);
              attributes_ =
                  std::map<std::string, std::string>(attr.begin(), attr.end());
            }
            if (name_obj_ == "count:poisson") {
              std::string max_delta_step;
              fi->Read(&max_delta_step);
              cfg_["max_delta_step"] = max_delta_step;
            }*/
            /*if (mparam.contain_eval_metrics != 0) {
              std::vector<std::string> metr;
              fi->Read(&metr);
              for (auto name : metr) {
                metrics_.emplace_back(Metric::Create(name));
              }
            }*/
        }


        inline float Sigmoid(float x) const {
            return 1.0f / (1.0f + std::exp(-x));
        }

        float Predict(RegTree::FVec &feats,
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
            std::cout << "base_socre: " << mparam.base_score << std::endl;
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
