/*!
 * Copyright by Contributors 2017
 */
#ifndef XGBOOST_GBTREE_MODEL_H
#define XGBOOST_GBTREE_MODEL_H

#include <dmlc/parameter.h>
#include <dmlc/io.h>
#include <utility>
#include <string>
#include <vector>
#include "tree_model.h"

namespace xgboost {
    namespace gbm {
/*! \brief model parameters */
        struct GBTreeModelParam : public dmlc::Parameter<GBTreeModelParam> {
            /*! \brief number of trees */
            int num_trees;
            /*! \brief number of roots */
            int num_roots;
            /*! \brief number of features to be used by trees */
            int num_feature;
            /*! \brief pad this space, for backward compatibility reason.*/
            int pad_32bit;
            /*! \brief deprecated padding space. */
            int64_t num_pbuffer_deprecated;
            /*!
             * \brief how many output group a single instance can produce
             *  this affects the behavior of number of output we have:
             *    suppose we have n instance and k group, output will be k * n
             */
            int num_output_group;
            /*! \brief size of leaf vector needed in tree */
            int size_leaf_vector;
            /*! \brief reserved parameters */
            int reserved[32];

            /*! \brief constructor */
            GBTreeModelParam() {
                std::memset(this, 0, sizeof(GBTreeModelParam));
                static_assert(sizeof(GBTreeModelParam) == (4 + 2 + 2 + 32) * sizeof(int),
                              "64/32 bit compatibility issue");
            }

            // declare parameters, only declare those that need to be set.
            DMLC_DECLARE_PARAMETER(GBTreeModelParam) {
                    DMLC_DECLARE_FIELD(num_output_group)
                            .set_lower_bound(1)
                            .set_default(1)
                            .describe(
                                    "Number of output groups to be predicted,"
                                            " used for multi-class classification.");
                    DMLC_DECLARE_FIELD(num_roots).set_lower_bound(1).set_default(1).describe(
                    "Tree updater sequence.");
                    DMLC_DECLARE_FIELD(num_feature)
                    .set_lower_bound(0)
                    .describe("Number of features used for training and prediction.");
                    DMLC_DECLARE_FIELD(size_leaf_vector)
                    .set_lower_bound(0)
                    .set_default(0)
                    .describe("Reserved option for vector tree.");
            }

        };

        class GBTreeModel {
        public:
            explicit GBTreeModel(bst_float base_margin) : base_margin(base_margin) {}

            void Configure(const std::vector <std::pair<std::string, std::string>> &cfg) {
                // initialize model parameters if not yet been initialized.
                if (trees.size() == 0) {
                    param.InitAllowUnknown(cfg);
                }
            }

            void InitTreesToUpdate() {
                /*if (trees_to_update.size() == 0u) {
                  for (size_t i = 0; i < trees.size(); ++i) {
                    trees_to_update.push_back(std::move(trees[i]));
                  } */
                trees.clear();
                param.num_trees = 0;
                tree_info.clear();
            }

            void Load(dmlc::Stream *fi) {
                CHECK_EQ(fi->Read(&param, sizeof(param)), sizeof(param))
                        << "GBTree: invalid model file";
                trees.clear();

                for (int i = 0; i < param.num_trees; ++i) {
                    std::unique_ptr <RegTree> ptr(new RegTree());
                    ptr->Load(fi);
                    trees.push_back(std::move(ptr));
                }
                tree_info.resize(param.num_trees);
                if (param.num_trees != 0) {
                    CHECK_EQ(
                            fi->Read(dmlc::BeginPtr(tree_info), sizeof(int) * param.num_trees),
                            sizeof(int) * param.num_trees);
                }
            }

            float PredictInstanceRaw(RegTree::FVec &feats, unsigned tree_begin, unsigned tree_end) {
                bst_float psum = base_margin;

                for (size_t i = tree_begin; i < tree_end; ++i) {
                    // bst_group = 1, for binary classification
                    // default root_index=0
                    int tid = trees[i]->GetLeafIndex(feats);
                    psum += (*trees[i])[tid].leaf_value();
                }
                return psum;
            }


        public:
            // base margin
            bst_float base_margin;
            // model parameter
            GBTreeModelParam param;
            /*! \brief vector of trees stored in the model */
            std::vector <std::unique_ptr<RegTree>> trees;
            /*! \brief for the update process, a place to keep the initial trees */
            //std::vector<std::unique_ptr<RegTree> > trees_to_update;
            /*! \brief some information indicator of the tree, reserved */
            std::vector<int> tree_info;
        };
    }  // namespace gbm
}  // namespace xgboost

#endif
