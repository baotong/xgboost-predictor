#ifndef XGBOOST_FVEC_H_
#define XGBOOST_FVEC_H_

#include <unordered_map>

/*!
 * \brief dense feature vector that can be taken by RegTree
 * and can be construct from sparse feature vector.
 */

typedef float bst_float;

namespace xgboost {
    class FVecBase {
    public:
        /*!
         * \brief initialize the vector with size vector
         * \param size The size of the feature vector.
         */
        virtual void Init(size_t size);
        /*!
         * \brief fill the vector with sparse vector
         * \param inst The sparse instance to fill.
         */
        //inline void Fill(const RowBatch::Inst& inst);
        virtual void Set(const std::unordered_map<size_t, bst_float>* feature_map);
        /*!
         * \brief drop the trace after fill, must be called after fill.
         * \param inst The sparse instance to drop.
         */
        //inline void Drop(const RowBatch::Inst& inst);
        /*!
         * \brief returns the size of the feature vector
         * \return the size of the feature vector
         */
        virtual size_t size() const;

        /*!
         * \brief get ith value
         * \param i feature index.
         * \return the i-th feature value
         */
        virtual bst_float fvalue(size_t i) const;

        /*!
         * \brief check whether i-th entry is missing
         * \param i feature index.
         * \return whether i-th value is missing.
         */
        virtual bool is_missing(size_t i) const;

    private:
        /*!
         * \brief a union value of value and flag
         *  when flag == -1, this indicate the value is missing
         */
        /*union Entry {
          bst_float fvalue;
          int flag;
          };*/
        // std::vector<Entry> data;
        // change inner implementation from vector to unordered map
        const std::unordered_map<size_t, bst_float>* data = nullptr;
    };

    class FVec {
    public:

        bst_float fvalue(size_t i) const {
            const auto& res = data->find(i);
            if (res != data->end()) {
                return data->at(i);
            } else {
                return 0;
            }
        }

        void Init(size_t size) {
            //Entry e;
            //e.flag = -1;
            // data.resize(size);
            // std::fill(data.begin(), data.end(), e);
        }

        void Set(const std::unordered_map<size_t, bst_float>* feature_map) {
            data = feature_map;
        }


        size_t size() const {
            return data->size();
        }


        bool is_missing(size_t i) const {
            const auto& res = data->find(i);
            if (res != data->end()) {
                return false;
            } else {
                return true;
            }
        }    
    
    private:
        const std::unordered_map<size_t, bst_float>* data = nullptr;
    };
}

#endif //XGBOOST_FVEC_H_
