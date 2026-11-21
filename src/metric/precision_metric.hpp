#ifndef LIGHTGBM_METRIC_PRECISION_METRIC_HPP_
#define LIGHTGBM_METRIC_PRECISION_METRIC_HPP_

#include <LightGBM/metric.h>
#include <LightGBM/utils/common.h>
#include <LightGBM/utils/log.h>
#include <LightGBM/utils/openmp_wrapper.h>

#include <string>
#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_set>
#include <sstream>
#include <vector>

namespace LightGBM {

/*! \brief A score for sorting: NaN as the lowest, so the order stays strict and weak */
inline double SortableScore(double score) {
  return std::isnan(score) ? -std::numeric_limits<double>::infinity() : score;
}

class PrecisionMetric:public Metric {
 public:
  explicit PrecisionMetric(const Config& config) {
    // get eval position
    eval_at_ = config.eval_at;
    DCGCalculator::DefaultEvalAt(&eval_at_);
  }

  ~PrecisionMetric() {
  }

  void Init(const Metadata& metadata, data_size_t num_data) override {
    for (auto k : eval_at_) {
      name_.emplace_back(std::string("precision@") + std::to_string(k));
    }
    num_data_ = num_data;
    // get label
    label_ = metadata.label();
    // get query boundaries
    query_boundaries_ = metadata.query_boundaries();
    if (query_boundaries_ == nullptr) {
      Log::Fatal("For PRECISION metric, there should be query information");
    }
    num_queries_ = metadata.num_queries();
    Log::Info("Total groups: %d, total data: %d", num_queries_, num_data_);
    // get query weights
    query_weights_ = metadata.query_weights();
    if (query_weights_ == nullptr) {
      sum_query_weights_ = static_cast<double>(num_queries_);
    } else {
      sum_query_weights_ = 0.0f;
      for (data_size_t i = 0; i < num_queries_; ++i) {
        sum_query_weights_ += query_weights_[i];
      }
    }
  }

  const std::vector<std::string>& GetName() const override {
    return name_;
  }

  double factor_to_bigger_better() const override {
    return 1.0f;
  }

  void CalPrecisionAtK(std::vector<int> ks, const label_t* label,
                 const double* score, data_size_t num_data, std::vector<double>* out) const {
    // get sorted indices by score
    std::vector<data_size_t> sorted_idx;
    for (data_size_t i = 0; i < num_data; ++i) {
      sorted_idx.emplace_back(i);
    }
    std::stable_sort(sorted_idx.begin(), sorted_idx.end(),
                     [score](data_size_t a, data_size_t b) {
                       return SortableScore(score[a]) > SortableScore(score[b]);
                     });

    int num_hit = 0;
    data_size_t cur_left = 0;
    for (size_t i = 0; i < ks.size(); ++i) {
      data_size_t cur_k = static_cast<data_size_t>(ks[i]);
      for (data_size_t j = cur_left; j < cur_k && j < num_data; ++j) {
        data_size_t idx = sorted_idx[j];
        if (label[idx] > 0.5f) {
          ++num_hit;
        }
      }
      (*out)[i] = static_cast<double>(num_hit) / cur_k;
      cur_left = cur_k;
    }
  }
  std::vector<double> Eval(const double* score, const ObjectiveFunction*) const override {
    std::vector<std::vector<double>> per_query(num_queries_, std::vector<double>(eval_at_.size(), 0.0f));
    #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(guided)
    for (data_size_t i = 0; i < num_queries_; ++i) {
      CalPrecisionAtK(eval_at_, label_ + query_boundaries_[i],
                score + query_boundaries_[i], query_boundaries_[i + 1] - query_boundaries_[i], &per_query[i]);
    }
    // Sum in query order: summing per thread made the result depend on the scheduling,
    // which broke the ties between iterations differently from run to run
    std::vector<double> result(eval_at_.size(), 0.0f);
    for (data_size_t i = 0; i < num_queries_; ++i) {
      const double weight = query_weights_ == nullptr ? 1.0f : query_weights_[i];
      for (size_t j = 0; j < result.size(); ++j) {
        result[j] += per_query[i][j] * weight;
      }
    }
    for (size_t j = 0; j < result.size(); ++j) {
      result[j] /= sum_query_weights_;
    }
    return result;
  }

 private:
  /*! \brief Number of data */
  data_size_t num_data_;
  /*! \brief Pointer of label */
  const label_t* label_;
  /*! \brief Query boundaries information */
  const data_size_t* query_boundaries_;
  /*! \brief Number of queries */
  data_size_t num_queries_;
  /*! \brief Weights of queries */
  const label_t* query_weights_;
  /*! \brief Sum weights of queries */
  double sum_query_weights_;
  /*! \brief Evaluate position of Nprecision */
  std::vector<data_size_t> eval_at_;
  std::vector<std::string> name_;
};

/*!
* \brief Precision at k after non-maximum suppression. Within a query, rows go by
*        decreasing score, equal scores by decreasing nms_tiebreak when it is set, then
*        by row order. A row counts only if its nms_coordinate is farther than
*        nms_tolerance from the coordinate of every row kept before, and if no row of its
*        nms_group was kept before (nms_group_mode first_kept) or came before at all
*        (best_row: a group whose best row is suppressed counts no row). Precision at k
*        is the share of relevant rows (label > 0) among the first k rows kept.
*/
class PrecisionNMSMetric: public Metric {
 public:
  explicit PrecisionNMSMetric(const Config& config) {
    eval_at_ = config.eval_at;
    DCGCalculator::DefaultEvalAt(&eval_at_);
    max_k_ = *std::max_element(eval_at_.begin(), eval_at_.end());
    tolerance_ = config.nms_tolerance;
    if (config.nms_tolerance_mode == std::string("absolute")) {
      relative_ = false;
    } else if (config.nms_tolerance_mode == std::string("relative")) {
      relative_ = true;
    } else {
      Log::Fatal("Unknown nms_tolerance_mode: %s", config.nms_tolerance_mode.c_str());
    }
    if (config.nms_group_mode == std::string("best_row")) {
      first_kept_ = false;
    } else if (config.nms_group_mode == std::string("first_kept")) {
      first_kept_ = true;
    } else {
      Log::Fatal("Unknown nms_group_mode: %s", config.nms_group_mode.c_str());
    }
  }

  void Init(const Metadata& metadata, data_size_t num_data) override {
    for (auto k : eval_at_) {
      name_.emplace_back(std::string("precision_nms@") + std::to_string(k));
    }
    num_data_ = num_data;
    label_ = metadata.label();
    query_boundaries_ = metadata.query_boundaries();
    if (query_boundaries_ == nullptr) {
      Log::Fatal("For the precision_nms metric, there should be query information");
    }
    groups_ = metadata.nms_groups();
    coordinates_ = metadata.nms_coordinates();
    tiebreaks_ = metadata.nms_tiebreaks();
    if (groups_ == nullptr || coordinates_ == nullptr) {
      // The booster creates the metrics of the training data too, which may not have them
      Log::Warning("The nms_group and nms_coordinate fields are not set: precision_nms is NaN for this dataset");
    }
    num_queries_ = metadata.num_queries();
    query_weights_ = metadata.query_weights();
    if (query_weights_ == nullptr) {
      sum_query_weights_ = static_cast<double>(num_queries_);
    } else {
      sum_query_weights_ = 0.0f;
      for (data_size_t i = 0; i < num_queries_; ++i) {
        sum_query_weights_ += query_weights_[i];
      }
    }
  }

  const std::vector<std::string>& GetName() const override {
    return name_;
  }

  double factor_to_bigger_better() const override {
    return 1.0f;
  }

  // The group rule as a template parameter: chosen once per query, not per row
  template <bool kFirstKept>
  void CalPrecisionAtK(data_size_t start, data_size_t num_data, const double* score,
                       std::vector<double>* out) const {
    std::vector<data_size_t> sorted_idx(num_data);
    for (data_size_t i = 0; i < num_data; ++i) {
      sorted_idx[i] = start + i;
    }
    const double* tiebreaks = tiebreaks_;
    std::stable_sort(sorted_idx.begin(), sorted_idx.end(),
                     [score, tiebreaks](data_size_t a, data_size_t b) {
                       const double sa = SortableScore(score[a]), sb = SortableScore(score[b]);
                       if (sa != sb || tiebreaks == nullptr) {
                         return sa > sb;
                       }
                       return tiebreaks[a] > tiebreaks[b];
                     });
    std::unordered_set<int32_t> seen_groups;
    std::vector<double> kept;
    std::vector<int> hits;  // Relevant rows among the first i + 1 rows kept
    for (data_size_t idx : sorted_idx) {
      if (kFirstKept ? seen_groups.count(groups_[idx]) > 0
                     : !seen_groups.insert(groups_[idx]).second) {
        continue;
      }
      const double x = coordinates_[idx];
      if (!kept.empty()) {
        double distance = std::numeric_limits<double>::infinity();
        for (double k : kept) {
          distance = std::min(distance, relative_ ? RelativeDistance(x, k) : std::fabs(x - k));
        }
        if (!(distance > tolerance_)) {
          continue;
        }
      }
      kept.push_back(x);
      if (kFirstKept) {
        seen_groups.insert(groups_[idx]);
      }
      hits.push_back((hits.empty() ? 0 : hits.back()) + (label_[idx] > 0.5f));
      if (static_cast<int>(kept.size()) == max_k_) {
        break;
      }
    }
    for (size_t i = 0; i < eval_at_.size(); ++i) {
      const size_t k = static_cast<size_t>(eval_at_[i]);
      const int num_hit = hits.empty() ? 0 : hits[std::min(k, hits.size()) - 1];
      (*out)[i] = static_cast<double>(num_hit) / k;
    }
  }

  /*! \brief |x - k| relative to the larger magnitude, 0 when both are 0 */
  static double RelativeDistance(double x, double k) {
    const double scale = std::max(std::fabs(x), std::fabs(k));
    return scale == 0.0 ? 0.0 : std::fabs(x - k) / scale;
  }

  std::vector<double> Eval(const double* score, const ObjectiveFunction*) const override {
    if (groups_ == nullptr || coordinates_ == nullptr) {
      return std::vector<double>(eval_at_.size(), std::numeric_limits<double>::quiet_NaN());
    }
    std::vector<std::vector<double>> per_query(num_queries_, std::vector<double>(eval_at_.size(), 0.0f));
    #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(guided)
    for (data_size_t i = 0; i < num_queries_; ++i) {
      const data_size_t start = query_boundaries_[i];
      const data_size_t count = query_boundaries_[i + 1] - start;
      if (first_kept_) {
        CalPrecisionAtK<true>(start, count, score, &per_query[i]);
      } else {
        CalPrecisionAtK<false>(start, count, score, &per_query[i]);
      }
    }
    // Summed in query order, the same on every run
    std::vector<double> result(eval_at_.size(), 0.0f);
    for (data_size_t i = 0; i < num_queries_; ++i) {
      const double weight = query_weights_ == nullptr ? 1.0f : query_weights_[i];
      for (size_t j = 0; j < result.size(); ++j) {
        result[j] += per_query[i][j] * weight;
      }
    }
    for (size_t j = 0; j < result.size(); ++j) {
      result[j] /= sum_query_weights_;
    }
    return result;
  }

 private:
  data_size_t num_data_;
  const label_t* label_;
  const data_size_t* query_boundaries_;
  data_size_t num_queries_;
  const label_t* query_weights_;
  double sum_query_weights_;
  const int32_t* groups_;
  const double* coordinates_;
  const double* tiebreaks_;
  double tolerance_;
  bool relative_;
  bool first_kept_;
  int max_k_;
  std::vector<data_size_t> eval_at_;
  std::vector<std::string> name_;
};

}  // namespace LightGBM

#endif   // LIGHTGBM_METRIC_PRECISION_METRIC_HPP_
