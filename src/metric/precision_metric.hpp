/*!
 * Copyright (c) 2025-2026 The LambdaGap contributors. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_SRC_METRIC_PRECISION_METRIC_HPP_
#define LIGHTGBM_SRC_METRIC_PRECISION_METRIC_HPP_

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

/*!
* \brief What the precision metrics share: the queries, their weights, and the weighted
*        average of a per-query value at each k of eval_at
*/
class QueryPrecisionMetric : public Metric {
 public:
  QueryPrecisionMetric(const Config& config, const char* name) : metric_name_(name) {
    eval_at_ = config.eval_at;
    DCGCalculator::DefaultEvalAt(&eval_at_);
  }

  void Init(const Metadata& metadata, data_size_t) override {
    for (auto k : eval_at_) {
      name_.emplace_back(metric_name_ + "@" + std::to_string(k));
    }
    label_ = metadata.label();
    query_boundaries_ = metadata.query_boundaries();
    if (query_boundaries_ == nullptr) {
      Log::Fatal("For the %s metric, there should be query information", metric_name_.c_str());
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

 protected:
  /*! \brief The weighted average over queries of per_query(query, &values at each k) */
  template <typename PerQuery>
  std::vector<double> AverageOverQueries(PerQuery per_query) const {
    std::vector<std::vector<double>> values(num_queries_, std::vector<double>(eval_at_.size(), 0.0f));
    #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(guided)
    for (data_size_t i = 0; i < num_queries_; ++i) {
      per_query(i, &values[i]);
    }
    // Sum in query order: summing per thread made the result depend on the scheduling,
    // which broke the ties between iterations differently from run to run
    std::vector<double> result(eval_at_.size(), 0.0f);
    for (data_size_t i = 0; i < num_queries_; ++i) {
      const double weight = query_weights_ == nullptr ? 1.0f : query_weights_[i];
      for (size_t j = 0; j < result.size(); ++j) {
        result[j] += values[i][j] * weight;
      }
    }
    for (size_t j = 0; j < result.size(); ++j) {
      result[j] /= sum_query_weights_;
    }
    return result;
  }

  std::string metric_name_;
  const label_t* label_;
  const data_size_t* query_boundaries_;
  data_size_t num_queries_;
  const label_t* query_weights_;
  double sum_query_weights_;
  std::vector<data_size_t> eval_at_;
  std::vector<std::string> name_;
};

class PrecisionMetric : public QueryPrecisionMetric {
 public:
  explicit PrecisionMetric(const Config& config) : QueryPrecisionMetric(config, "precision") {}

  void CalPrecisionAtK(const label_t* label, const double* score, data_size_t num_data,
                       std::vector<double>* out) const {
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
    for (size_t i = 0; i < eval_at_.size(); ++i) {
      data_size_t cur_k = static_cast<data_size_t>(eval_at_[i]);
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
    return AverageOverQueries([this, score](data_size_t i, std::vector<double>* out) {
      const data_size_t start = query_boundaries_[i];
      CalPrecisionAtK(label_ + start, score + start, query_boundaries_[i + 1] - start, out);
    });
  }
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
class PrecisionNMSMetric : public QueryPrecisionMetric {
 public:
  explicit PrecisionNMSMetric(const Config& config) : QueryPrecisionMetric(config, "precision_nms") {
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
    QueryPrecisionMetric::Init(metadata, num_data);
    groups_ = metadata.nms_groups();
    coordinates_ = metadata.nms_coordinates();
    tiebreaks_ = metadata.nms_tiebreaks();
    if (groups_ == nullptr || coordinates_ == nullptr) {
      // The booster creates the metrics of the training data too, which may not have them
      Log::Warning("The nms_group and nms_coordinate fields are not set: precision_nms is NaN for this dataset");
    }
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
    return AverageOverQueries([this, score](data_size_t i, std::vector<double>* out) {
      const data_size_t start = query_boundaries_[i];
      const data_size_t count = query_boundaries_[i + 1] - start;
      if (first_kept_) {
        CalPrecisionAtK<true>(start, count, score, out);
      } else {
        CalPrecisionAtK<false>(start, count, score, out);
      }
    });
  }

 private:
  const int32_t* groups_;
  const double* coordinates_;
  const double* tiebreaks_;
  double tolerance_;
  bool relative_;
  bool first_kept_;
  int max_k_;
};

}  // namespace LightGBM

#endif  // LIGHTGBM_SRC_METRIC_PRECISION_METRIC_HPP_
