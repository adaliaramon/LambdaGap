/*!
 * Copyright (c) 2020 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#ifndef LIGHTGBM_OBJECTIVE_RANK_OBJECTIVE_HPP_
#define LIGHTGBM_OBJECTIVE_RANK_OBJECTIVE_HPP_

#include <LightGBM/metric.h>
#include <LightGBM/objective_function.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace LightGBM {

enum class LambdaRankTarget {
  NDCG,
  LAMBDALOSS_NDCG,
  LAMBDALOSS_NDCG_PLUS_PLUS,
  BNDCG,
  LAMBDALOSS_BNDCG,
  LAMBDALOSS_BNDCG_PLUS_PLUS,
  PRECISION,
  ARP_K,
  LAMBDALOSS_ARP1,
  LAMBDALOSS_ARP2,
  RANKNET,
  BIN_RANKNET,
  LAMBDAGAP_S,
  LAMBDAGAP_X,
  LAMBDAGAP_S_PLUS,
  LAMBDAGAP_X_PLUS,
  LAMBDAGAP_S_PLUS_PLUS,
  LAMBDAGAP_X_PLUS_PLUS,
};

/*!
 * \brief Objective function for Ranking
 */
class RankingObjective : public ObjectiveFunction {
 public:
  explicit RankingObjective(const Config& config)
      : seed_(config.objective_seed) {
    learning_rate_ = config.learning_rate;
    position_bias_regularization_ = config.lambdarank_position_bias_regularization;
  }

  explicit RankingObjective(const std::vector<std::string>&) : seed_(0) {}

  ~RankingObjective() {}

  void Init(const Metadata& metadata, data_size_t num_data) override {
    num_data_ = num_data;
    // get label
    label_ = metadata.label();
    // get weights
    weights_ = metadata.weights();
    // get positions
    positions_ = metadata.positions();
    // get position ids
    position_ids_ = metadata.position_ids();
    // get number of different position ids
    num_position_ids_ = static_cast<data_size_t>(metadata.num_position_ids());
    // get boundaries
    query_boundaries_ = metadata.query_boundaries();
    if (query_boundaries_ == nullptr) {
      Log::Fatal("Ranking tasks require query information");
    }
    num_queries_ = metadata.num_queries();
    // initialize position bias vectors
    pos_biases_.resize(num_position_ids_, 0.0);
    // Allocate a vector of size num_queries_ to store the number of effective pairs per query
    effective_pairs_.resize(num_queries_, 0.0);
  }

  void GetGradients(const double* score, const data_size_t num_sampled_queries, const data_size_t* sampled_query_indices,
                    score_t* gradients, score_t* hessians) const override {
    const data_size_t num_queries = (sampled_query_indices == nullptr ? num_queries_ : num_sampled_queries);
#pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(guided)
    for (data_size_t i = 0; i < num_queries; ++i) {
      const data_size_t query_index = (sampled_query_indices == nullptr ? i : sampled_query_indices[i]);
      const data_size_t start = query_boundaries_[query_index];
      const data_size_t cnt = query_boundaries_[query_index + 1] - query_boundaries_[query_index];
      std::vector<double> score_adjusted;
      if (num_position_ids_ > 0) {
        for (data_size_t j = 0; j < cnt; ++j) {
          score_adjusted.push_back(score[start + j] + pos_biases_[positions_[start + j]]);
        }
      }
      GetGradientsForOneQuery(query_index, cnt, label_ + start, num_position_ids_ > 0 ? score_adjusted.data() : score + start,
                              gradients + start, hessians + start);
      if (weights_ != nullptr) {
        for (data_size_t j = 0; j < cnt; ++j) {
          gradients[start + j] =
              static_cast<score_t>(gradients[start + j] * weights_[start + j]);
          hessians[start + j] =
              static_cast<score_t>(hessians[start + j] * weights_[start + j]);
        }
      }
    }

    // Log the average of effective_pairs_ over all queries
    double avg_effective_pairs = 0.0;
    for (data_size_t i = 0; i < num_queries_; ++i) {
      if (std::isnan(effective_pairs_[i])) {
        continue;
      }
      avg_effective_pairs += effective_pairs_[i];
    }
    Log::Debug("Average effective pairs per query: %.4f%%", 100.0 * avg_effective_pairs / num_queries_);

    if (num_position_ids_ > 0) {
      UpdatePositionBiasFactors(gradients, hessians);
    }
  }

  void GetGradients(const double* score, score_t* gradients, score_t* hessians) const override {
    GetGradients(score, num_queries_, nullptr, gradients, hessians);
  }

  virtual void GetGradientsForOneQuery(data_size_t query_id, data_size_t cnt,
                                       const label_t* label,
                                       const double* score, score_t* lambdas,
                                       score_t* hessians) const = 0;

  virtual void UpdatePositionBiasFactors(const score_t* /*lambdas*/, const score_t* /*hessians*/) const {}

  const char* GetName() const override = 0;

  std::string ToString() const override {
    std::stringstream str_buf;
    str_buf << GetName();
    return str_buf.str();
  }

  bool NeedAccuratePrediction() const override { return false; }

 protected:
  int seed_;
  data_size_t num_queries_;
  /*! \brief Number of data */
  data_size_t num_data_;
  /*! \brief Pointer of label */
  const label_t* label_;
  /*! \brief Pointer of weights */
  const label_t* weights_;
  /*! \brief Pointer of positions */
  const data_size_t* positions_;
  /*! \brief Pointer of position IDs */
  const std::string* position_ids_;
  /*! \brief Pointer of label */
  data_size_t num_position_ids_;
  /*! \brief Query boundaries */
  const data_size_t* query_boundaries_;
  /*! \brief Position bias factors */
  mutable std::vector<label_t> pos_biases_;
  /*! \brief Learning rate to update position bias factors */
  double learning_rate_;
  /*! \brief Position bias regularization */
  double position_bias_regularization_;
  /*! \brief Effective pairs per query */
  mutable std::vector<double> effective_pairs_;
};

/*!
 * \brief Objective function for LambdaRank with NDCG
 */
class LambdarankNDCG : public RankingObjective {
 public:
  explicit LambdarankNDCG(const Config& config)
      : RankingObjective(config),
        sigmoid_(config.sigmoid),
        norm_(config.lambdarank_norm),
        truncation_level_(config.lambdarank_truncation_level) {
    label_gain_ = config.label_gain;
    std::map<std::string, LambdaRankTarget> lambdarank_target_map = {
      {"ndcg", LambdaRankTarget::NDCG},
      {"lambdaloss-ndcg", LambdaRankTarget::LAMBDALOSS_NDCG},
      {"lambdaloss-ndcg-plus-plus", LambdaRankTarget::LAMBDALOSS_NDCG_PLUS_PLUS},
      {"bndcg", LambdaRankTarget::BNDCG},
      {"lambdaloss-bndcg", LambdaRankTarget::LAMBDALOSS_BNDCG},
      {"lambdaloss-bndcg-plus-plus", LambdaRankTarget::LAMBDALOSS_BNDCG_PLUS_PLUS},
      {"precision", LambdaRankTarget::PRECISION},
      {"arpk", LambdaRankTarget::ARP_K},
      {"lambdaloss-arp1", LambdaRankTarget::LAMBDALOSS_ARP1},
      {"lambdaloss-arp2", LambdaRankTarget::LAMBDALOSS_ARP2},
      {"ranknet", LambdaRankTarget::RANKNET},
      {"bin-ranknet", LambdaRankTarget::BIN_RANKNET},
      {"lambdagap-s", LambdaRankTarget::LAMBDAGAP_S},
      {"lambdagap-x", LambdaRankTarget::LAMBDAGAP_X},
      {"lambdagap-s-plus", LambdaRankTarget::LAMBDAGAP_S_PLUS},
      {"lambdagap-x-plus", LambdaRankTarget::LAMBDAGAP_X_PLUS},
      {"lambdagap-s-plus-plus", LambdaRankTarget::LAMBDAGAP_S_PLUS_PLUS},
      {"lambdagap-x-plus-plus", LambdaRankTarget::LAMBDAGAP_X_PLUS_PLUS}
    };
    if (lambdarank_target_map.count(config.lambdarank_target) == 0) {
      Log::Fatal("Unknown lambdarank target '%s'", config.lambdarank_target.c_str());
    } else {
      Log::Info("Using lambdarank objective with target '%s'", config.lambdarank_target.c_str());
    }
    lambdarank_target_ = lambdarank_target_map[config.lambdarank_target];
    // initialize DCG calculator
    DCGCalculator::DefaultLabelGain(&label_gain_);
    DCGCalculator::Init(label_gain_);
    sigmoid_table_.clear();
    inverse_max_dcgs_.clear();
    inverse_max_bdcgs_.clear();
    if (sigmoid_ <= 0.0) {
      Log::Fatal("Sigmoid param %f should be greater than zero", sigmoid_);
    }
    lambdagap_weight_ = config.lambdagap_weight;
  }

  explicit LambdarankNDCG(const std::vector<std::string>& strs)
      : RankingObjective(strs) {}

  ~LambdarankNDCG() {}

  void Init(const Metadata& metadata, data_size_t num_data) override {
    RankingObjective::Init(metadata, num_data);
    DCGCalculator::CheckMetadata(metadata, num_queries_);
    DCGCalculator::CheckLabel(label_, num_data_);
    inverse_max_dcgs_.resize(num_queries_);
    inverse_max_bdcgs_.resize(num_queries_);
#pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static)
    for (data_size_t i = 0; i < num_queries_; ++i) {
      inverse_max_dcgs_[i] = DCGCalculator::CalMaxDCGAtK(
          truncation_level_, label_ + query_boundaries_[i],
          query_boundaries_[i + 1] - query_boundaries_[i]);

      if (inverse_max_dcgs_[i] > 0.0) {
        inverse_max_dcgs_[i] = 1.0f / inverse_max_dcgs_[i];
      }

      inverse_max_bdcgs_[i] = DCGCalculator::CalMaxBDCGAtK(
          truncation_level_, label_ + query_boundaries_[i],
          query_boundaries_[i + 1] - query_boundaries_[i]);

      if (inverse_max_bdcgs_[i] > 0.0) {
        inverse_max_bdcgs_[i] = 1.0f / inverse_max_bdcgs_[i];
      }
    }
    // construct Sigmoid table to speed up Sigmoid transform
    ConstructSigmoidTable();
  }

  inline void GetGradientsForOneQuery(data_size_t query_id, data_size_t cnt,
                                      const label_t* label, const double* score,
                                      score_t* lambdas,
                                      score_t* hessians) const override {
    // initialize with zero
    for (data_size_t i = 0; i < cnt; ++i) {
      lambdas[i] = 0.0f;
      hessians[i] = 0.0f;
    }
    // get sorted indices for scores
    std::vector<data_size_t> sorted_idx(cnt);
    for (data_size_t i = 0; i < cnt; ++i) {
      sorted_idx[i] = i;
    }
    double best_score, worst_score;
    if (lambdarank_target_ == LambdaRankTarget::PRECISION) {
      std::nth_element(
          sorted_idx.begin(), sorted_idx.begin() + truncation_level_ - 1, sorted_idx.end(),
          [score](data_size_t a, data_size_t b) { return score[a] > score[b]; });
      const auto [worst, best] = std::minmax_element(score, score + cnt);
      best_score = *best;
      worst_score = *worst;
    } else if (lambdarank_target_ == LambdaRankTarget::BIN_RANKNET ||
      lambdarank_target_ == LambdaRankTarget::RANKNET ||
      lambdarank_target_ == LambdaRankTarget::LAMBDALOSS_ARP1 ||
      lambdarank_target_ == LambdaRankTarget::LAMBDALOSS_ARP2
    ) {
      // No need to sort, since weighting factor does not depend neither on i nor j
      const auto [worst, best] = std::minmax_element(score, score + cnt);
      best_score = *best;
      worst_score = *worst;
    } else {
      std::stable_sort(
          sorted_idx.begin(), sorted_idx.end(),
          [score](data_size_t a, data_size_t b) { return score[a] > score[b]; });
      best_score = score[sorted_idx[0]];
      data_size_t worst_idx = cnt - 1;
      if (worst_idx > 0 && score[sorted_idx[worst_idx]] == kMinScore) {
        worst_idx -= 1;
      }
      worst_score = score[sorted_idx[worst_idx]];
    }
    // get best and worst score
    double sum_lambdas = 0.0;
    int count_lambdas = 0;

    // Pairs must include at least one document within the top k for some metrics
    data_size_t i_end;
    switch (lambdarank_target_) {
      case LambdaRankTarget::NDCG:
      case LambdaRankTarget::LAMBDALOSS_NDCG:
      case LambdaRankTarget::LAMBDALOSS_NDCG_PLUS_PLUS:
      case LambdaRankTarget::BNDCG:
      case LambdaRankTarget::LAMBDALOSS_BNDCG:
      case LambdaRankTarget::LAMBDALOSS_BNDCG_PLUS_PLUS:
      case LambdaRankTarget::PRECISION:
        i_end = std::min(cnt - 1, truncation_level_);
        break;
      default:
        i_end = cnt - 1;
    }

    for (data_size_t i = 0; i < i_end; ++i) {
      if (score[sorted_idx[i]] == kMinScore) {
        continue;
      }

      // Exclude pairs guaranteed to have 0 contribution to the gradient
      data_size_t start;
      data_size_t end;
      switch (lambdarank_target_) {
        case LambdaRankTarget::PRECISION:
          start = truncation_level_;
          end = cnt;
          break;
        case LambdaRankTarget::ARP_K:
        case LambdaRankTarget::LAMBDAGAP_S_PLUS:
        case LambdaRankTarget::LAMBDAGAP_X_PLUS:
        case LambdaRankTarget::LAMBDAGAP_S_PLUS_PLUS:
        case LambdaRankTarget::LAMBDAGAP_X_PLUS_PLUS:
          start = std::max(i + 1, truncation_level_);
          end = cnt;
          break;
        case LambdaRankTarget::LAMBDAGAP_S:
          start = i + truncation_level_;
          end = std::min(start + 1, cnt);
          break;
        case LambdaRankTarget::LAMBDAGAP_X:
          start = i + truncation_level_;
          end = cnt;
          break;
        default:
          start = i + 1;
          end = cnt;
      }

      for (data_size_t j = start; j < end; ++j) {
        if (score[sorted_idx[j]] == kMinScore) {
          continue;
        }
        // skip pairs with the same labels
        const score_t label_i = label[sorted_idx[i]];
        const score_t label_j = label[sorted_idx[j]];
        if (label_i == label_j) {
          continue;
        }

        // Skip pairs where none of the labels is 0, if the target is binary
        if (label_i > 0 && label_j > 0) {
          if (lambdarank_target_ == LambdaRankTarget::PRECISION ||
              lambdarank_target_ == LambdaRankTarget::BNDCG ||
              lambdarank_target_ == LambdaRankTarget::LAMBDALOSS_BNDCG ||
              lambdarank_target_ == LambdaRankTarget::LAMBDALOSS_BNDCG_PLUS_PLUS ||
              lambdarank_target_ == LambdaRankTarget::ARP_K ||
              lambdarank_target_ == LambdaRankTarget::BIN_RANKNET ||
              lambdarank_target_ == LambdaRankTarget::LAMBDAGAP_S ||
              lambdarank_target_ == LambdaRankTarget::LAMBDAGAP_X ||
              lambdarank_target_ == LambdaRankTarget::LAMBDAGAP_S_PLUS ||
              lambdarank_target_ == LambdaRankTarget::LAMBDAGAP_X_PLUS ||
              lambdarank_target_ == LambdaRankTarget::LAMBDAGAP_S_PLUS_PLUS ||
              lambdarank_target_ == LambdaRankTarget::LAMBDAGAP_X_PLUS_PLUS) {
            continue;
          }
        }

        data_size_t high_rank, low_rank;
        if (label[sorted_idx[i]] > label[sorted_idx[j]]) {
          high_rank = i;
          low_rank = j;
        } else {
          high_rank = j;
          low_rank = i;
        }
        const data_size_t high = sorted_idx[high_rank];
        const data_size_t low = sorted_idx[low_rank];

        const double high_score = score[high];
        const double low_score = score[low];
        const double delta_score = high_score - low_score;

        double delta_pair;
        if (lambdarank_target_ == LambdaRankTarget::NDCG) {
          const int high_label = static_cast<int>(label[high]);
          const int low_label = static_cast<int>(label[low]);
          const double high_label_gain = label_gain_[high_label];
          const double high_discount = DCGCalculator::GetDiscount(high_rank);
          const double low_label_gain = label_gain_[low_label];
          const double low_discount = DCGCalculator::GetDiscount(low_rank);
          // get dcg gap
          const double dcg_gap = high_label_gain - low_label_gain;
          // get discount of this pair
          const double paired_discount = fabs(high_discount - low_discount);
          // get max DCG on current query & get delta NDCG
          const double inverse_max_dcg = inverse_max_dcgs_[query_id];
          delta_pair = dcg_gap * paired_discount * inverse_max_dcg;
        } else if (lambdarank_target_ == LambdaRankTarget::LAMBDALOSS_NDCG) {
          const int high_label = static_cast<int>(label[high]);
          const int low_label = static_cast<int>(label[low]);
          const double high_label_gain = label_gain_[high_label];
          const double low_label_gain = label_gain_[low_label];
          const int rank_diff = j - i;
          const double left_discount = DCGCalculator::GetDiscount(rank_diff);
          const double right_discount = DCGCalculator::GetDiscount(rank_diff + 1);
          // get dcg gap
          const double dcg_gap = high_label_gain - low_label_gain;
          // get discount of this pair
          const double paired_discount = left_discount - right_discount;
          // get max DCG on current query & get delta NDCG
          const double inverse_max_dcg = inverse_max_dcgs_[query_id];
          delta_pair = dcg_gap * paired_discount * inverse_max_dcg;
        } else if (lambdarank_target_ == LambdaRankTarget::LAMBDALOSS_NDCG_PLUS_PLUS) {
          const int high_label = static_cast<int>(label[high]);
          const int low_label = static_cast<int>(label[low]);
          const double high_label_gain = label_gain_[high_label];
          const double high_discount = DCGCalculator::GetDiscount(high_rank);
          const double low_label_gain = label_gain_[low_label];
          const double low_discount = DCGCalculator::GetDiscount(low_rank);
          const int rank_diff = j - i;
          const double left_discount = DCGCalculator::GetDiscount(rank_diff);
          const double right_discount = DCGCalculator::GetDiscount(rank_diff + 1);
          const double inverse_max_dcg = inverse_max_dcgs_[query_id];
          // get dcg gap
          const double dcg_gap = high_label_gain - low_label_gain;
          // get discount of this pair
          const double paired_discount_lambdarank = fabs(high_discount - low_discount);
          const double paired_discount_lambdaloss = left_discount - right_discount;
          // get max DCG on current query & get delta NDCG
          delta_pair = dcg_gap * (paired_discount_lambdarank + lambdagap_weight_ * paired_discount_lambdaloss) * inverse_max_dcg;
        } else if (lambdarank_target_ == LambdaRankTarget::BNDCG) {
          const double high_discount = DCGCalculator::GetDiscount(high_rank);
          const double low_discount = DCGCalculator::GetDiscount(low_rank);
          const double paired_discount = fabs(high_discount - low_discount);
          delta_pair = paired_discount * inverse_max_bdcgs_[query_id];
        } else if (lambdarank_target_ == LambdaRankTarget::LAMBDALOSS_BNDCG) {
          const int rank_diff = j - i;
          const double left_discount = DCGCalculator::GetDiscount(rank_diff);
          const double right_discount = DCGCalculator::GetDiscount(rank_diff + 1);
          const double paired_discount = left_discount - right_discount;
          delta_pair = paired_discount * inverse_max_bdcgs_[query_id];
        } else if (lambdarank_target_ == LambdaRankTarget::LAMBDALOSS_BNDCG_PLUS_PLUS) {
          const double high_discount = DCGCalculator::GetDiscount(high_rank);
          const double low_discount = DCGCalculator::GetDiscount(low_rank);
          const double paired_discount_lambdarank = fabs(high_discount - low_discount);
          const int rank_diff = j - i;
          const double left_discount = DCGCalculator::GetDiscount(rank_diff);
          const double right_discount = DCGCalculator::GetDiscount(rank_diff + 1);
          const double paired_lambdaloss_discount = left_discount - right_discount;
          delta_pair = (paired_discount_lambdarank + lambdagap_weight_ * paired_lambdaloss_discount) * inverse_max_bdcgs_[query_id];
        } else if (lambdarank_target_ == LambdaRankTarget::PRECISION ||
                   lambdarank_target_ == LambdaRankTarget::LAMBDAGAP_S ||
                   lambdarank_target_ == LambdaRankTarget::LAMBDAGAP_X ||
                   lambdarank_target_ == LambdaRankTarget::BIN_RANKNET ||
                   lambdarank_target_ == LambdaRankTarget::RANKNET) {
          delta_pair = 1;
        } else if (lambdarank_target_ == LambdaRankTarget::LAMBDAGAP_S_PLUS) {
          delta_pair = (j - i == truncation_level_) * lambdagap_weight_ + (i < truncation_level_);
        } else if (lambdarank_target_ == LambdaRankTarget::LAMBDAGAP_X_PLUS) {
          delta_pair = (j - i >= truncation_level_) * lambdagap_weight_ + (i < truncation_level_);
        } else if (lambdarank_target_ == LambdaRankTarget::LAMBDAGAP_S_PLUS_PLUS) {
          delta_pair = (j - i == truncation_level_) * lambdagap_weight_ + (j + 1 - truncation_level_) - (i >= truncation_level_) * (i + 1 - truncation_level_);
        } else if (lambdarank_target_ == LambdaRankTarget::LAMBDAGAP_X_PLUS_PLUS) {
          delta_pair = (j - i >= truncation_level_) * lambdagap_weight_ + (j + 1 - truncation_level_) - (i >= truncation_level_) * (i + 1 - truncation_level_);
        } else if (lambdarank_target_ == LambdaRankTarget::ARP_K) {
          delta_pair = (j + 1 - truncation_level_) - (i >= truncation_level_) * (i + 1 - truncation_level_);
        } else if (lambdarank_target_ == LambdaRankTarget::LAMBDALOSS_ARP1) {
          delta_pair = static_cast<double>(label[high]);
        } else if (lambdarank_target_ == LambdaRankTarget::LAMBDALOSS_ARP2) {
          const double high_label = static_cast<double>(label[high]);
          const double low_label = static_cast<double>(label[low]);
          delta_pair = high_label - low_label;
        } else {
          Log::Fatal("LambdaRank target %d not implemented", lambdarank_target_);
        }

        if (delta_pair == 0) {
          continue;
        }

        // regular the delta_pair by score distance
        if (norm_ && best_score != worst_score) {
          delta_pair /= (0.01f + fabs(delta_score));
        }

        // calculate lambda for this pair
        double p_lambda = GetSigmoid(delta_score);
        double p_hessian = p_lambda * (1.0f - p_lambda);
        // update
        p_lambda *= -sigmoid_ * delta_pair;
        p_hessian *= sigmoid_ * sigmoid_ * delta_pair;
        lambdas[low] -= static_cast<score_t>(p_lambda);
        hessians[low] += static_cast<score_t>(p_hessian);
        lambdas[high] += static_cast<score_t>(p_lambda);
        hessians[high] += static_cast<score_t>(p_hessian);
        // lambda is negative, so use minus to accumulate
        sum_lambdas -= 2 * p_lambda;
        // update counter
        count_lambdas++;
      }
    }
    if (norm_ && sum_lambdas > 0) {
      double norm_factor = std::log2(1 + sum_lambdas) / sum_lambdas;
      for (data_size_t i = 0; i < cnt; ++i) {
        lambdas[i] = static_cast<score_t>(lambdas[i] * norm_factor);
        hessians[i] = static_cast<score_t>(hessians[i] * norm_factor);
      }
    }
    effective_pairs_[query_id] = 2.0 * static_cast<double>(count_lambdas) / (cnt * (cnt - 1));
  }

  inline double GetSigmoid(double score) const {
    if (score <= min_sigmoid_input_) {
      // too small, use lower bound
      return sigmoid_table_[0];
    } else if (score >= max_sigmoid_input_) {
      // too large, use upper bound
      return sigmoid_table_[_sigmoid_bins - 1];
    } else {
      return sigmoid_table_[static_cast<size_t>((score - min_sigmoid_input_) *
                                                sigmoid_table_idx_factor_)];
    }
  }

  void ConstructSigmoidTable() {
    // get boundary
    min_sigmoid_input_ = min_sigmoid_input_ / sigmoid_ / 2;
    max_sigmoid_input_ = -min_sigmoid_input_;
    sigmoid_table_.resize(_sigmoid_bins);
    // get score to bin factor
    sigmoid_table_idx_factor_ =
        _sigmoid_bins / (max_sigmoid_input_ - min_sigmoid_input_);
    // cache
    for (size_t i = 0; i < _sigmoid_bins; ++i) {
      const double score = i / sigmoid_table_idx_factor_ + min_sigmoid_input_;
      sigmoid_table_[i] = 1.0f / (1.0f + std::exp(score * sigmoid_));
    }
  }

  void UpdatePositionBiasFactors(const score_t* lambdas, const score_t* hessians) const override {
    /// get number of threads
    int num_threads = OMP_NUM_THREADS();
    // create per-thread buffers for first and second derivatives of utility w.r.t. position bias factors
    std::vector<double> bias_first_derivatives(num_position_ids_ * num_threads, 0.0);
    std::vector<double> bias_second_derivatives(num_position_ids_ * num_threads, 0.0);
    std::vector<int> instance_counts(num_position_ids_ * num_threads, 0);
    #pragma omp parallel for schedule(guided) num_threads(num_threads)
    for (data_size_t i = 0; i < num_data_; i++) {
      // get thread ID
      const int tid = omp_get_thread_num();
      size_t offset = static_cast<size_t>(positions_[i] + tid * num_position_ids_);
      // accumulate first derivatives of utility w.r.t. position bias factors, for each position
      bias_first_derivatives[offset] -= lambdas[i];
      // accumulate second derivatives of utility w.r.t. position bias factors, for each position
      bias_second_derivatives[offset] -= hessians[i];
      instance_counts[offset]++;
    }
    #pragma omp parallel for schedule(guided) num_threads(num_threads)
    for (data_size_t i = 0; i < num_position_ids_; i++) {
      double bias_first_derivative = 0.0;
      double bias_second_derivative = 0.0;
      int instance_count = 0;
      // aggregate derivatives from per-thread buffers
      for (int tid = 0; tid < num_threads; tid++) {
        size_t offset = static_cast<size_t>(i + tid * num_position_ids_);
        bias_first_derivative += bias_first_derivatives[offset];
        bias_second_derivative += bias_second_derivatives[offset];
        instance_count += instance_counts[offset];
      }
      // L2 regularization on position bias factors
      bias_first_derivative -= pos_biases_[i] * position_bias_regularization_ * instance_count;
      bias_second_derivative -= position_bias_regularization_ * instance_count;
      // do Newton-Raphson step to update position bias factors
      pos_biases_[i] += learning_rate_ * bias_first_derivative / (std::abs(bias_second_derivative) + 0.001);
    }
    LogDebugPositionBiasFactors();
  }

  void GetGradients(const double* score, score_t* gradients, score_t* hessians) const override {
    RankingObjective::GetGradients(score, gradients, hessians);
    iter_++;
    Log::Debug("Iteration %d completed", iter_);
  }

  const char* GetName() const override { return "lambdarank"; }

 protected:
  void LogDebugPositionBiasFactors() const {
    std::stringstream message_stream;
    message_stream << std::setw(15) << "position"
      << std::setw(15) << "bias_factor"
      << std::endl;
    Log::Debug(message_stream.str().c_str());
    message_stream.str("");
    for (int i = 0; i < num_position_ids_; ++i) {
      message_stream << std::setw(15) << position_ids_[i]
        << std::setw(15) << pos_biases_[i];
      Log::Debug(message_stream.str().c_str());
      message_stream.str("");
    }
  }
  /*! \brief Sigmoid param */
  double sigmoid_;
  /*! \brief Normalize the lambdas or not */
  bool norm_;
  /*! \brief Truncation position for max DCG */
  int truncation_level_;
  /*! \brief Cache inverse max DCG, speed up calculation */
  std::vector<double> inverse_max_dcgs_;
  /*! \brief Cache inverse max BDCG, speed up calculation */
  std::vector<double> inverse_max_bdcgs_;
  /*! \brief Cache result for sigmoid transform to speed up */
  std::vector<double> sigmoid_table_;
  /*! \brief Gains for labels */
  std::vector<double> label_gain_;
  /*! \brief Target metric of lambdarank */
  LambdaRankTarget lambdarank_target_;
  /*! \brief Number of bins in simoid table */
  size_t _sigmoid_bins = 1024 * 1024;
  /*! \brief Minimal input of sigmoid table */
  double min_sigmoid_input_ = -50;
  /*! \brief Maximal input of Sigmoid table */
  double max_sigmoid_input_ = 50;
  /*! \brief Factor that covert score to bin in Sigmoid table */
  double sigmoid_table_idx_factor_;
  /*! \brief Current iteration */
  mutable int iter_ = 0;
  /*! \brief LambdaGap weight */
  double lambdagap_weight_;
};

/*!
 * \brief Implementation of the learning-to-rank objective function, XE_NDCG
 * [arxiv.org/abs/1911.09798].
 */
class RankXENDCG : public RankingObjective {
 public:
  explicit RankXENDCG(const Config& config) : RankingObjective(config) {}

  explicit RankXENDCG(const std::vector<std::string>& strs)
      : RankingObjective(strs) {}

  ~RankXENDCG() {}

  void Init(const Metadata& metadata, data_size_t num_data) override {
    RankingObjective::Init(metadata, num_data);
    for (data_size_t i = 0; i < num_queries_; ++i) {
      rands_.emplace_back(seed_ + i);
    }
  }

  inline void GetGradientsForOneQuery(data_size_t query_id, data_size_t cnt,
                                      const label_t* label, const double* score,
                                      score_t* lambdas,
                                      score_t* hessians) const override {
    // Skip groups with too few items.
    if (cnt <= 1) {
      for (data_size_t i = 0; i < cnt; ++i) {
        lambdas[i] = 0.0f;
        hessians[i] = 0.0f;
      }
      return;
    }

    // Turn scores into a probability distribution using Softmax.
    std::vector<double> rho(cnt, 0.0);
    Common::Softmax(score, rho.data(), cnt);

    // An auxiliary buffer of parameters used to form the ground-truth
    // distribution and compute the loss.
    std::vector<double> params(cnt);

    double inv_denominator = 0;
    for (data_size_t i = 0; i < cnt; ++i) {
      params[i] = Phi(label[i], rands_[query_id].NextFloat());
      inv_denominator += params[i];
    }
    // sum_labels will always be positive number
    inv_denominator = 1. / std::max<double>(kEpsilon, inv_denominator);

    // Approximate gradients and inverse Hessian.
    // First order terms.
    double sum_l1 = 0.0;
    for (data_size_t i = 0; i < cnt; ++i) {
      double term = -params[i] * inv_denominator + rho[i];
      lambdas[i] = static_cast<score_t>(term);
      // Params will now store terms needed to compute second-order terms.
      params[i] = term / (1. - rho[i]);
      sum_l1 += params[i];
    }
    // Second order terms.
    double sum_l2 = 0.0;
    for (data_size_t i = 0; i < cnt; ++i) {
      double term = rho[i] * (sum_l1 - params[i]);
      lambdas[i] += static_cast<score_t>(term);
      // Params will now store terms needed to compute third-order terms.
      params[i] = term / (1. - rho[i]);
      sum_l2 += params[i];
    }
    for (data_size_t i = 0; i < cnt; ++i) {
      lambdas[i] += static_cast<score_t>(rho[i] * (sum_l2 - params[i]));
      hessians[i] = static_cast<score_t>(rho[i] * (1.0 - rho[i]));
    }
  }

  double Phi(const label_t l, double g) const {
    return Common::Pow(2, static_cast<int>(l)) - g;
  }

  const char* GetName() const override { return "rank_xendcg"; }

 protected:
  mutable std::vector<Random> rands_;
};

}  // namespace LightGBM
#endif  // LightGBM_OBJECTIVE_RANK_OBJECTIVE_HPP_
