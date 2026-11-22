/*!
 * Copyright (c) 2020-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2020-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#ifndef LIGHTGBM_SRC_OBJECTIVE_RANK_OBJECTIVE_HPP_
#define LIGHTGBM_SRC_OBJECTIVE_RANK_OBJECTIVE_HPP_

#include <LightGBM/metric.h>
#include <LightGBM/objective_function.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <map>
#include <string>
#include <utility>
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
  LAMBDAGAP_S_PLUS_NMS,
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
  }

  void GetGradientsWithSampledQueries(const double* score, const data_size_t num_sampled_queries, const data_size_t* sampled_query_indices,
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

    if (num_position_ids_ > 0) {
      UpdatePositionBiasFactors(gradients, hessians);
    }
  }

  void GetGradients(const double* score, score_t* gradients, score_t* hessians) const override {
    GetGradientsWithSampledQueries(score, num_queries_, nullptr, gradients, hessians);
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
      {"lambdagap-x-plus-plus", LambdaRankTarget::LAMBDAGAP_X_PLUS_PLUS},
      {"lambdagap-s-plus-nms", LambdaRankTarget::LAMBDAGAP_S_PLUS_NMS}
    };
    if (lambdarank_target_map.count(config.lambdarank_target) == 0) {
      Log::Fatal("Unknown lambdarank target '%s'", config.lambdarank_target.c_str());
    } else {
      Log::Info("Using lambdarank objective with target '%s'", config.lambdarank_target.c_str());
    }
    lambdarank_target_ = lambdarank_target_map[config.lambdarank_target];
    // Targets where relevance is binary: a label above 0 or not
    binary_target_ = lambdarank_target_ != LambdaRankTarget::NDCG &&
                     lambdarank_target_ != LambdaRankTarget::LAMBDALOSS_NDCG &&
                     lambdarank_target_ != LambdaRankTarget::LAMBDALOSS_NDCG_PLUS_PLUS &&
                     lambdarank_target_ != LambdaRankTarget::LAMBDALOSS_ARP1 &&
                     lambdarank_target_ != LambdaRankTarget::LAMBDALOSS_ARP2 &&
                     lambdarank_target_ != LambdaRankTarget::RANKNET;
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
    if (lambdarank_target_ == LambdaRankTarget::LAMBDAGAP_S_PLUS_NMS) {
      // The rows kept as precision_nms keeps them with nms_group_mode none
      if (config.nms_group_mode != std::string("none")) {
        Log::Fatal("lambdagap-s-plus-nms needs nms_group_mode none");
      }
      if (config.nms_tolerance_mode == std::string("absolute")) {
        nms_relative_ = false;
      } else if (config.nms_tolerance_mode == std::string("relative")) {
        nms_relative_ = true;
      } else {
        Log::Fatal("Unknown nms_tolerance_mode: %s", config.nms_tolerance_mode.c_str());
      }
      nms_tolerance_ = config.nms_tolerance;
    }
  }

  explicit LambdarankNDCG(const std::vector<std::string>& strs)
      : RankingObjective(strs) {}

  ~LambdarankNDCG() {}

  void Init(const Metadata& metadata, data_size_t num_data) override {
    RankingObjective::Init(metadata, num_data);
    // The NMS fields are read at each round, as they may be set after the booster is created
    metadata_ = &metadata;
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

  void GetGradientsWithSampledQueries(const double* score, const data_size_t num_sampled_queries,
                                      const data_size_t* sampled_query_indices, score_t* gradients,
                                      score_t* hessians) const override {
    // Checked here, not in the parallel loop over the queries
    if (lambdarank_target_ == LambdaRankTarget::LAMBDAGAP_S_PLUS_NMS && metadata_->nms_coordinates() == nullptr) {
      Log::Fatal("lambdagap-s-plus-nms needs the nms_coordinate field");
    }
    RankingObjective::GetGradientsWithSampledQueries(score, num_sampled_queries, sampled_query_indices,
                                                     gradients, hessians);
  }

  /*!
  * \brief Greedy non-maximum suppression over rows sorted by decreasing score, as
  *        precision_nms with nms_group_mode none. Replaces sorted_idx by the rows kept, in
  *        order, and returns per kept position below the truncation level the first
  *        relevant row it suppressed, if the kept row is not relevant (else -1)
  */
  std::vector<data_size_t> SuppressNonMaxima(data_size_t start, const label_t* label, const double* score,
                                             std::vector<data_size_t>* sorted_idx) const {
    const double* coordinates = metadata_->nms_coordinates() + start;
    std::vector<data_size_t> hidden(std::min<size_t>(truncation_level_, sorted_idx->size()), -1);
    std::map<double, data_size_t> kept_at;  // Coordinate of each row kept, to its position
    std::vector<data_size_t> kept;
    for (data_size_t idx : *sorted_idx) {
      if (score[idx] == kMinScore) {
        continue;
      }
      const double x = coordinates[idx];
      // Every kept coordinate within the tolerance of x lies within w of it: with relative
      // distances, max(|x|, |c|) <= |x| / (1 - tolerance). Widened, the exact test decides
      double w = nms_tolerance_;
      if (nms_relative_) {
        w = nms_tolerance_ < 1.0 ? nms_tolerance_ * std::fabs(x) / (1.0 - nms_tolerance_)
                                 : std::numeric_limits<double>::infinity();
      }
      w += 1e-9 * (1.0 + std::fabs(x));
      data_size_t owner = -1;  // First kept position within the tolerance
      for (auto it = kept_at.lower_bound(x - w); it != kept_at.end() && it->first <= x + w; ++it) {
        const double c = it->first;
        const double scale = std::max(std::fabs(x), std::fabs(c));
        const double distance = nms_relative_ ? (scale == 0.0 ? 0.0 : std::fabs(x - c) / scale)
                                              : std::fabs(x - c);
        if (!(distance > nms_tolerance_) && (owner < 0 || it->second < owner)) {
          owner = it->second;
        }
      }
      if (owner < 0) {
        kept_at[x] = static_cast<data_size_t>(kept.size());
        kept.push_back(idx);
      } else if (owner < static_cast<data_size_t>(hidden.size()) && label[idx] > 0 && label[kept[owner]] <= 0 &&
                 hidden[owner] < 0) {
        hidden[owner] = idx;
      }
    }
    *sorted_idx = std::move(kept);
    return hidden;
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
    // The NMS target: LambdaGap-S+ on the rows NMS keeps, plus the pairs of SuppressNonMaxima
    const bool nms = lambdarank_target_ == LambdaRankTarget::LAMBDAGAP_S_PLUS_NMS;
    // get sorted indices for scores
    std::vector<data_size_t> sorted_idx(cnt);
    for (data_size_t i = 0; i < cnt; ++i) {
      sorted_idx[i] = i;
    }
    double best_score, worst_score;
    if (lambdarank_target_ == LambdaRankTarget::PRECISION ||
        lambdarank_target_ == LambdaRankTarget::BIN_RANKNET ||
        lambdarank_target_ == LambdaRankTarget::RANKNET ||
        lambdarank_target_ == LambdaRankTarget::LAMBDALOSS_ARP1 ||
        lambdarank_target_ == LambdaRankTarget::LAMBDALOSS_ARP2) {
      if (lambdarank_target_ == LambdaRankTarget::PRECISION) {
        // Only the top k apart from the rest. Clamped: a query may have fewer documents
        // than the truncation level. Equal scores by row order, as a stable sort: else
        // which tied rows enter the top k would depend on the standard library
        std::nth_element(
            sorted_idx.begin(), sorted_idx.begin() + std::min(truncation_level_, cnt) - 1,
            sorted_idx.end(), [score](data_size_t a, data_size_t b) {
              return score[a] > score[b] || (score[a] == score[b] && a < b);
            });
      }
      // The others need no order: their pair weights depend on neither i nor j
      best_score = *std::max_element(score, score + cnt);
      // As the sorted branch below, the worst score skips one row at kMinScore
      const double* worst = std::min_element(score, score + cnt);
      worst_score = *worst;
      if (cnt > 1 && worst_score == kMinScore) {
        worst_score = std::numeric_limits<double>::infinity();
        for (data_size_t i = 0; i < cnt; ++i) {
          if (score + i != worst) {
            worst_score = std::min(worst_score, score[i]);
          }
        }
      }
    } else {
      // The NMS target breaks ties by decreasing nms_tiebreak, then row order, as precision_nms
      const double* tiebreaks = nms ? metadata_->nms_tiebreaks() : nullptr;
      if (tiebreaks != nullptr) {
        tiebreaks += query_boundaries_[query_id];
        std::stable_sort(
            sorted_idx.begin(), sorted_idx.end(),
            [score, tiebreaks](data_size_t a, data_size_t b) {
              return score[a] != score[b] ? score[a] > score[b] : tiebreaks[a] > tiebreaks[b];
            });
      } else {
        std::stable_sort(
            sorted_idx.begin(), sorted_idx.end(),
            [score](data_size_t a, data_size_t b) { return score[a] > score[b]; });
      }
      best_score = score[sorted_idx[0]];
      data_size_t worst_idx = cnt - 1;
      if (worst_idx > 0 && score[sorted_idx[worst_idx]] == kMinScore) {
        worst_idx -= 1;
      }
      worst_score = score[sorted_idx[worst_idx]];
    }
    // From here on, the pairs are among the rows NMS keeps
    std::vector<data_size_t> hidden;
    if (nms) {
      hidden = SuppressNonMaxima(query_boundaries_[query_id], label, score, &sorted_idx);
    }
    const data_size_t n = static_cast<data_size_t>(sorted_idx.size());
    // get best and worst score
    double sum_lambdas = 0.0;

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
        i_end = std::min(n - 1, truncation_level_);
        break;
      default:
        i_end = n - 1;
    }

    for (data_size_t i = 0; i < i_end; ++i) {
      if (score[sorted_idx[i]] == kMinScore) {
        continue;
      }

      // Exclude pairs guaranteed to have 0 contribution to the gradient. With i < k, a
      // pair with j < k too is skipped by the precision targets: swapping two rows of the
      // top k does not change precision@k, so their (i < k) term only holds for j >= k.
      // i + truncation_level_ capped at n, so a huge truncation level cannot overflow
      const data_size_t i_plus_k = static_cast<data_size_t>(
          std::min<int64_t>(static_cast<int64_t>(i) + truncation_level_, n));
      data_size_t start;
      data_size_t end;
      switch (lambdarank_target_) {
        case LambdaRankTarget::PRECISION:
          start = truncation_level_;
          end = n;
          break;
        case LambdaRankTarget::LAMBDAGAP_S_PLUS:
        case LambdaRankTarget::LAMBDAGAP_S_PLUS_NMS:
          // Beyond the truncation level, only the pair at distance truncation_level_
          // has a non-zero weight
          if (i >= truncation_level_) {
            start = i_plus_k;
            end = std::min(start + 1, n);
          } else {
            start = truncation_level_;
            end = n;
          }
          break;
        case LambdaRankTarget::LAMBDAGAP_X_PLUS:
          // Beyond the truncation level, only pairs at distance truncation_level_ or
          // more have a non-zero weight
          start = i >= truncation_level_ ? i_plus_k : truncation_level_;
          end = n;
          break;
        case LambdaRankTarget::ARP_K:
        case LambdaRankTarget::LAMBDAGAP_S_PLUS_PLUS:
        case LambdaRankTarget::LAMBDAGAP_X_PLUS_PLUS:
          start = std::max(i + 1, truncation_level_);
          end = n;
          break;
        case LambdaRankTarget::LAMBDAGAP_S:
          start = i_plus_k;
          end = std::min(start + 1, n);
          break;
        case LambdaRankTarget::LAMBDAGAP_X:
          start = i_plus_k;
          end = n;
          break;
        default:
          start = i + 1;
          end = n;
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
        if (binary_target_ && label_i > 0 && label_j > 0) {
          continue;
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

        // The pieces of the NDCG-like pair weights, computed only by the targets using them
        const auto gain_gap = [&]() {
          return label_gain_[static_cast<int>(label[high])] - label_gain_[static_cast<int>(label[low])];
        };
        const auto lambdarank_discount = [&]() {
          return fabs(DCGCalculator::GetDiscount(high_rank) - DCGCalculator::GetDiscount(low_rank));
        };
        const auto lambdaloss_discount = [&]() {
          return DCGCalculator::GetDiscount(j - i) - DCGCalculator::GetDiscount(j - i + 1);
        };
        double delta_pair;
        if (lambdarank_target_ == LambdaRankTarget::NDCG) {
          delta_pair = gain_gap() * lambdarank_discount() * inverse_max_dcgs_[query_id];
        } else if (lambdarank_target_ == LambdaRankTarget::LAMBDALOSS_NDCG) {
          delta_pair = gain_gap() * lambdaloss_discount() * inverse_max_dcgs_[query_id];
        } else if (lambdarank_target_ == LambdaRankTarget::LAMBDALOSS_NDCG_PLUS_PLUS) {
          delta_pair = gain_gap() * (lambdarank_discount() + lambdagap_weight_ * lambdaloss_discount()) * inverse_max_dcgs_[query_id];
        } else if (lambdarank_target_ == LambdaRankTarget::BNDCG) {
          delta_pair = lambdarank_discount() * inverse_max_bdcgs_[query_id];
        } else if (lambdarank_target_ == LambdaRankTarget::LAMBDALOSS_BNDCG) {
          delta_pair = lambdaloss_discount() * inverse_max_bdcgs_[query_id];
        } else if (lambdarank_target_ == LambdaRankTarget::LAMBDALOSS_BNDCG_PLUS_PLUS) {
          delta_pair = (lambdarank_discount() + lambdagap_weight_ * lambdaloss_discount()) * inverse_max_bdcgs_[query_id];
        } else if (lambdarank_target_ == LambdaRankTarget::PRECISION ||
                   lambdarank_target_ == LambdaRankTarget::LAMBDAGAP_S ||
                   lambdarank_target_ == LambdaRankTarget::LAMBDAGAP_X ||
                   lambdarank_target_ == LambdaRankTarget::BIN_RANKNET ||
                   lambdarank_target_ == LambdaRankTarget::RANKNET) {
          delta_pair = 1;
        } else if (lambdarank_target_ == LambdaRankTarget::LAMBDAGAP_S_PLUS ||
                   lambdarank_target_ == LambdaRankTarget::LAMBDAGAP_S_PLUS_NMS) {
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
          Log::Fatal("LambdaRank target %d not implemented", static_cast<int>(lambdarank_target_));
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
      }
    }
    // The NMS target: a relevant row hidden by a kept row below the truncation level that is
    // not relevant goes above it, with the weight of a pair across the cutoff, 1 + lambdagap_weight
    for (size_t p = 0; p < hidden.size(); ++p) {
      if (hidden[p] < 0) {
        continue;
      }
      const data_size_t high = hidden[p];
      const data_size_t low = sorted_idx[p];
      const double delta_score = score[high] - score[low];
      double delta_pair = 1.0 + lambdagap_weight_;
      if (norm_ && best_score != worst_score) {
        delta_pair /= (0.01f + fabs(delta_score));
      }
      double p_lambda = GetSigmoid(delta_score);
      double p_hessian = p_lambda * (1.0f - p_lambda);
      p_lambda *= -sigmoid_ * delta_pair;
      p_hessian *= sigmoid_ * sigmoid_ * delta_pair;
      lambdas[low] -= static_cast<score_t>(p_lambda);
      hessians[low] += static_cast<score_t>(p_hessian);
      lambdas[high] += static_cast<score_t>(p_lambda);
      hessians[high] += static_cast<score_t>(p_hessian);
      sum_lambdas -= 2 * p_lambda;
    }
    if (norm_ && sum_lambdas > 0) {
      double norm_factor = std::log2(1 + sum_lambdas) / sum_lambdas;
      for (data_size_t i = 0; i < cnt; ++i) {
        lambdas[i] = static_cast<score_t>(lambdas[i] * norm_factor);
        hessians[i] = static_cast<score_t>(hessians[i] * norm_factor);
      }
    }
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
  LambdaRankTarget lambdarank_target_ = LambdaRankTarget::NDCG;
  /*! \brief Whether the target only tells labels of 0 from labels above 0 */
  bool binary_target_ = false;
  /*! \brief Number of bins in simoid table */
  size_t _sigmoid_bins = 1024 * 1024;
  /*! \brief Minimal input of sigmoid table */
  double min_sigmoid_input_ = -50;
  /*! \brief Maximal input of Sigmoid table */
  double max_sigmoid_input_ = 50;
  /*! \brief Factor that covert score to bin in Sigmoid table */
  double sigmoid_table_idx_factor_;
  /*! \brief LambdaGap weight */
  double lambdagap_weight_ = 0.0;
  /*! \brief For the NMS target: the dataset fields, the tolerance and its mode */
  const Metadata* metadata_ = nullptr;
  double nms_tolerance_ = 0.0;
  bool nms_relative_ = false;
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
#endif  // LIGHTGBM_SRC_OBJECTIVE_RANK_OBJECTIVE_HPP_
