# CIS 5200 Project TODO List
## Project: Get Rich Quick - Stock Movement Prediction with Multimodal Data
**Team: Schuylkill River Trading**

---

## 📅 Timeline Overview

### Phase 1: Proposal & Setup (Oct 20 - Oct 31)
### Phase 2: Data & Feature Engineering (Nov 1 - Nov 5)
### Phase 3: Model Development (Nov 6 - Nov 15)
### Phase 4: Advanced Models & Tuning (Nov 16 - Nov 20)
### Phase 5: Checkpoint (Nov 21 - Nov 24)
### Phase 6: Analysis & Interpretation (Nov 25 - Nov 30)
### Phase 7: Final Deliverables (Dec 1 - Dec 8)

---

## 👤 Rafael's Tasks

### Data Pipeline & Infrastructure
- [ ] Set up yfinance data collection for DJIA stocks
- [ ] Implement Twitter/X API scraper for financial keywords
- [ ] Set up FinBERT model and sentiment scoring pipeline
- [ ] Ensure point-in-time data integrity (no look-ahead bias)
- [ ] Create data preprocessing and feature alignment system
- [ ] Merge sentiment features with market data by date

### Evaluation Framework
- [ ] Implement walk-forward validation splits
- [ ] Build metric calculation system (Accuracy, Precision, Recall, F1, ROC-AUC, Log Loss)
- [ ] Implement ranking metrics (Spearman, Kendall-τ, NDCG)
- [ ] Create backtesting framework for top-k trading strategy
- [ ] Build system to track cumulative returns, Sharpe ratio, max drawdown
- [ ] Set up evaluation pipeline for all models

### Neural Network
- [ ] Design initial multimodal neural network architecture
- [ ] Implement separate subnetworks for numerical and textual features
- [ ] Implement differentiable ranking loss (soft Spearman or Kendall surrogate)
- [ ] Add auxiliary BCE head with light weighting
- [ ] Implement L2 regularization and early stopping
- [ ] Add daily feature normalization (z-scoring)
- [ ] Tune hyperparameters and validate on ranking metrics

### Writing (Final Report)
- [ ] Write Dataset section
- [ ] Write Evaluation section
- [ ] Write Conclusion section
- [ ] Contribute to Abstract
- [ ] Contribute to Motivation
- [ ] Final editing pass

---

## 👤 Monica's Tasks

### Model Development - Baselines
- [ ] Implement Logistic Regression baseline with L2 regularization
- [ ] Implement binary cross-entropy loss function
- [ ] Train and tune Logistic Regression model
- [ ] Evaluate Logistic Regression on test set
- [ ] Implement Random Forest classifier
- [ ] Tune Random Forest hyperparameters (n_estimators, max_depth, min_samples_leaf)
- [ ] Train and evaluate Random Forest model
- [ ] Compare RF vs Logistic Regression performance

### Loss Functions & Metrics Design
- [ ] Help design custom loss functions for classification task
- [ ] Collaborate on defining model comparison metrics
- [ ] Assist with probability calibration analysis
- [ ] Create calibration plots and Brier score calculations

### Neural Network Collaboration
- [ ] Test and validate neural network outputs
- [ ] Help tune neural network hyperparameters
- [ ] Contribute to model ensemble strategy

### Writing (Final Report)
- [ ] Write Problem Formulation section
- [ ] Write Methods section (Logistic Regression + Random Forest)
- [ ] Contribute to Abstract
- [ ] Contribute to Motivation
- [ ] Final editing pass

---

## 👤 Kylie's Tasks

### Model Development - Advanced Models
- [ ] Implement XGBoost classifier with binary:logistic objective
- [ ] Configure early stopping on validation set
- [ ] Tune XGBoost hyperparameters
- [ ] Map XGBoost probabilities to confidence buckets (Strong Up, Up, Neutral, Down, Strong Down)
- [ ] Train and evaluate XGBoost model
- [ ] Implement ranking model formulation
- [ ] Train ranking model with Spearman/Kendall loss
- [ ] Evaluate ranking model performance

### Interpretability & Analysis
- [ ] Compute feature importances for tree-based models
- [ ] Generate SHAP values for XGBoost and Random Forest
- [ ] Create visualizations of feature importance
- [ ] Plot correlations between sentiment and next-day returns
- [ ] Analyze keyword-level effects (e.g., "beat", "miss", "recall")
- [ ] Create ranking stability plots (Spearman correlation over time)
- [ ] Generate ROC and precision-recall curves
- [ ] Create calibration plots

### Backtesting Support
- [ ] Help Rafael with backtesting analysis
- [ ] Analyze strategy performance metrics
- [ ] Create performance visualization plots

### Neural Network Collaboration
- [ ] Test and validate neural network outputs
- [ ] Help tune neural network hyperparameters
- [ ] Contribute to model ensemble strategy

### Writing (Final Report)
- [ ] Write Methods section (XGBoost + Ranking Model)
- [ ] Write Related Work section
- [ ] Create all visualizations for report
- [ ] Contribute to Abstract
- [ ] Contribute to Motivation
- [ ] Final editing pass

---

## 👥 Shared Tasks (All Team Members)

### Model Ensemble & Integration
- [ ] Collaborate on improving the neural network architecture
- [ ] Implement model output combining strategies
- [ ] Test static ensembling (weighted average, stacking)
- [ ] Implement adaptive weighting scheme
- [ ] Compare ensemble performance

### Ablation Studies
- [ ] Run models with numerical features only
- [ ] Run models with textual (FinBERT) sentiment only
- [ ] Run models with combined multimodal features
- [ ] Compare and analyze ablation results

### Checkpoint Report (Due Nov 21-24)
- [ ] Draft checkpoint report collaboratively
- [ ] Include preliminary results and analysis
- [ ] Review and edit checkpoint together
- [ ] Submit checkpoint report

### Final Presentation (Dec 4-5)
- [ ] Create presentation slides (Dec 1-3)
- [ ] Prepare demo/visualizations
- [ ] Rehearse presentation
- [ ] Present during recitation session

### Final Report & Code (Due Dec 6-8)
- [ ] Integrate all written sections
- [ ] Proofread and edit entire report
- [ ] Create final Jupyter notebook with all experiments
- [ ] Clean and comment code
- [ ] Write comprehensive README
- [ ] Submit final report and notebook

---

## 📊 Milestones Checklist

- [ ] **Oct 27**: Proposal submitted ✓
- [ ] **Oct 31**: TA check-in completed, data pipeline initialized
- [ ] **Nov 5**: Feature engineering complete
- [ ] **Nov 10**: Baseline models (LR, RF) trained
- [ ] **Nov 15**: Neural network baseline complete, all metrics tested
- [ ] **Nov 20**: Model tuning and ranking experiments complete
- [ ] **Nov 24**: Checkpoint report submitted
- [ ] **Nov 30**: Interpretability analysis complete
- [ ] **Dec 3**: Presentation slides ready
- [ ] **Dec 5**: Presentation delivered
- [ ] **Dec 8**: Final report and code submitted

---

## 📝 Notes

### Dataset Sources
- Yahoo Finance (yfinance) - DJIA daily market data (2008-2016)
- Kaggle "Daily News for Stock Prediction" - Top 25 headlines per day
- Twitter/X API - Financial keywords and sentiment
- FinBERT - Pre-trained financial sentiment model

### Key Technical Details
- Use walk-forward validation (train on earlier, test on later)
- No look-ahead bias in feature engineering
- Daily feature normalization for cross-sectional patterns
- Binary classification: predict up (1) or down (0)
- Ranking: order stocks by expected performance

### Communication
- Regular team check-ins to sync progress
- Share code via Git repository
- Document experiments and findings
- Collaborate on writing using shared LaTeX/Overleaf document

