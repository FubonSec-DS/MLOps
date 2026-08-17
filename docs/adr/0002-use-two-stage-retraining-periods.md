---
status: accepted
---

# Use model-building and OOT backtest periods

Retraining uses two temporal periods: all configured historical snapshots form the model-building set, and the latest fully labelled snapshot is reserved for OOT backtesting after a label-window purge gap. We rejected a separate temporal validation month because it materially reduces the samples available for model building; formal training therefore uses fixed estimator settings without early stopping, and acceptance is based on the untouched OOT backtest.
