# RFM methodology

## Feature definitions

- **Recency:** days between the dataset snapshot date and the most recent customer invoice. Lower is better.
- **Frequency:** distinct invoice count. Higher generally indicates stronger engagement.
- **Monetary:** total valid line revenue. Higher indicates greater historical value.

The pipeline uses `log1p` before scaling because retail monetary and frequency distributions are heavily right-skewed. K-Means is fitted only on the transformed RFM matrix.

## Campaign policy

Segment names and recommendations are heuristic labels over cluster profiles. They are not causal conclusions and should be validated with campaign experiments.
