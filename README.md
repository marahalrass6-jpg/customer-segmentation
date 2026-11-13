# Customer Segmentation with RFM and K-Means

An end-to-end customer analytics project that segments retail customers by purchasing behavior using **Recency, Frequency, Monetary (RFM)** analysis and K-Means clustering.

## Business questions

- Which customers are high-value and loyal?
- Which customers are at risk because they have not purchased recently?
- Which customers are frequent but low-value buyers?
- How should marketing campaigns differ by segment?

## Methodology

1. Download the real UCI Online Retail II dataset.
2. Remove cancellations, invalid quantities/prices, and rows without customer identifiers.
3. Aggregate customer behavior into RFM features:
   - `recency_days`: days since the last purchase.
   - `frequency`: number of distinct invoices.
   - `monetary_value`: total revenue.
4. Apply `log1p` and `StandardScaler` to reduce skew and prevent high-spend customers from dominating all dimensions.
5. Compare K-Means values from 2 to 8 clusters using Silhouette Score and inertia.
6. Select the best `k`, profile clusters, assign interpretable segment names, and export campaign recommendations.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
python scripts/download_data.py
python scripts/run_segmentation.py
```

## Outputs

- `artifacts/customer_segments.csv`
- `artifacts/segment_profiles.csv`
- `artifacts/segment_recommendations.csv`
- `artifacts/cluster_selection.csv`
- `artifacts/rfm_distribution.png`
- `artifacts/elbow_silhouette.png`
- `artifacts/segment_scatter.png`
- `artifacts/segment_profile_heatmap.png`
- `artifacts/segmentation_report.json`
- `artifacts/customer_segmentation.joblib`

The generated README results section is updated with the actual customer count, selected cluster count, and Silhouette Score from the run.

## Interpretation

Segments are behavioral groups, not immutable customer identities. Re-run the analysis periodically, monitor segment drift, and validate campaign outcomes. RFM does not establish why a customer behaves a certain way; it is a prioritization tool for marketing and retention decisions.

## Verified run results

On `805,549` cleaned transactions and `5,878` customers, the pipeline evaluated `k=2..8` and selected `k=2` using the highest Silhouette Score of `0.4380`.

| Segment | Customers | Mean Recency (days) | Mean Frequency | Mean Monetary Value |
|---|---:|---:|---:|---:|
| At-Risk | 3,517 | 302.81 | 2.08 | 617.96 |
| High-Value Loyal | 2,361 | 51.50 | 12.56 | 6,594.68 |

These numbers are produced by the included run and may change if the source dataset or cleaning rules change.
