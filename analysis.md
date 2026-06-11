# Exploratory Data Analysis

## Dataset Overview

### Leads Dataset
- Rows: 2045
- Columns: 21

### Interactions Dataset
- Rows: 40000
- Columns: 36

## Missing Values

### Leads
- browser: 101
- company_size: 101
- annual_revenue_band: 42

### Interactions
- page_name: 829
- browser: 1906
- utm_source: 7572

## Conversion Distribution

- Converted: 267
- Not Converted: 1778

The dataset is imbalanced, therefore F1 Score and Recall were used as primary evaluation metrics.

## Behavioral Signals

Important features:
- session_count
- total_duration
- pricing_page_view
- return_visitor
- Evaluation
- Decision

Event-based features:

- pricing_page_view
- blog_read
- webinar_registration
- page_view
- document_download

Funnel stage features:

- Awareness
- Consideration
- Evaluation
- Decision

## Model Comparison

| Model | Accuracy | F1 Score | AUC ROC |
|---------|---------|---------|---------|
| Logistic Regression | 94.87% | 82.05% | 98.46% |
| Random Forest | 96.09% | 86.44% | 99.04% |
| XGBoost | 94.38% | 79.28% | 98.58% |

## Model Results

Best Model: Random Forest

Accuracy: 96.09%
F1 Score: 86.44%
AUC ROC: 99.04%
## Feature Importance Analysis

Random Forest feature importance revealed that behavioral engagement metrics were the strongest predictors of lead conversion.
## Key EDA Findings

1. Most interactions occur in the Awareness and Consideration stages.
2. Leads reaching Evaluation and Decision stages show higher conversion intent.
3. Pricing page visits, webinar registrations, and blog engagement are common among converted leads.
4. Return visitors demonstrate higher likelihood of conversion.
5. Multi-session users spend significantly more time on the platform.

Top predictive features:

1. Decision funnel stage
2. Pricing page views
3. Blog article reads
4. Total clicks
5. Webinar registrations
6. Total session duration
7. Average session duration
8. Evaluation funnel stage
9. Page views
10. Session count

These findings suggest that leads progressing deeper into the sales funnel and interacting with high-intent content (pricing pages, webinars, blogs) are significantly more likely to convert.

## Key Business Insights

1. Leads reaching the Decision stage have the highest probability of conversion.
2. Pricing page views are a strong indicator of buying intent.
3. Webinar registrations significantly increase conversion likelihood.
4. Higher session duration and repeat visits correlate with conversions.
5. Deep funnel engagement is more predictive than demographic attributes.