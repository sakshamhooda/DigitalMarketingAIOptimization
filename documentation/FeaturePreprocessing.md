# Feature Engineering

Feature engineering is a crucial step in our AI-driven media investment plan optimization. We create several types of features to capture the complexities of ad performance and temporal patterns. This document outlines the feature engineering process and the rationale behind each feature.

## 1. Time-based Features

Time-based features help capture seasonal trends and periodic patterns in ad performance.

### Day of Week
```python
combined_data['Day_of_Week'] = combined_data['Date'].dt.dayofweek
```
Rationale: Ad performance often varies by day of the week (e.g., weekends vs. weekdays).

### Month
```python
combined_data['Month'] = combined_data['Date'].dt.month
```
Rationale: Captures monthly seasonality in ad performance.

### Quarter
```python
combined_data['Quarter'] = combined_data['Date'].dt.quarter
```
Rationale: Identifies quarterly trends, which can be important for businesses with seasonal cycles.

### Is Weekend
```python
combined_data['Is_Weekend'] = (combined_data['Day_of_Week'] >= 5).astype(int)
```
Rationale: Weekend performance often differs significantly from weekdays.

## 2. Rolling Average Features

Rolling averages help smooth out short-term fluctuations and highlight longer-term trends.

### 7-day Rolling Averages
```python
metrics = ['CTR', 'CPC', 'CVR', 'ROAS']
for metric in metrics:
    combined_data[f'{metric}_7day_avg'] = combined_data.groupby('Source')[metric].rolling(window=7, min_periods=1).mean().reset_index(0, drop=True)
```

### 30-day Rolling Averages
```python
for metric in metrics:
    combined_data[f'{metric}_30day_avg'] = combined_data.groupby('Source')[metric].rolling(window=30, min_periods=1).mean().reset_index(0, drop=True)
```
Rationale: These features capture recent performance trends and can help identify anomalies or shifts in performance.

## 3. Relative Performance Metrics

Relative performance metrics help compare the performance of a specific instance to the average performance for that channel.

```python
for metric in metrics:
    combined_data[f'{metric}_ratio'] = combined_data[metric] / combined_data.groupby('Source')[metric].transform('mean')
```

Rationale: These ratios provide context for performance metrics, allowing the model to understand if a particular day's performance is above or below average for that channel.

## 4. Channel Interaction Features

Channel interaction features capture the diversity of channel usage and potential synergies between channels.

### Channel Diversity
```python
def calculate_channel_diversity(group):
    total_spend = group['Spend'].sum()
    channel_shares = (group.groupby('Source')['Spend'].sum() / total_spend) ** 2
    return 1 - (channel_shares.sum() / len(channel_shares))

combined_data['Channel_Diversity'] = combined_data.groupby('Date').apply(calculate_channel_diversity).reset_index(level=0, drop=True)
```

Rationale: This feature measures how evenly the budget is distributed across channels. A higher diversity score indicates a more balanced distribution of spend across channels.

## 5. Lag Features

Lag features capture the relationship between current performance and past performance.

```python
lag_periods = [1, 7, 30]
for metric in ['Spend', 'Clicks', 'Conversions', 'Revenue']:
    for lag in lag_periods:
        combined_data[f'{metric}_lag_{lag}'] = combined_data.groupby('Source')[metric].shift(lag)
```

Rationale: These features allow the model to learn from past performance and identify trends or cyclical patterns.

## 6. Interaction Features

Interaction features capture the combined effect of multiple features.

```python
combined_data['Spend_CTR_Interaction'] = combined_data['Spend'] * combined_data['CTR']
combined_data['CPC_CVR_Interaction'] = combined_data['CPC'] * combined_data['CVR']
```

Rationale: These features can help the model capture non-linear relationships between features.

## 7. Categorical Encoding

For categorical variables like 'Source' and 'Campaign type', we use one-hot encoding:

```python
combined_data = pd.get_dummies(combined_data, columns=['Source', 'Campaign type'], prefix=['Src', 'Camp'])
```

Rationale: One-hot encoding allows the model to learn the impact of different sources and campaign types on performance.

[Placeholder for Feature Importance Plot]

These engineered features provide a rich set of inputs for our machine learning models, capturing various aspects of ad performance, temporal patterns, and channel interactions. The feature importance plot above shows the relative importance of these features in predicting ad performance.