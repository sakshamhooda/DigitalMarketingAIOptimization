# Data Preprocessing

## Data Sources

Our project integrates data from four main sources:

1. Google Ads performance data
2. Meta Ads performance data
3. Microsoft Ads performance data
4. Website landing and conversion data

## Preprocessing Pipeline

### 1. Data Loading

We use pandas to load the CSV files:

```python
google_ads = pd.read_csv('data/raw/googleads-performance.csv')
meta_ads = pd.read_csv('data/raw/metaads-performance.csv')
microsoft_ads = pd.read_csv('data/raw/microsoftads-performance.csv')
website_landings = pd.read_csv('data/raw/website_landing.csv')
```

### 2. Data Cleaning and Normalization

For each dataset:

- Convert 'Date' column to datetime format
- Add 'Source' column to identify the data origin
- Ensure consistent column names across datasets

```python
google_ads['Date'] = pd.to_datetime(google_ads['Date'])
google_ads['Source'] = 'Google Ads'

meta_ads['Date'] = pd.to_datetime(meta_ads['Date'])
meta_ads['Source'] = 'Meta Ads'
meta_ads['Campaign type'] = 'Cross-network'

microsoft_ads['Date'] = pd.to_datetime(microsoft_ads['Date'])
microsoft_ads['Source'] = 'Microsoft Ads'
```

### 3. Data Integration

Combine ad performance data from all sources:

```python
ad_performance = pd.concat([google_ads, meta_ads, microsoft_ads], ignore_index=True)
```

### 4. Website Landings Data Preprocessing

Clean and aggregate website landings data:

```python
website_landings['Website Landing Time'] = pd.to_datetime(website_landings['Website Landing Time'])
website_landings['Date'] = website_landings['Website Landing Time'].dt.date
website_landings['Date'] = pd.to_datetime(website_landings['Date'])
website_landings['Campaign type'] = website_landings['Campaign Type'].fillna('Unknown')

website_landings_agg = website_landings.groupby(['Date', 'Source', 'Channel', 'Campaign type']).agg({
    'User Id': 'count',
    'Is Converted': 'sum'
}).reset_index()

website_landings_agg = website_landings_agg.rename(columns={
    'User Id': 'Sessions',
    'Is Converted': 'Website Conversions'
})
```

### 5. Data Merging

Join ad performance data with aggregated website landings data:

```python
combined_data = pd.merge(
    ad_performance,
    website_landings_agg,
    on=['Date', 'Source', 'Campaign type'],
    how='left'
)
```

### 6. Handling Missing Values

Fill NaN values with 0 for metrics that should always have a value:

```python
metrics_to_fill = ['Impressions', 'Clicks', 'Spend', 'Conversions', 'Revenue', 'Sessions', 'Website Conversions']
combined_data[metrics_to_fill] = combined_data[metrics_to_fill].fillna(0)
```

### 7. Calculating Derived Metrics

Calculate additional performance metrics:

```python
combined_data['CTR'] = combined_data['Clicks'] / combined_data['Impressions']
combined_data['CPC'] = combined_data['Spend'] / combined_data['Clicks']
combined_data['CVR'] = combined_data['Conversions'] / combined_data['Clicks']
combined_data['ROAS'] = combined_data['Revenue'] / combined_data['Spend']
```

### 8. Final Data Cleaning

Replace infinity and NaN values with 0 in calculated metrics:

```python
combined_data = combined_data.replace([np.inf, -np.inf], np.nan).fillna(0)
```

## Output

The preprocessed data is saved as a CSV file for further analysis and modeling:

```python
combined_data.to_csv('data/processed/combined_ad_data.csv', index=False)
```

[Placeholder for Data Distribution Plot]

This preprocessing pipeline ensures that our data is clean, integrated, and ready for feature engineering and model development.