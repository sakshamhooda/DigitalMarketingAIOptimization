# DigitalMarketingAIOptimization

/project_root
│
├── data/
│   ├── raw/
│   │   ├── googleads-performance.csv
│   │   ├── metaads-performance.csv
│   │   ├── microsoftads-performance.csv
│   │   ├── merged-ads-performance.csv
|   |   └── website_landing.csv
│   ├── processed/
│   └── synthetic/
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_development.ipynb
│   ├── 05_model_evaluation.ipynb
│   ├── 06_synthetic_data_generation.ipynb
│   └── main.ipynb
│
├── src/
│   ├── data/
│   │   ├── data_loader.py
│   │   └── data_preprocessor.py
│   ├── features/
│   │   └── feature_engineering.py
│   ├── models/
│   │   ├── model_trainer.py
│   │   └── model_evaluator.py
│   ├── synthetic/
│   │   └── synthetic_data_generator.py
│   └── utils/
│       └── visualization.py
│
├── app/
│   ├── dashboard/
│   │   ├── app.py
│   │   └── components/
│   ├── api/
│   │   └── main.py
│   └── requirements.txt
│
├── tests/
│   ├── test_data_loader.py
│   ├── test_model_trainer.py
│   └── ...
│
├── config/
│   └── config.yaml
│
├── requirements.txt
├── README.md
└── .gitignore