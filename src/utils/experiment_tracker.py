import mlflow

def track_experiment(model, params, metrics):
    mlflow.set_experiment("ad_spend_optimization")
    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")