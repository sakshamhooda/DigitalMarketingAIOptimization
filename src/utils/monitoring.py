import logging
from prometheus_client import start_http_server, Summary

logger = logging.getLogger(__name__)

PREDICTION_TIME = Summary('prediction_latency_seconds', 'Time for predictions')

def setup_monitoring(port=8000):
    logging.basicConfig(level=logging.INFO)
    start_http_server(port)

@PREDICTION_TIME.time()
def monitored_predict(model, input_data):
    try:
        prediction = model.predict(input_data)
        logger.info(f"Prediction made: {prediction}")
        return prediction
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise