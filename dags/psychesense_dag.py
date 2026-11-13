from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator



default_args = {
    "owner":'psychesense',
    "retries":1,
    "retry_delay":timedelta(minutes=2)
}

with DAG(
    dag_id = "psychesense_pipeline",
    default_args=default_args,
    start_date=datetime(26,11,1),
    schedule="@weekly",
    catchup=False,
) as dag :
    
    clean_task = BashOperator(
        task_id="clean_data",
        bash_command="cd /project && python -m src.clean",
    )
    
    train_task = BashOperator(
        task_id="train_models",
        bash_command="cd /project && python -m src.train"
    )
    
    evaluate_task = BashOperator(
        task_id="evaluation_model",
        bash_command="cd /project && python -m src.evaluate"
    )
    
    clean_task >> train_task >> evaluate_task
    