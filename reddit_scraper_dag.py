from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

# --- DAG CONFIGURATION ---
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=10),
}

dag = DAG(
    'reddit_scraper_hourly',
    default_args=default_args,
    description='Run Reddit scraper every hour',
    schedule_interval='@hourly',
    catchup=False,
)

run_scraper = BashOperator(
    task_id='run_reddit_scraper',
    bash_command='python3 /Users/satviknayak/work/projects/sentiment_trading/reddit_scraper.py',
    dag=dag,
)
