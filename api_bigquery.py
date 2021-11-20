from google.cloud import bigquery
from google.oauth2 import service_account

BYTES_PER_GB = 2**30
QUERY_LIMIT = 1*BYTES_PER_GB

def estimate_bytes_processed(query, bq_client):
    """A function to estimate query size without execution of query.
    Modified from: https://www.kaggle.com/sohier/beyond-queries-exploring-the-bigquery-api/
    """
    # Below we initialize the dry_run attribute to true so that our query
    # is not actually executed and does not use any of our precious quota
    estjob_config = bigquery.job.QueryJobConfig(dry_run=True)

    estjob = bq_client.query(query, job_config=estjob_config)

    return estjob.total_bytes_processed


creds_path = "secrets/ml-for-finance-296211-4c54724037a8.json"
credentials = service_account.Credentials.from_service_account_file(
    creds_path, scopes=["https://www.googleapis.com/auth/cloud-platform"])

#Establishes our connection with the bigQuery API
client = bigquery.Client(credentials=credentials, project=credentials.project_id)

# Queries written in SQL
query_smol = """
        SELECT
          timestamp,
          size,
          version,
          bits,
          transaction_count
        FROM `bigquery-public-data.crypto_bitcoin.blocks`
"""

query = """
        SELECT
          block_timestamp,
          is_coinbase,
          input_value,
          output_value,
          input_count,
          output_count,
          fee
        FROM `bigquery-public-data.crypto_bitcoin.transactions`
"""

estimate = estimate_bytes_processed(query_smol, client)
print(f"This query will process {estimate/BYTES_PER_GB:.3f} GB")

safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=QUERY_LIMIT)

query_job = client.query(query_smol, job_config=safe_config)

df = query_job.result().to_dataframe()
print(df)
print(df.memory_usage(deep=True))

# Save DataFrame to csv file
df.to_csv('data/blocks.csv', index=False)
