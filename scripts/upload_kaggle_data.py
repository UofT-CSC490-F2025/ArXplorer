import json
import boto3
from botocore.exceptions import ClientError

RAW_KEY = "test_data/sample_paper.json"
PROCESSED_KEY = "processed_data/sample_paper_processed.json"

def main():
    s3 = boto3.client("s3")

    buckets = {
        "raw_data": "arxplorer-dev-dev-raw-data-19v18ab2",
        "processed": "arxplorer-dev-dev-processed-19v18ab2"
    }

    sample_paper = {
        "id": "sample-001",
        "title": "Testing ArXplorer Infrastructure",
        "authors": ["Dev Teammate"],
        "abstract": "Placeholder abstract used to test S3 uploads.",
        "categories": ["cs.LG"],
        "submitted_date": "2025-01-01"
    }

    processed_paper = {
        **sample_paper,
        "cleaned_abstract": "Placeholder cleaned abstract.",
        "keywords": ["testing", "infrastructure"],
        "processed_at": "2025-01-02T12:00:00Z"
    }

    try:
        s3.put_object(
            Bucket=buckets["raw_data"],
            Key=RAW_KEY,
            Body=json.dumps(sample_paper),
            ContentType="application/json"
        )
        print(f"Uploaded sample paper to s3://{buckets['raw_data']}/{RAW_KEY}")

        s3.put_object(
            Bucket=buckets["processed"],
            Key=PROCESSED_KEY,
            Body=json.dumps(processed_paper),
            ContentType="application/json"
        )
        print(f"Uploaded processed sample to s3://{buckets['processed']}/{PROCESSED_KEY}")
    except ClientError as err:
        print(f"Upload failed: {err}")

if __name__ == "__main__":
    main()
