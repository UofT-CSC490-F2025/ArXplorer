# Offline Pipeline (Ingest → Enrich → Embed → Milvus)

This directory contains a runnable pipeline for the offline processing flow:

1. Read arXiv/Kaggle metadata from S3 (`raw-data` bucket).
2. Filter to ML-related papers and normalize fields.
3. Enrich with citation counts via OpenAlex (optional, rate‑limited).
4. Build embedding text template.
5. Generate embeddings (dense: SPECTER2, sparse: SPLADEv3).
6. Upsert into Milvus and build indexes (IVF for dense, SPARSE_INVERTED for sparse).

## Expected AWS resources
- S3 buckets (raw-data, processed, metadata, embeddings) already created by Terraform.
- Milvus reachable at the private endpoint (see Terraform output `milvus_endpoint`).
- ECS Fargate cluster/service `arxplorer-jjb08-offline` created by Terraform (set the image in tfvars).

## How to run
1. Build and push the image to ECR:
   ```bash
   aws ecr create-repository --repository-name arxplorer-offline || true
   docker build -t arxplorer-offline:latest offline_pipeline
   docker tag arxplorer-offline:latest <account>.dkr.ecr.ca-central-1.amazonaws.com/arxplorer-offline:latest
   aws ecr get-login-password --region ca-central-1 | docker login --username AWS --password-stdin <account>.dkr.ecr.ca-central-1.amazonaws.com
   docker push <account>.dkr.ecr.ca-central-1.amazonaws.com/arxplorer-offline:latest
   ```
2. Update `terraform.tfvars`:
   ```
   offline_task_image = "<account>.dkr.ecr.ca-central-1.amazonaws.com/arxplorer-offline:latest"
   ```
   Then `terraform apply`.
3. Kick off a one-shot task (example):
   ```bash
   aws ecs run-task \
     --cluster arxplorer-jjb08-offline \
     --task arxplorer-jjb08-offline \
     --launch-type FARGATE \
     --network-configuration "awsvpcConfiguration={subnets=[<private-subnet-id>],securityGroups=[<offline_sg_id>],assignPublicIp=DISABLED}" \
     --overrides '{"containerOverrides":[{"name":"offline","environment":[
       {"name":"AWS_REGION","value":"ca-central-1"},
       {"name":"RAW_BUCKET","value":"arxplorer-jjb08-prod-raw-data"},
       {"name":"PROCESSED_BUCKET","value":"arxplorer-jjb08-prod-processed"},
       {"name":"METADATA_BUCKET","value":"arxplorer-jjb08-prod-metadata"},
       {"name":"EMBEDDINGS_BUCKET","value":"arxplorer-jjb08-prod-embeddings"},
       {"name":"MILVUS_HOST","value":"10.0.4.123"},
       {"name":"MILVUS_PORT","value":"19530"}
     ]}]}'
   ```

## Config (env vars)
- `AWS_REGION` – target region (ca-central-1).
- `RAW_BUCKET`, `PROCESSED_BUCKET`, `METADATA_BUCKET`, `EMBEDDINGS_BUCKET` – S3 buckets.
- `MILVUS_HOST`, `MILVUS_PORT` – Milvus endpoint.
- `DENSE_MODEL` (default `allenai/specter2_base`), `SPARSE_MODEL` (default `naver/splade-cocondenser-ensembledistil`).
- `BATCH_SIZE` (default 8), `MAX_SAMPLES` (optional, limit for testing).
- `OPENALEX_EMAIL` – passed to OpenAlex; `OPENALEX_RATE_LIMIT` (req/sec).

## Important notes
- Fargate has limited CPU/RAM; adjust `offline_task_cpu` / `offline_task_memory` in tfvars and embedding batch size accordingly.
- OpenAlex is rate‑limited; the pipeline backs off automatically but large crawls may be slow.
- Milvus stays private; tasks must run inside the VPC (already handled by Terraform).
