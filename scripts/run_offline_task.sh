#!/usr/bin/env bash
# Kick off a one-shot offline ECS task. Edit values if you changed names/regions.
set -euo pipefail

AWS_PROFILE=${AWS_PROFILE:-sandbox-sso}
AWS_REGION=${AWS_REGION:-ca-central-1}

CLUSTER="arxplorer-jjb08-offline"
TASK_DEF="arxplorer-jjb08-offline"
SUBNETS="subnet-0cba7c8d5a1f5c16f"         # private app subnet
SECURITY_GROUPS="sg-02b825ae044d1b307"     # offline_ingest SG

RAW_BUCKET="arxplorer-jjb08-prod-raw-data"
PROCESSED_BUCKET="arxplorer-jjb08-prod-processed"
METADATA_BUCKET="arxplorer-jjb08-prod-metadata"
EMBEDDINGS_BUCKET="arxplorer-jjb08-prod-embeddings"

MILVUS_HOST="10.0.4.123"
MILVUS_PORT="19530"

aws ecs run-task \
  --profile "$AWS_PROFILE" \
  --region "$AWS_REGION" \
  --cluster "$CLUSTER" \
  --task "$TASK_DEF" \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[$SUBNETS],securityGroups=[$SECURITY_GROUPS],assignPublicIp=DISABLED}" \
  --overrides "{\"containerOverrides\":[{\"name\":\"offline\",\"environment\":[
    {\"name\":\"AWS_REGION\",\"value\":\"$AWS_REGION\"},
    {\"name\":\"RAW_BUCKET\",\"value\":\"$RAW_BUCKET\"},
    {\"name\":\"PROCESSED_BUCKET\",\"value\":\"$PROCESSED_BUCKET\"},
    {\"name\":\"METADATA_BUCKET\",\"value\":\"$METADATA_BUCKET\"},
    {\"name\":\"EMBEDDINGS_BUCKET\",\"value\":\"$EMBEDDINGS_BUCKET\"},
    {\"name\":\"MILVUS_HOST\",\"value\":\"$MILVUS_HOST\"},
    {\"name\":\"MILVUS_PORT\",\"value\":\"$MILVUS_PORT\"}
  ]}]}"
