#!/bin/bash
# Quick backup/restore helper for AWS Milvus
# Usage:
#   ./backup_restore_aws.sh backup   # Create backup and upload to S3
#   ./backup_restore_aws.sh restore <backup-file>  # Restore from S3

set -e

# Configuration
MILVUS_IP="${MILVUS_IP:-}"  # Set via environment or pass as argument
S3_BUCKET="${S3_BUCKET:-arxplorer-backups-prod}"
REGION="${AWS_REGION:-ca-central-1}"
SSH_KEY="${SSH_KEY:-~/.ssh/arxplorer-key.pem}"

if [ -z "$MILVUS_IP" ]; then
    echo "Error: MILVUS_IP not set"
    echo "Usage: MILVUS_IP=<ip> ./backup_restore_aws.sh backup|restore [backup-file]"
    exit 1
fi

case "$1" in
    backup)
        echo "Creating Milvus backup on AWS instance..."
        ssh -i "$SSH_KEY" ubuntu@"$MILVUS_IP" << 'ENDSSH'
cd ~
echo "Stopping Milvus..."
sudo docker compose down

echo "Creating backup..."
BACKUP_NAME="milvus-volumes-backup-$(date +%Y%m%d-%H%M%S).tar.gz"
sudo tar -czf "$BACKUP_NAME" -C /milvus-data volumes/

echo "Uploading to S3..."
aws s3 cp "$BACKUP_NAME" "s3://${S3_BUCKET:-arxplorer-backups-prod}/volumes/$BACKUP_NAME"

echo "Restarting Milvus..."
sudo docker compose up -d

echo "Cleaning up local backup..."
rm "$BACKUP_NAME"

echo "✓ Backup complete: $BACKUP_NAME"
ENDSSH
        ;;
    
    restore)
        if [ -z "$2" ]; then
            echo "Available backups:"
            aws s3 ls "s3://$S3_BUCKET/volumes/" --region "$REGION"
            echo ""
            echo "Usage: ./backup_restore_aws.sh restore <backup-filename>"
            exit 1
        fi
        
        BACKUP_FILE="$2"
        echo "Restoring Milvus from backup: $BACKUP_FILE"
        
        ssh -i "$SSH_KEY" ubuntu@"$MILVUS_IP" << ENDSSH
cd ~
echo "Stopping Milvus..."
sudo docker compose down

echo "Downloading backup from S3..."
aws s3 cp "s3://${S3_BUCKET}/volumes/${BACKUP_FILE}" .

echo "Removing old volumes..."
sudo rm -rf /milvus-data/volumes/*

echo "Extracting backup..."
sudo tar -xzf "${BACKUP_FILE}" -C /milvus-data/

echo "Restarting Milvus..."
sudo docker compose up -d

echo "Waiting for Milvus to start..."
sleep 30

echo "Checking Milvus status..."
sudo docker logs milvus-standalone --tail 20

echo "Cleaning up downloaded backup..."
rm "${BACKUP_FILE}"

echo "✓ Restore complete!"
ENDSSH
        ;;
    
    list)
        echo "Available backups in S3:"
        aws s3 ls "s3://$S3_BUCKET/volumes/" --region "$REGION"
        ;;
    
    status)
        echo "Checking Milvus status..."
        ssh -i "$SSH_KEY" ubuntu@"$MILVUS_IP" << 'ENDSSH'
sudo docker ps
echo ""
echo "Milvus logs:"
sudo docker logs milvus-standalone --tail 30
ENDSSH
        ;;
    
    *)
        echo "ArXplorer AWS Milvus Backup/Restore Tool"
        echo ""
        echo "Usage: MILVUS_IP=<ip> ./backup_restore_aws.sh <command> [args]"
        echo ""
        echo "Commands:"
        echo "  backup              Create backup and upload to S3"
        echo "  restore <filename>  Restore from S3 backup"
        echo "  list                List available backups in S3"
        echo "  status              Check Milvus status"
        echo ""
        echo "Environment Variables:"
        echo "  MILVUS_IP    IP address of Milvus instance (required)"
        echo "  S3_BUCKET    S3 bucket for backups (default: arxplorer-backups-prod)"
        echo "  AWS_REGION   AWS region (default: ca-central-1)"
        echo "  SSH_KEY      Path to SSH key (default: ~/.ssh/arxplorer-key.pem)"
        echo ""
        echo "Examples:"
        echo "  MILVUS_IP=15.156.207.51 ./backup_restore_aws.sh backup"
        echo "  MILVUS_IP=15.156.207.51 ./backup_restore_aws.sh list"
        echo "  MILVUS_IP=15.156.207.51 ./backup_restore_aws.sh restore milvus-volumes-backup-20251201-120000.tar.gz"
        exit 1
        ;;
esac
