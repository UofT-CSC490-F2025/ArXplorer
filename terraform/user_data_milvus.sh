#!/bin/bash
# User data script for Milvus EC2 instance

set -e

# Log all output
exec > >(tee /var/log/user-data.log)
exec 2>&1

echo "Starting Milvus instance setup..."

# Update system
apt-get update
apt-get upgrade -y

# Install Docker
apt-get install -y \
    ca-certificates \
    curl \
    gnupg \
    lsb-release

mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null

apt-get update
apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# Install AWS CLI for S3 backups
apt-get install -y awscli

# Wait for EBS volume to be attached
echo "Waiting for EBS volume..."
while [ ! -e /dev/nvme1n1 ]; do
    sleep 5
done

# Format and mount EBS volume (only if not already formatted)
if ! blkid /dev/nvme1n1; then
    echo "Formatting EBS volume..."
    mkfs -t ext4 /dev/nvme1n1
fi

# Create mount point
mkdir -p /milvus-data

# Mount volume
mount /dev/nvme1n1 /milvus-data

# Add to fstab for automatic mounting on reboot
UUID=$(blkid -s UUID -o value /dev/nvme1n1)
echo "UUID=$UUID /milvus-data ext4 defaults,nofail 0 2" >> /etc/fstab

# Set permissions
chown -R ubuntu:ubuntu /milvus-data

# Create directories for Milvus
mkdir -p /milvus-data/volumes/etcd
mkdir -p /milvus-data/volumes/minio
mkdir -p /milvus-data/volumes/milvus
mkdir -p /milvus-data/backups

# Download Milvus docker-compose.yml
cd /home/ubuntu
curl -sfL https://raw.githubusercontent.com/milvus-io/milvus/master/deployments/docker/standalone/docker-compose.yml -o docker-compose.yml

# Modify docker-compose.yml to use EBS mount
sed -i 's|volumes:|volumes:\n      - /milvus-data/volumes/etcd:/etcd|g' docker-compose.yml
sed -i 's|volumes:|volumes:\n      - /milvus-data/volumes/minio:/minio_data|g' docker-compose.yml
sed -i 's|volumes:|volumes:\n      - /milvus-data/volumes/milvus:/var/lib/milvus|g' docker-compose.yml

# Update Milvus version
sed -i "s|milvusdb/milvus:.*|milvusdb/milvus:${milvus_version}|g" docker-compose.yml

# Start Milvus
echo "Starting Milvus..."
docker compose up -d

# Wait for Milvus to start
echo "Waiting for Milvus to initialize..."
sleep 60

# Install Milvus Backup tool
echo "Installing Milvus Backup..."
wget https://github.com/zilliztech/milvus-backup/releases/download/v0.4.0/milvus-backup_Linux_x86_64.tar.gz
tar -xvf milvus-backup_Linux_x86_64.tar.gz
mv milvus-backup /usr/local/bin/
chmod +x /usr/local/bin/milvus-backup

# Create Milvus Backup configuration
mkdir -p /etc/milvus-backup
cat > /etc/milvus-backup/backup.yaml <<EOF
# Milvus Backup Configuration
milvus:
  address: localhost
  port: 19530
  authorizationEnabled: false
  
minio:
  address: localhost
  port: 9000
  accessKeyID: minioadmin
  secretAccessKey: minioadmin
  useSSL: false
  bucketName: milvus-bucket
  rootPath: file
  useIAM: false
  
backup:
  maxBackupNum: 10
  backupPath: /milvus-data/backups
EOF

# Create backup script
cat > /usr/local/bin/backup-milvus.sh <<'EOF'
#!/bin/bash
set -e

BACKUP_NAME="milvus-backup-$(date +%Y%m%d-%H%M%S)"
BACKUP_DIR="/milvus-data/backups"
S3_BUCKET="${s3_bucket}"
AWS_REGION="${aws_region}"

echo "Creating Milvus backup: $BACKUP_NAME"

# Create backup using Milvus Backup tool
cd /etc/milvus-backup
/usr/local/bin/milvus-backup create -n "$BACKUP_NAME"

# Compress backup
echo "Compressing backup..."
cd "$BACKUP_DIR"
tar -czf "$BACKUP_NAME.tar.gz" "$BACKUP_NAME"

# Upload to S3
echo "Uploading to S3..."
aws s3 cp "$BACKUP_NAME.tar.gz" "s3://$S3_BUCKET/backups/$BACKUP_NAME.tar.gz" --region "$AWS_REGION"

# Cleanup old local backups (keep last 3)
echo "Cleaning up old backups..."
cd "$BACKUP_DIR"
ls -t *.tar.gz | tail -n +4 | xargs -r rm

echo "Backup complete: $BACKUP_NAME"
EOF

chmod +x /usr/local/bin/backup-milvus.sh

# Create restore script
cat > /usr/local/bin/restore-milvus.sh <<'EOF'
#!/bin/bash
set -e

if [ -z "$1" ]; then
    echo "Usage: restore-milvus.sh <backup-name>"
    echo "Available backups:"
    aws s3 ls s3://${s3_bucket}/backups/ --region ${aws_region}
    exit 1
fi

BACKUP_NAME="$1"
BACKUP_DIR="/milvus-data/backups"
S3_BUCKET="${s3_bucket}"
AWS_REGION="${aws_region}"

# Download from S3
echo "Downloading backup from S3..."
aws s3 cp "s3://$S3_BUCKET/backups/$BACKUP_NAME" "$BACKUP_DIR/$BACKUP_NAME" --region "$AWS_REGION"

# Extract backup
echo "Extracting backup..."
cd "$BACKUP_DIR"
tar -xzf "$BACKUP_NAME"

# Restore using Milvus Backup tool
BACKUP_NAME_NO_EXT=$(basename "$BACKUP_NAME" .tar.gz)
echo "Restoring Milvus from: $BACKUP_NAME_NO_EXT"
cd /etc/milvus-backup
/usr/local/bin/milvus-backup restore -n "$BACKUP_NAME_NO_EXT"

echo "Restore complete!"
EOF

chmod +x /usr/local/bin/restore-milvus.sh

# Schedule daily backups at 2 AM
echo "Setting up daily backup cron job..."
(crontab -l 2>/dev/null; echo "0 2 * * * /usr/local/bin/backup-milvus.sh >> /var/log/milvus-backup.log 2>&1") | crontab -

# Install CloudWatch agent (optional)
wget https://s3.amazonaws.com/amazoncloudwatch-agent/ubuntu/amd64/latest/amazon-cloudwatch-agent.deb
dpkg -i -E ./amazon-cloudwatch-agent.deb

echo "Milvus setup complete!"
echo "Data directory: /milvus-data"
echo "Backup directory: /milvus-data/backups"
echo "S3 bucket: ${s3_bucket}"

# Health check
docker ps
docker logs milvus-standalone --tail 50

echo "Milvus is ready!"
echo "Connection: localhost:19530"
