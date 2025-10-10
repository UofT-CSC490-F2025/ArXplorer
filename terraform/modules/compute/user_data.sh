#!/bin/bash
yum update -y
yum install -y python3 python3-pip git

# Install your teammate's ArXplorer pipeline
cd /home/ec2-user
git clone https://github.com/yourusername/arxplorer.git # Replace with actual repo URL later
cd arxplorer

# Install Python dependencies
pip3 install -r requirements.txt

# Set environment variables
echo "export DB_ENDPOINT=${database_endpoint}" >> /home/ec2-user/.bashrc
echo "export S3_RAW_BUCKET=${s3_bucket_raw}" >> /home/ec2-user/.bashrc
echo "export S3_PROCESSED_BUCKET=${s3_bucket_processed}" >> /home/ec2-user/.bashrc

# Start the pipeline service (placeholder)
# You'll replace this with actual pipeline startup commands
python3 -m http.server 8000 &
