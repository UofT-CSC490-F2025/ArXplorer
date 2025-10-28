"""
AWS S3 Storage Integration for ArXplorer Pipeline
Integrates existing pipeline with AWS S3 infrastructure
"""

import boto3
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from botocore.exceptions import ClientError, NoCredentialsError
import io
import pickle
import numpy as np

# Pipeline imports
from schemas import ArXivPaper, ProcessedPaper, PaperEmbedding


class S3StorageManager:
    """Manages data storage and retrieval from AWS S3 buckets"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize S3 storage manager
        
        Args:
            config: Configuration dictionary with S3 bucket names
        """
        self.config = config
        self.region = config.get('aws_region', 'ca-central-1')
        
        # S3 bucket names from Terraform
        self.buckets = {
            'raw_data': config['s3_buckets']['raw_data'],
            'processed': config['s3_buckets']['processed'], 
            'embeddings': config['s3_buckets']['embeddings'],
            'backups': config['s3_buckets']['backups']
        }
        
        # Initialize S3 client
        try:
            self.s3_client = boto3.client('s3', region_name=self.region)
            self.s3_resource = boto3.resource('s3', region_name=self.region)
            logging.info(f"S3 client initialized for region: {self.region}")
        except NoCredentialsError:
            logging.error("AWS credentials not found. Please configure AWS CLI or environment variables.")
            raise
        except Exception as e:
            logging.error(f"Failed to initialize S3 client: {e}")
            raise

    def upload_raw_papers(self, papers: List[ArXivPaper], batch_id: str) -> bool:
        """
        Upload raw arXiv papers to S3 raw data bucket
        
        Args:
            papers: List of ArXivPaper objects
            batch_id: Unique identifier for this batch
            
        Returns:
            bool: Success status
        """
        try:
            # Convert papers to JSON format
            papers_data = [paper.to_dict() for paper in papers]
            
            # Create file key with timestamp and batch ID
            timestamp = datetime.now().strftime("%Y/%m/%d")
            file_key = f"raw_papers/{timestamp}/batch_{batch_id}.json"
            
            # Upload to S3
            self.s3_client.put_object(
                Bucket=self.buckets['raw_data'],
                Key=file_key,
                Body=json.dumps(papers_data, indent=2, default=str),
                ContentType='application/json',
                Metadata={
                    'batch_id': batch_id,
                    'paper_count': str(len(papers)),
                    'upload_timestamp': datetime.now().isoformat()
                }
            )
            
            logging.info(f"Uploaded {len(papers)} raw papers to s3://{self.buckets['raw_data']}/{file_key}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to upload raw papers: {e}")
            return False

    def upload_processed_papers(self, papers: List[ProcessedPaper], batch_id: str) -> bool:
        """
        Upload processed papers to S3 processed data bucket
        
        Args:
            papers: List of ProcessedPaper objects
            batch_id: Unique identifier for this batch
            
        Returns:
            bool: Success status
        """
        try:
            # Convert papers to JSON format
            papers_data = [paper.to_dict() for paper in papers]
            
            # Create file key
            timestamp = datetime.now().strftime("%Y/%m/%d")
            file_key = f"processed_papers/{timestamp}/batch_{batch_id}.json"
            
            # Upload to S3
            self.s3_client.put_object(
                Bucket=self.buckets['processed'],
                Key=file_key,
                Body=json.dumps(papers_data, indent=2, default=str),
                ContentType='application/json',
                Metadata={
                    'batch_id': batch_id,
                    'paper_count': str(len(papers)),
                    'processing_timestamp': datetime.now().isoformat()
                }
            )
            
            logging.info(f"Uploaded {len(papers)} processed papers to s3://{self.buckets['processed']}/{file_key}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to upload processed papers: {e}")
            return False

    def upload_embeddings(self, embeddings: List[PaperEmbedding], batch_id: str) -> bool:
        """
        Upload paper embeddings to S3 embeddings bucket
        
        Args:
            embeddings: List of PaperEmbedding objects
            batch_id: Unique identifier for this batch
            
        Returns:
            bool: Success status
        """
        try:
            # Prepare embeddings data
            embeddings_data = []
            embedding_vectors = []
            
            for embedding in embeddings:
                embeddings_data.append({
                    'paper_id': embedding.arxiv_id,
                    'arxiv_id': embedding.arxiv_id,
                    'model_name': embedding.model_name,
                    'model_version': embedding.model_version,
                    'embedding_dimension': embedding.embedding_dimension,
                    'created_at': embedding.created_at.isoformat() if embedding.created_at else None
                })
                embedding_vectors.append(embedding.combined_embedding)
            
            # Convert to numpy array for efficient storage
            embeddings_matrix = np.array(embedding_vectors, dtype=np.float32)
            
            # Create file keys
            timestamp = datetime.now().strftime("%Y/%m/%d")
            metadata_key = f"embeddings/{timestamp}/batch_{batch_id}_metadata.json"
            vectors_key = f"embeddings/{timestamp}/batch_{batch_id}_vectors.npy"
            
            # Upload metadata
            self.s3_client.put_object(
                Bucket=self.buckets['embeddings'],
                Key=metadata_key,
                Body=json.dumps(embeddings_data, indent=2),
                ContentType='application/json'
            )
            
            # Upload vectors as numpy array
            buffer = io.BytesIO()
            np.save(buffer, embeddings_matrix)
            buffer.seek(0)
            
            self.s3_client.put_object(
                Bucket=self.buckets['embeddings'],
                Key=vectors_key,
                Body=buffer.getvalue(),
                ContentType='application/octet-stream',
                Metadata={
                    'batch_id': batch_id,
                    'embedding_count': str(len(embeddings)),
                    'dimensions': str(embeddings_matrix.shape[1]),
                    'upload_timestamp': datetime.now().isoformat()
                }
            )
            
            logging.info(f"Uploaded {len(embeddings)} embeddings to s3://{self.buckets['embeddings']}/{timestamp}/")
            return True
            
        except Exception as e:
            logging.error(f"Failed to upload embeddings: {e}")
            return False

    def download_raw_papers(self, date_prefix: str = None) -> List[ArXivPaper]:
        """
        Download raw papers from S3
        
        Args:
            date_prefix: Optional date prefix (YYYY/MM/DD) to filter files
            
        Returns:
            List[ArXivPaper]: Downloaded papers
        """
        try:
            papers = []
            prefix = f"raw_papers/{date_prefix}" if date_prefix else "raw_papers/"
            
            # List objects in bucket
            response = self.s3_client.list_objects_v2(
                Bucket=self.buckets['raw_data'],
                Prefix=prefix
            )
            
            for obj in response.get('Contents', []):
                if obj['Key'].endswith('.json'):
                    # Download and parse file
                    file_obj = self.s3_client.get_object(
                        Bucket=self.buckets['raw_data'],
                        Key=obj['Key']
                    )
                    data = json.loads(file_obj['Body'].read().decode('utf-8'))
                    
                    # Convert to ArXivPaper objects
                    batch_papers = [ArXivPaper(**paper_data) for paper_data in data]
                    papers.extend(batch_papers)
            
            logging.info(f"Downloaded {len(papers)} raw papers from S3")
            return papers
            
        except Exception as e:
            logging.error(f"Failed to download raw papers: {e}")
            return []

    def create_backup(self, source_bucket: str, backup_prefix: str) -> bool:
        """
        Create backup of data from source bucket to backup bucket
        
        Args:
            source_bucket: Name of source bucket
            backup_prefix: Prefix for backup files
            
        Returns:
            bool: Success status
        """
        try:
            # List all objects in source bucket
            response = self.s3_client.list_objects_v2(Bucket=source_bucket)
            
            backup_count = 0
            for obj in response.get('Contents', []):
                source_key = obj['Key']
                backup_key = f"{backup_prefix}/{datetime.now().strftime('%Y%m%d')}/{source_key}"
                
                # Copy object to backup bucket
                copy_source = {'Bucket': source_bucket, 'Key': source_key}
                self.s3_client.copy_object(
                    CopySource=copy_source,
                    Bucket=self.buckets['backups'],
                    Key=backup_key
                )
                backup_count += 1
            
            logging.info(f"Backed up {backup_count} objects from {source_bucket}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to create backup: {e}")
            return False

    def get_bucket_info(self) -> Dict[str, Any]:
        """
        Get information about all S3 buckets
        
        Returns:
            Dict with bucket information
        """
        bucket_info = {}
        
        for bucket_type, bucket_name in self.buckets.items():
            try:
                # Get bucket location
                location = self.s3_client.get_bucket_location(Bucket=bucket_name)
                
                # Count objects
                response = self.s3_client.list_objects_v2(Bucket=bucket_name)
                object_count = response.get('KeyCount', 0)
                
                # Calculate total size
                total_size = sum(obj.get('Size', 0) for obj in response.get('Contents', []))
                
                bucket_info[bucket_type] = {
                    'name': bucket_name,
                    'region': location.get('LocationConstraint', 'us-east-1'),
                    'object_count': object_count,
                    'total_size_bytes': total_size,
                    'total_size_mb': round(total_size / (1024 * 1024), 2)
                }
                
            except Exception as e:
                logging.error(f"Failed to get info for bucket {bucket_name}: {e}")
                bucket_info[bucket_type] = {'error': str(e)}
        
        return bucket_info


class CloudPipelineConfig:
    """Configuration for cloud-integrated pipeline"""
    
    @staticmethod
    def load_aws_config() -> Dict[str, Any]:
        """
        Load AWS configuration from Terraform outputs
        
        Returns:
            Dict with AWS configuration
        """
        # In a real setup, this would read from Terraform outputs or environment variables
        return {
            'aws_region': 'ca-central-1',
            's3_buckets': {
                'raw_data': 'arxplorer-dev-dev-raw-data-k9j69o8v',
                'processed': 'arxplorer-dev-dev-processed-k9j69o8v',
                'embeddings': 'arxplorer-dev-dev-embeddings-k9j69o8v',
                'backups': 'arxplorer-dev-dev-backups-k9j69o8v'
            }
        }


if __name__ == "__main__":
    # Test S3 integration
    logging.basicConfig(level=logging.INFO)
    
    config = CloudPipelineConfig.load_aws_config()
    storage = S3StorageManager(config)
    
    # Print bucket information
    info = storage.get_bucket_info()
    print(json.dumps(info, indent=2))