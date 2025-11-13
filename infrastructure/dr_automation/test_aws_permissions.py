#!/usr/bin/env python3
"""
AWS Student Account Permission Checker
Test what services are available in student AWS account
"""

import boto3
from botocore.exceptions import ClientError, NoCredentialsError

def test_aws_permissions():
    """Test various AWS services to see what's accessible"""
    
    print("🔍 AWS STUDENT ACCOUNT PERMISSION CHECK")
    print("=" * 45)
    print()
    
    # We'll use environment variables or default credentials
    try:
        session = boto3.Session(region_name='us-east-2')  # Based on your error message
        
        # Test S3 (most important for disaster recovery)
        print("📦 Testing S3 Access...")
        try:
            s3_client = session.client('s3')
            buckets = s3_client.list_buckets()
            print(f"✅ S3: Found {len(buckets['Buckets'])} buckets")
            for bucket in buckets['Buckets'][:3]:  # Show first 3
                print(f"   - {bucket['Name']}")
        except ClientError as e:
            print(f"❌ S3: {e.response['Error']['Code']}")
        
        # Test IAM (for user info)
        print("\n👤 Testing IAM Access...")
        try:
            iam_client = session.client('iam')
            user = iam_client.get_user()
            print(f"✅ IAM: User {user['User']['UserName']}")
        except ClientError as e:
            print(f"❌ IAM: {e.response['Error']['Code']}")
        
        # Test STS (for current identity)
        print("\n🆔 Testing STS Access...")
        try:
            sts_client = session.client('sts')
            identity = sts_client.get_caller_identity()
            print(f"✅ STS: User ARN {identity['Arn']}")
            print(f"✅ Account ID: {identity['Account']}")
        except ClientError as e:
            print(f"❌ STS: {e.response['Error']['Code']}")
        
        # Test EC2 (for regions)
        print("\n🌍 Testing EC2 Access...")
        try:
            ec2_client = session.client('ec2')
            regions = ec2_client.describe_regions()
            print(f"✅ EC2: Found {len(regions['Regions'])} regions")
        except ClientError as e:
            print(f"❌ EC2: {e.response['Error']['Code']}")
            
    except NoCredentialsError:
        print("❌ No AWS credentials found!")
        print("Please set up your credentials first.")
        return False
    except Exception as e:
        print(f"❌ General error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("This will test what AWS services you can access.")
    print("First, you need to set your AWS credentials as environment variables:")
    print()
    print("In PowerShell, run these commands:")
    print('$Env:AWS_ACCESS_KEY_ID="YOUR_ACCESS_KEY"')
    print('$Env:AWS_SECRET_ACCESS_KEY="YOUR_SECRET_KEY"') 
    print('$Env:AWS_SESSION_TOKEN="YOUR_SESSION_TOKEN"')
    print('$Env:AWS_DEFAULT_REGION="us-east-2"')
    print()
    
    input("Press Enter after setting credentials to test permissions...")
    test_aws_permissions()