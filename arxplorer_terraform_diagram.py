from diagrams import Diagram, Cluster, Edge
from diagrams.aws.compute import EC2, ECS, AutoScaling
from diagrams.aws.database import Database
from diagrams.aws.network import ALB, InternetGateway, NATGateway, VPC
from diagrams.aws.storage import S3, EBS
from diagrams.aws.security import IAM
from diagrams.aws.management import Cloudwatch
from diagrams.aws.ml import Bedrock
from diagrams.aws.general import User

with Diagram("ArXplorer AWS Infrastructure", show=False, direction="TB", graph_attr={"splines": "ortho"}):
    users = User("Users")
    
    with Cluster("AWS VPC\n(Multi-AZ Architecture)"):
        igw = InternetGateway("Internet Gateway")
        
        with Cluster("Public Subnets"):
            with Cluster("AZ-1a"):
                alb = ALB("Application\nLoad Balancer")
                nat = NATGateway("NAT Gateway")
        
        with Cluster("Private App Subnet (AZ-1a)"):
            with Cluster("Auto Scaling Group"):
                search_api = AutoScaling("Search API\nInstances")
        
        with Cluster("Private ML Subnet (AZ-1b)"):
            with Cluster("ECS Fargate"):
                offline_workers = ECS("Offline Pipeline\nWorkers")
        
        with Cluster("Private Data Subnet (AZ-1a)"):
            milvus = EC2("Milvus Vector DB\n(c5.2xlarge)")
            ebs = EBS("EBS Volume\n(500GB gp3)")
    
    with Cluster("S3 Storage"):
        s3_raw = S3("Raw Data")
        s3_processed = S3("Processed Data")
        s3_embeddings = S3("Embeddings")
        s3_metadata = S3("Metadata")
        s3_backups = S3("Backups")
    
    with Cluster("AWS Managed Services"):
        bedrock = Bedrock("Amazon Bedrock\n(LLM Services)")
        cloudwatch = Cloudwatch("CloudWatch\nLogs & Monitoring")
        iam = IAM("IAM Roles\n& Policies")
    
    # Traffic Flow
    users >> Edge(label="HTTPS") >> igw >> alb
    alb >> Edge(label="HTTP") >> search_api
    search_api >> Edge(label="gRPC:19530") >> milvus
    search_api >> Edge(label="API Calls") >> bedrock
    
    # Offline Processing
    offline_workers >> Edge(label="gRPC:19530") >> milvus
    offline_workers >> Edge(label="Data Processing") >> [s3_raw, s3_processed, s3_embeddings, s3_metadata]
    
    # Storage Connections
    milvus >> Edge(label="Attached") >> ebs
    milvus >> Edge(label="Backups") >> s3_backups
    
    # Outbound Internet Access
    search_api >> Edge(label="Updates") >> nat >> igw
    offline_workers >> Edge(label="Updates") >> nat
    milvus >> Edge(label="Updates") >> nat
    
    # Monitoring & Security
    [search_api, offline_workers, milvus] >> Edge(label="Logs") >> cloudwatch
    iam >> Edge(label="Permissions", style="dashed") >> [search_api, offline_workers, milvus]