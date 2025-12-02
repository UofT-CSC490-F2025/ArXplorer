from diagrams import Diagram, Cluster, Edge
from diagrams.aws.compute import EC2, ECS, AutoScaling
from diagrams.aws.database import Database
from diagrams.aws.network import ALB, InternetGateway, NATGateway, VPC
from diagrams.aws.storage import S3, EBS
from diagrams.aws.security import IAM
from diagrams.aws.management import Cloudwatch
from diagrams.aws.ml import Bedrock
from diagrams.aws.general import User
from diagrams.custom import Custom

with Diagram("ArXplorer Terraform Infrastructure", show=False, direction="LR", graph_attr={"splines": "spline", "bgcolor": "white"}, filename="ArXplorer_Terraform"):
    
    # Left: User Entry
    user = User("Users")
    igw = InternetGateway("Internet\nGateway")
    
    with Cluster("AWS VPC (Multi-AZ)", graph_attr={"bgcolor": "#e8f4fd", "style": "solid"}):
        
        with Cluster("Public Subnets (AZ-1a, AZ-1b)", graph_attr={"bgcolor": "#fff2cc", "style": "dashed"}):
            alb = ALB("Application\nLoad Balancer")
            nat = NATGateway("NAT\nGateway")
        
        with Cluster("Private App Subnet (AZ-1a)", graph_attr={"bgcolor": "#e1f5fe", "style": "dashed"}):
            with Cluster("Auto Scaling Group"):
                search_api = AutoScaling("Search API\nEC2 Instances")
        
        with Cluster("Private ML Subnet (AZ-1b)", graph_attr={"bgcolor": "#f3e5f5", "style": "dashed"}):
            mistral_icon = "./Mistral.png"
            bedrock = Custom("Mistral-7B\n(Query Expansion)", mistral_icon)
            specter2 = EC2("SPECTER2\n(Dense)")
            splade = EC2("SPLADE\n(Sparse)")
            reranker = EC2("Jina Reranker\n(Top-K)")
            
            with Cluster("ECS Fargate"):
                offline_workers = ECS("Offline Pipeline\nWorkers")
        
        with Cluster("Private Data Subnet (AZ-1a)", graph_attr={"bgcolor": "#fff3e0", "style": "dashed"}):
            milvus = EC2("Milvus Vector DB\n(c5.2xlarge)")
            ebs = EBS("EBS Volume\n(500GB gp3)")
        
        with Cluster("S3 Storage", graph_attr={"bgcolor": "#e8f5e8", "style": "dashed"}):
            s3_raw = S3("Raw Data")
            s3_processed = S3("Processed")
            s3_embeddings = S3("Embeddings")
            s3_metadata = S3("Metadata")
            s3_backups = S3("Backups")
    
    # Right: Security & Monitoring (outside VPC)
    with Cluster("Security Groups", graph_attr={"bgcolor": "#fff8e1", "style": "dashed"}):
        alb_sg = IAM("ALB SG\n80/443")
        api_sg = IAM("API SG\n8000")
        milvus_sg = IAM("Milvus SG\n19530/9091")
        offline_sg = IAM("Offline SG\nSSH")
    
    with Cluster("Monitoring & IAM", graph_attr={"bgcolor": "#e1f5fe", "style": "dashed"}):
        cloudwatch = Cloudwatch("CloudWatch")
        iam = IAM("IAM Roles")
    
    # Online Flow: Left to Right
    user >> Edge(label="1. HTTPS", color="black") >> igw >> alb
    alb >> Edge(label="2. Route", color="black") >> search_api
    search_api >> Edge(label="3. Expand", color="black") >> bedrock
    bedrock >> Edge(label="4. Embed", color="black") >> [specter2, splade]
    [specter2, splade] >> Edge(label="5. Search", color="black") >> milvus
    milvus >> Edge(label="6. Top-M", color="black") >> reranker
    
    # Return Flow: Right to Left
    reranker >> Edge(label="7. Top-K", color="green", style="dashed") >> search_api
    search_api >> Edge(label="8. Response", color="green", style="dashed") >> alb >> igw >> user
    
    # Offline Flow
    offline_workers >> Edge(label="Ingest", color="blue", style="dotted") >> [s3_raw, s3_processed, s3_metadata]
    offline_workers >> Edge(label="Embed", color="blue", style="dotted") >> [specter2, splade]
    [specter2, splade] >> Edge(label="Upsert", color="blue", style="dotted") >> milvus
    
    # Storage Connections
    milvus >> Edge(color="gray") >> ebs
    milvus >> Edge(label="Backup", color="gray") >> s3_backups
    offline_workers >> Edge(color="blue", style="dotted") >> s3_embeddings
    
    # NAT for Outbound
    [search_api, offline_workers, milvus] >> Edge(color="orange", style="dotted") >> nat >> igw
    
    # Security Groups (from right)
    alb_sg >> Edge(color="gold", style="dashed") >> alb
    api_sg >> Edge(color="gold", style="dashed") >> search_api
    milvus_sg >> Edge(color="gold", style="dashed") >> milvus
    offline_sg >> Edge(color="gold", style="dashed") >> offline_workers
    
    # Monitoring & IAM (from right)
    [search_api, offline_workers, milvus] >> Edge(color="cyan", style="dotted") >> cloudwatch
    iam >> Edge(color="gold", style="dashed") >> [search_api, offline_workers, milvus]