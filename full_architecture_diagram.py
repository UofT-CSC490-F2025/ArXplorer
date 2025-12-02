from diagrams import Diagram, Cluster, Edge
from diagrams.aws.compute import EC2, ECS, AutoScaling
from diagrams.aws.database import Database
from diagrams.aws.network import ALB, InternetGateway, NATGateway, RouteTable
from diagrams.aws.storage import S3
from diagrams.aws.ml import SagemakerModel
from diagrams.aws.security import IAM
from diagrams.aws.general import User

with Diagram("Full ArXplorer Architecture (Online + Offline)", show=False, direction="TB", graph_attr={"splines": "ortho"}):
    user = User("User")
    
    with Cluster("AWS VPC"):
        igw = InternetGateway("IGW")
        
        with Cluster("Public Subnets"):
            alb = ALB("ALB\n(DNS Entrypoint)")
            nat = NATGateway("NAT")
        
        with Cluster("Private App Subnets"):
            with Cluster("Search API ASG"):
                with Cluster("Online Pipeline Stages"):
                    stage1 = SagemakerModel("Stage 1: Query Expansion\n(Mistral-7b-Instruct)")
                    stage2 = Database("Stage 2: Vector Search\n(SPECTER2 + SPLADE)")
                    stage3 = EC2("Stage 3: Metadata Rerank\n(Fuzzy Match + Boosting)")
                    stage4 = SagemakerModel("Stage 4: CrossEncoder\n(Jina-reranker-v3)")
                
                search_api = AutoScaling("Search API Container\n(Placeholder Image)")
        
        with Cluster("Private Data Subnets"):
            milvus = Database("Milvus EC2\n(No Public IP)\nPorts: 19530/9091")
            
            with Cluster("S3 Storage"):
                s3_raw = S3("Raw")
                s3_processed = S3("Processed")
                s3_metadata = S3("Metadata")
                s3_embeddings = S3("Embeddings")
                s3_backups = S3("Backups")
        
        with Cluster("Offline Pipeline (ECS Fargate)"):
            offline_worker = ECS("Offline Worker\n(Private Subnet)")
            arxiv_ingest = EC2("ArXiv/Kaggle\nIngest")
            openalex_enrich = EC2("OpenAlex\nEnrich")
            templates = EC2("Templates")
            sparse_embed = SagemakerModel("Sparse Embed\n(SPLADE)")
            dense_embed = SagemakerModel("Dense Embed\n(SPECTER2)")
    
    with Cluster("Security Groups"):
        alb_sg = IAM("ALB SG\n80/443 from Internet")
        app_sg = IAM("App SG\n8000 from ALB")
        milvus_sg = IAM("Milvus SG\n19530/9091 from App/Offline")
    
    # Entry/Return Flow
    user >> Edge(label="Request") >> igw >> alb
    alb >> Edge(label="Route") >> search_api
    
    # Online Pipeline Flow
    search_api >> Edge(label="1") >> stage1
    stage1 >> Edge(label="N queries + filters/intent") >> stage2
    stage2 >> Edge(label="Vector Search") >> milvus
    milvus >> Edge(label="Top-M") >> stage3
    stage3 >> Edge(label="Boosted") >> stage4
    stage4 >> Edge(label="Top-K") >> search_api
    search_api >> Edge(label="Response") >> alb >> user
    
    # Offline Pipeline Flow
    offline_worker >> arxiv_ingest >> openalex_enrich >> templates
    templates >> [sparse_embed, dense_embed]
    [sparse_embed, dense_embed] >> Edge(label="Upsert + Index") >> milvus
    offline_worker >> Edge(label="Writes") >> [s3_raw, s3_processed, s3_metadata, s3_embeddings, s3_backups]
    
    # NAT Egress
    offline_worker >> nat >> igw
    
    # Security Groups
    alb_sg >> alb
    app_sg >> search_api
    milvus_sg >> milvus