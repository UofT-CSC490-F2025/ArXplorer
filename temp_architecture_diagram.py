from diagrams import Diagram, Cluster, Edge
from diagrams.aws.compute import EC2, ECS, AutoScaling, Batch
from diagrams.aws.database import Database
from diagrams.aws.network import ALB, InternetGateway, NATGateway
from diagrams.aws.storage import S3
from diagrams.aws.ml import SagemakerModel
from diagrams.aws.general import User

with Diagram("ML Pipeline Architecture from temp.PNG", show=False, direction="TB", graph_attr={"splines": "ortho"}):
    users = User("User Browser\nClient/Edge (Public)")
    
    with Cluster("AWS VPC"):
        # Public Layer
        alb = ALB("Application\nLoad Balancer")
        
        # Application Layer
        with Cluster("Search API Auto Scaling Group"):
            search_api1 = EC2("EC2 #1\n(FastAPI Search Service)")
            search_api2 = EC2("EC2 #2\n(FastAPI Search Service)")
        
        # ML Services Layer (Private Subnet B)
        with Cluster("ML Services Layer (Private Subnet B)"):
            qwen_rewriter = SagemakerModel("Qwen Rewriter")
            specter2_encoder = SagemakerModel("SPECTER2 Encoder\n(Dense Embeddings)")
            splade_encoder = SagemakerModel("SPLADE Encoder\n(Sparse Embeddings)")
            reranker = SagemakerModel("Reranker\n(Cross Encoder, Top-M candidates)")
        
        # Data Layer (Private Subnet C)
        with Cluster("Data Layer (Private Subnet C)"):
            milvus = Database("Milvus Vector DB (EC2 instance)\n- dense_collection\n- sparse_collection")
            
            with Cluster("S3 Buckets"):
                s3_raw = S3("raw-data\n(Kaggle/arXiv dumps)")
                s3_processed = S3("processed\n(clean JSONL)")
                s3_metadata = S3("metadata\n(per-doc JSON by doc_id)")
                s3_embeddings = S3("embeddings/backups\n(optional)")
        
        # Offline Processing
        with Cluster("Offline Ingestion Layer"):
            offline_batch = Batch("AWS Batch / ECS Task\n- Download Kaggle/arXiv\n- Parse title, abstract → processed\n- Split into per-doc metadata JSON")
    
    # Traffic Flow
    users >> Edge(label="HTTPS") >> alb
    alb >> Edge(label="Load Balance") >> [search_api1, search_api2]
    
    # Search API to ML Services
    search_api1 >> Edge(label="Query Processing") >> qwen_rewriter
    search_api2 >> Edge(label="Query Processing") >> qwen_rewriter
    qwen_rewriter >> Edge(label="Embeddings") >> [specter2_encoder, splade_encoder]
    
    # Search API to Database
    [search_api1, search_api2] >> Edge(label="Vector Search") >> milvus
    milvus >> Edge(label="Candidates") >> reranker
    
    # Offline Processing Flow
    offline_batch >> Edge(label="Process Data") >> [s3_raw, s3_processed, s3_metadata]
    offline_batch >> Edge(label="Run embeddings jobs\n(SPECTER2/SPLADE)\nUpload vectors into Milvus") >> milvus
    milvus >> Edge(label="Backups") >> s3_embeddings