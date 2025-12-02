from diagrams import Diagram, Cluster, Edge
from diagrams.aws.compute import EC2, ECS, Batch, AutoScaling
from diagrams.aws.database import Database
from diagrams.aws.network import ALB
from diagrams.aws.storage import S3
from diagrams.aws.ml import Bedrock
from diagrams.aws.general import User
from diagrams.custom import Custom
import os

with Diagram("ArXplorer", show=False, direction="LR", graph_attr={"splines": "spline", "bgcolor": "white", "rankdir": "LR"}, filename="ArXplorer"):
    
    # Left: User Entry
    user = User("User Browser")
    
    with Cluster("AWS VPC", graph_attr={"bgcolor": "#e8f4fd", "style": "solid"}):
        with Cluster("Public Subnet", graph_attr={"bgcolor": "#fff2cc", "style": "dashed"}):
            alb = ALB("Application\nLoad Balancer")
        
        with Cluster("Private Subnet A - Application Layer", graph_attr={"bgcolor": "#e1f5fe", "style": "dashed"}):
            asg = AutoScaling("Search API\nAuto Scaling Group")
        
        with Cluster("Private Subnet B - ML Services Layer", graph_attr={"bgcolor": "#f3e5f5", "style": "dashed"}):
            mistral_icon = "./Mistral.png"
            bedrock = Custom("Mistral-7B\n(Query Expansion)", mistral_icon)
            specter2 = EC2("SPECTER2\n(Dense)")
            splade = EC2("SPLADE\n(Sparse)")
            reranker = EC2("Reranker\n(Jina)")
        
        with Cluster("Private Subnet C - Data Layer", graph_attr={"bgcolor": "#fff3e0", "style": "dashed"}):
            milvus = Database("Milvus\nVector DB")
            s3 = S3("S3 Buckets")
        
        with Cluster("Offline Ingestion", graph_attr={"bgcolor": "#f5f5f5", "style": "dashed"}):
            ingester = EC2("ArxivData\nIngester")
            batch = Batch("AWS Batch\nEmbeddings")
    
    # Flow: Left to Right
    user >> Edge(label="1. Request", color="black") >> alb
    alb >> Edge(label="2. Route", color="black") >> asg
    asg >> Edge(label="3. Expand", color="black") >> bedrock
    bedrock >> Edge(label="4a. Dense", color="black") >> specter2
    bedrock >> Edge(label="4b. Sparse", color="black") >> splade
    specter2 >> Edge(label="5. Search", color="black") >> milvus
    splade >> Edge(label="5. Search", color="black") >> milvus
    milvus >> Edge(label="6. Top-M", color="black") >> reranker
    
    # Flow: Right to Left (Return)
    reranker >> Edge(label="7. Top-K", color="green", style="dashed") >> asg
    asg >> Edge(label="8. Response", color="green", style="dashed") >> alb
    alb >> Edge(label="9. Result", color="green", style="dashed") >> user
    
    # Offline Flow
    ingester >> Edge(color="gray", style="dotted") >> s3
    batch >> Edge(color="gray", style="dotted") >> milvus
    batch >> Edge(color="gray", style="dotted") >> s3