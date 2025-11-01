# ArXplorer Infrastructure Guide

## Overview

This document explains the Infrastructure as Code (IaC) setup for the ArXplorer academic search assistant using Terraform and AWS.

## Project Architecture

```
ArXplorer Infrastructure
├── Web Layer (Load Balancer + Compute)
├── Application Layer (Pipeline Processing)
├── Data Layer (Storage + Database)
└── Vector Search Layer (Embeddings + FAISS)
```

## Why Infrastructure as Code?

### Traditional Problems
- **Manual Setup**: Hours spent clicking through AWS console
- **Inconsistency**: Dev environment different from production
- **No Version Control**: Can't track infrastructure changes
- **Human Error**: Typos, missed configurations, security gaps
- **Difficult Scaling**: Hard to replicate for multiple environments

### Terraform Benefits
- **Declarative**: Describe what you want, Terraform figures out how
- **Version Controlled**: Infrastructure changes tracked in Git
- **Repeatable**: Same code creates identical environments
- **Planning**: See changes before applying them
- **State Management**: Knows what exists and what needs to change

## Team Division of Work

### Storage & Database Team (Your Focus)
**Responsibilities:**
- **Data Storage**: S3 buckets for arXiv dataset and processed files
- **Database Management**: RDS for paper metadata and search results
- **Vector Storage**: OpenSearch or vector database for FAISS indices
- **Backup & Recovery**: Automated backups and disaster recovery
- **Data Security**: Encryption, access controls, compliance

**Files You'll Create:**
- `modules/storage/` - S3 buckets, data lifecycle
- `modules/database/` - RDS instances, schemas
- `security.tf` - IAM roles, encryption
- `backup.tf` - Backup policies

### Compute & Networking Team (Teammate's Focus)
**Responsibilities:**
- **Compute Infrastructure**: EC2 instances, containers (ECS/EKS)
- **Pipeline Orchestration**: How ArXplorer pipeline runs
- **Load Balancing**: Traffic distribution and availability
- **Monitoring**: CloudWatch, alerts, logging
- **Auto-scaling**: Handle varying loads automatically
- **CI/CD**: Automated deployment pipeline

**Files They'll Create:**
- `modules/compute/` - EC2, containers
- `modules/networking/` - VPC, subnets, load balancers
- `monitoring.tf` - CloudWatch, alarms
- `pipeline-deployment.tf` - Application deployment

## Multi-Environment Strategy

### Development Environment
```
Purpose: Developer experimentation and testing
Resources: 
- t3.micro EC2 instances (free tier)
- Small RDS db.t3.micro
- 10GB S3 storage
- Subset of data (1,000 papers)
Use Case: Feature development, basic testing
```

### Staging Environment
```
Purpose: Pre-production testing and QA
Resources:
- t3.small EC2 instances
- RDS db.t3.small with Multi-AZ
- Full dataset (2M+ papers)
- Production-like configuration
Use Case: Integration testing, performance validation
```

### Production Environment
```
Purpose: Live system for real users
Resources:
- t3.medium+ EC2 instances with auto-scaling
- RDS db.t3.medium with Multi-AZ + Read Replicas
- Full dataset with automated backups
- CloudWatch monitoring and alerts
Use Case: Real student and researcher access
```

## Data Flow Architecture

### 1. Data Ingestion
```
Kaggle arXiv Dataset → S3 Raw Data Bucket
                    ↓
            ArXplorer Pipeline Processing
                    ↓
        S3 Processed Data Bucket + RDS Metadata
```

### 2. Search Processing
```
User Query → Load Balancer → Web Servers
                                ↓
                    Text Processing Pipeline
                                ↓
                Vector Search (FAISS/OpenSearch)
                                ↓
                    Results + Paper Metadata
```

### 3. Storage Layer
```
S3 Buckets:
├── arxplorer-raw-data (Kaggle dataset)
├── arxplorer-processed (Cleaned papers)
├── arxplorer-embeddings (Vector representations)
└── arxplorer-backups (Automated backups)

RDS Database:
├── papers_metadata (Title, authors, abstract)
├── processed_papers (Cleaned text, keywords)
├── embeddings_index (Vector indices)
└── search_cache (Cached results)
```

## Network Architecture

### VPC Design
```
ArXplorer VPC (10.0.0.0/16)
├── Public Subnets (10.0.1.0/24, 10.0.2.0/24)
│   ├── Load Balancer
│   └── NAT Gateway
└── Private Subnets (10.0.10.0/24, 10.0.20.0/24)
    ├── Application Servers
    ├── Database Instances
    └── Vector Search Engines
```

### Security Groups
```
Web Tier:
- Inbound: HTTP (80), HTTPS (443) from Internet
- Outbound: All traffic to App Tier

Application Tier:
- Inbound: HTTP (8000) from Web Tier
- Outbound: HTTPS (443) to Internet, MySQL (3306) to DB Tier

Database Tier:
- Inbound: MySQL (3306) from App Tier only
- Outbound: None
```

## Terraform Project Structure

```
terraform/
├── main.tf                 # Provider configuration
├── variables.tf           # Shared variables
├── outputs.tf            # Shared outputs
├── terraform.tfvars      # Variable values
├── environments/
│   ├── dev/
│   │   ├── main.tf       # Dev-specific config
│   │   └── terraform.tfvars
│   ├── staging/
│   │   ├── main.tf       # Staging-specific config
│   │   └── terraform.tfvars
│   └── prod/
│       ├── main.tf       # Production-specific config
│       └── terraform.tfvars
└── modules/
    ├── storage/          # Your modules
    │   ├── main.tf
    │   ├── variables.tf
    │   └── outputs.tf
    ├── database/         # Your modules
    │   ├── main.tf
    │   ├── variables.tf
    │   └── outputs.tf
    ├── compute/          # Teammate's modules
    │   ├── main.tf
    │   ├── variables.tf
    │   └── outputs.tf
    └── networking/       # Teammate's modules
        ├── main.tf
        ├── variables.tf
        └── outputs.tf
```

## Module Dependencies

### Dependency Flow
```
networking → compute
networking → database
storage → database
database → compute
```

### Shared Resources
```
Networking Module Provides:
- VPC ID
- Subnet IDs
- Security Group IDs

Storage Module Provides:
- S3 bucket names
- S3 bucket ARNs

Database Module Provides:
- Database endpoint
- Database connection string
- Database security group

Compute Module Uses:
- All of the above
```

## Development Workflow

### Parallel Development Strategy
1. **Phase 1 (Days 1-2)**: Foundation setup
   - Both: Create shared Terraform structure
   - Both: Define common variables and naming

2. **Phase 2 (Days 3-5)**: Independent module development
   - You: Storage and database modules
   - Teammate: Networking and compute modules

3. **Phase 3 (Days 6-7)**: Integration and testing
   - Both: Connect modules together
   - Both: Test complete infrastructure

### Daily Coordination
- **15-minute sync**: What's done, what's needed
- **Shared variables document**: Track interface requirements
- **Version control**: Separate branches until integration

## Security Considerations

### Data Protection
- **Encryption at Rest**: All S3 buckets and RDS instances encrypted
- **Encryption in Transit**: HTTPS/TLS for all communications
- **Access Control**: IAM roles with least privilege principle
- **Network Security**: Private subnets for sensitive resources

### Backup Strategy
- **RDS Automated Backups**: 7-day retention
- **S3 Versioning**: Protect against accidental deletion
- **Cross-Region Replication**: Disaster recovery
- **Point-in-Time Recovery**: Database restore capability

## Cost Optimization

### Free Tier Resources (Development)
- **EC2**: t3.micro instances (750 hours/month)
- **RDS**: db.t3.micro (750 hours/month)
- **S3**: 5GB storage, 20,000 GET requests
- **Data Transfer**: 1GB/month

### Production Scaling
- **Auto Scaling**: Scale up during peak usage
- **Reserved Instances**: 1-year commitment for cost savings
- **S3 Lifecycle Policies**: Move old data to cheaper storage
- **CloudWatch**: Monitor and optimize resource usage

## Monitoring and Alerting

### Key Metrics
- **Application Health**: HTTP response times, error rates
- **Database Performance**: Connection count, query performance
- **Storage Usage**: S3 storage growth, access patterns
- **Cost Monitoring**: Daily spend alerts

### Alert Thresholds
```
Critical Alerts:
- Database CPU > 80%
- Application errors > 5%
- Storage usage > 90%

Warning Alerts:
- Response time > 2 seconds
- Database connections > 80%
- Daily cost > budget threshold
```

## Deployment Process

### Environment Promotion
```
1. Developer commits code
2. Terraform plan in dev environment
3. Apply changes to dev
4. Test functionality
5. Promote to staging
6. QA testing in staging
7. Production deployment (with approval)
```

### Terraform Commands
```bash
# Initialize Terraform
terraform init

# Plan changes
terraform plan -var-file="dev.tfvars"

# Apply changes
terraform apply -var-file="dev.tfvars"

# Destroy environment (cleanup)
terraform destroy -var-file="dev.tfvars"
```

## Troubleshooting Guide

### Common Issues
1. **Permission Errors**: Check IAM policies and roles
2. **Resource Conflicts**: Ensure unique naming across environments
3. **State Lock**: Handle concurrent Terraform runs
4. **Network Connectivity**: Verify security groups and NACLs

### Debug Commands
```bash
# Validate Terraform syntax
terraform validate

# Format code consistently
terraform fmt

# Show current state
terraform show

# List all resources
terraform state list
```

## Next Steps

### Immediate Actions
1. **Create shared configuration files** (main.tf, variables.tf, outputs.tf)
2. **Set up environment-specific configurations**
3. **Begin module development in parallel**
4. **Test in development environment**

### Future Enhancements
- **Container orchestration** with ECS or EKS
- **API Gateway** for external integrations
- **CloudFront CDN** for global content delivery
- **Machine learning pipeline** with SageMaker

---

*This infrastructure supports the ArXplorer academic search assistant for CSC490 Part Two assignment.*