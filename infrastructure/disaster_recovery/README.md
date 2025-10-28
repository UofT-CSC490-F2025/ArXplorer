# ArXplorer Disaster Recovery Demonstration

## Overview
This folder contains a comprehensive disaster recovery demonstration for the ArXplorer research paper processing system. The demonstration systematically destroys all infrastructure and data, then restores everything from backup to prove complete disaster recovery capabilities.

## 🚨 Part Three: Disaster Recovery Demonstration (50 marks)

This demonstration covers all required components for the CSC490 disaster recovery assessment:

### ✅ What Will Be Destroyed and Restored:

1. **Application Services**
   - Enhanced Cloud Pipeline (dual storage architecture)
   - MongoDB Atlas integration with async operations  
   - AWS S3 storage management
   - Configuration management system

2. **Database Systems and Their Data**
   - MongoDB Atlas cluster with all collections
   - Document data and metadata
   - Database indexes and configurations
   - User authentication and access controls

3. **Infrastructure as Code (IaaC)**
   - Terraform state files (.tfstate)
   - AWS S3 bucket infrastructure
   - Resource configurations and policies
   - Terraform lock files and modules

4. **Configuration Settings**
   - Application configuration (config.yaml)
   - Environment variables and settings
   - Connection strings and credentials
   - Processing parameters and thresholds

5. **Access Controls and Security Settings**
   - AWS IAM roles and policies
   - MongoDB Atlas user permissions
   - S3 bucket access controls
   - Network security configurations

6. **Verification of System Functionality**
   - Pipeline initialization and execution
   - Database connectivity and operations
   - Storage accessibility and permissions
   - Cross-service communication

## 📁 Demonstration Scripts

### `00_complete_demo.py` - Full Demonstration Orchestrator
- **Purpose**: Orchestrates the entire disaster recovery demonstration
- **Usage**: Run this for a complete screen recording demonstration
- **Features**: Automated timing, status checks, and demonstration flow

### `01_pre_disaster_backup.py` - Pre-Disaster Backup
- **Purpose**: Creates comprehensive backup of all systems
- **Backs Up**: S3 state, MongoDB data, Terraform state, configurations
- **Output**: Timestamped backup directory with recovery manifests

### `02_disaster_simulation.py` - Disaster Simulation  
- **Purpose**: Systematically destroys all infrastructure and data
- **Destroys**: S3 buckets, MongoDB collections, Terraform state, configurations
- **Safety**: Requires explicit confirmation before proceeding

### `03_disaster_recovery.py` - Disaster Recovery
- **Purpose**: Restores all systems from backup
- **Restores**: Infrastructure, databases, configurations, access controls
- **Verifies**: System functionality and data integrity

## 🎬 Screen Recording Instructions

### Option 1: Complete Automated Demo
```bash
python disaster_recovery/00_complete_demo.py
```
This runs the entire demonstration automatically with proper timing for screen recording.

### Option 2: Manual Step-by-Step
```bash
# Step 1: Create backup
python disaster_recovery/01_pre_disaster_backup.py

# Step 2: Simulate disaster (destroys everything!)
python disaster_recovery/02_disaster_simulation.py

# Step 3: Recover from disaster
python disaster_recovery/03_disaster_recovery.py
```

## ⚠️ Important Safety Notes

1. **Real Destruction**: This demonstration actually destroys your infrastructure - it's not simulated!
2. **AWS Costs**: Recreating infrastructure may incur small AWS charges
3. **MongoDB Data**: All collections will be completely dropped
4. **Terraform State**: State files will be deleted (but backed up)
5. **Configuration**: Config files will be corrupted to simulate loss

## 🛡️ Prerequisites

Before running the demonstration:

1. **AWS Credentials**: Ensure AWS CLI is configured with proper permissions
2. **MongoDB Access**: Verify MongoDB Atlas connection is working
3. **Terraform**: Ensure Terraform is installed and accessible
4. **Python Dependencies**: Install required packages from requirements.txt
5. **Backup Space**: Ensure sufficient disk space for backups
6. **Screen Recording**: Set up screen recording software

## 📊 Expected Results

### Recovery Time Objectives (RTO)
- **Configuration Recovery**: < 1 minute
- **Infrastructure Recreation**: < 5 minutes
- **Database Restoration**: < 2 minutes  
- **Full System Verification**: < 3 minutes
- **Total Recovery Time**: < 11 minutes

### Success Criteria
- ✅ All S3 buckets recreated with proper permissions
- ✅ MongoDB collections restored with correct indexes
- ✅ Terraform state fully restored and consistent
- ✅ Application configuration completely recovered
- ✅ Pipeline functionality verified end-to-end
- ✅ Security settings and access controls restored

## 📋 Demonstration Output

The demonstration generates:
- **Backup Manifests**: Detailed pre-disaster system state
- **Recovery Reports**: Post-recovery verification results
- **Demo Summary**: Comprehensive demonstration report for submission
- **Timestamped Logs**: Complete audit trail of all operations

## 🎯 Marking Criteria Alignment

This demonstration addresses all marking criteria:

| Criteria | Coverage |
|----------|----------|
| **Completeness of Deletion** | All systems systematically destroyed with verification |
| **Restoration Process** | Complete restoration with automated verification |
| **Clarity of Process/Code** | Well-documented scripts with clear output |
| **Application Services** | Enhanced pipeline fully restored and tested |
| **Database Systems** | MongoDB Atlas completely recreated with data |
| **Configuration Settings** | All configs backed up and restored |
| **Access Controls** | Security settings verified post-recovery |
| **System Functionality** | End-to-end pipeline testing included |

## 🚀 Getting Started

1. **Review Prerequisites**: Ensure all requirements are met
2. **Start Screen Recording**: Begin recording your screen
3. **Run Complete Demo**: Execute `python disaster_recovery/00_complete_demo.py`
4. **Follow Prompts**: Respond to confirmation prompts as needed
5. **Document Results**: Save the generated demonstration report

The demonstration will clearly show the complete destruction and restoration of your ArXplorer infrastructure, providing compelling evidence of your disaster recovery capabilities for the CSC490 assessment.

---
**⚠️ This is a real disaster recovery test - ensure you're prepared before starting!**