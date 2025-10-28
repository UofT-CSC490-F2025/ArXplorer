# Quick Setup for Teammate

## Prerequisites Done ✅
- AWS IAM user created by teammate
- Access keys generated
- Terraform folder structure ready

## Your Next Steps:

### 1. Configure AWS CLI
```powershell
aws configure
```
Enter the credentials your teammate provided:
- AWS Access Key ID: [from teammate]
- AWS Secret Access Key: [from teammate]
- Default region name: ca-central-1
- Default output format: json

### 2. Test AWS Access
```powershell
aws sts get-caller-identity
```
Should show Account: 487432950445 with your username

### 3. Test Terraform
```powershell
cd terraform
terraform --version
terraform validate
```

### 4. Start with Networking Module
```powershell
cd modules\networking
```
Follow the detailed guide in `TEAMMATE_SETUP_GUIDE.md`

### 5. Coordination Points

**What you need from storage/database teammate:**
- Storage bucket names (will be available after they build storage module)  
- Database endpoint (will be available after they build database module)
- Database security group requirements

**What you provide to storage/database teammate:**
- vpc_id (from your networking module)
- private_subnet_ids (from your networking module) 
- database_security_group_id (from your networking module)

### 6. Development Strategy

**Phase 1 (Independent Work):**
- You: Build networking and compute modules
- Teammate: Build storage and database modules
- Both: Test modules independently with mock dependencies

**Phase 2 (Integration):**
- Connect real module outputs/inputs
- Test complete infrastructure together
- Deploy to dev environment

### 7. Daily Sync
- Morning: "What did you complete? Any blockers?"
- Evening: "What outputs do you need from my modules?"

### 8. Success Criteria

You'll know your networking/compute work is done when:
- [ ] VPC and subnets created successfully
- [ ] Security groups allow proper traffic flow  
- [ ] Load balancer accessible from internet
- [ ] EC2 instances can connect to teammate's database
- [ ] `terraform plan` shows no errors for your modules

## Quick Reference

**Your AWS Account ID:** 487432950445
**Region:** ca-central-1  
**Your modules:** networking, compute
**Teammate's modules:** storage, database

**File to start with:** `TEAMMATE_SETUP_GUIDE.md` (has all the code you need)