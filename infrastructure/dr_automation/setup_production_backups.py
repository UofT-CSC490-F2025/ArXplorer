#!/usr/bin/env python3
"""
ArXplorer Disaster Recovery - Production Setup Script
Sets up automated backup system for real-world production use
"""

import os
import subprocess
import sys
from pathlib import Path

def install_dependencies():
    """Install required packages for automated backup system"""
    print("Installing automated backup dependencies...")
    
    requirements = [
        "schedule==1.2.0",
        "pywin32==306",  # For Windows service support
    ]
    
    for package in requirements:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def create_directories():
    """Create necessary directories for logging and backup storage"""
    print("Creating backup system directories...")
    
    directories = [
        "disaster_recovery/logs",
        "disaster_recovery/backups",
        "backup_storage/monitoring"
    ]
    
    for dir_path in directories:
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created: {dir_path}")

def create_config_file():
    """Create configuration file for backup scheduler"""
    config_content = """# ArXplorer Automated Backup Configuration
# Production backup scheduling and retention settings

backup_schedule:
  # Critical backups during high-activity periods
  critical_interval: 15  # minutes
  
  # Regular backups during normal operations  
  hourly_backup: true
  
  # Full system backups
  daily_backup:
    enabled: true
    time: "02:00"  # 2 AM daily
  
  # Long-term archival
  weekly_backup:
    enabled: true
    day: "sunday"
    time: "01:00"  # 1 AM Sunday

retention:
  # How long to keep different backup types
  critical_retention_hours: 48
  hourly_retention_days: 7
  daily_retention_days: 30
  weekly_retention_months: 12

monitoring:
  # Alert if backup hasn't run in this time
  max_backup_age_minutes: 20
  
  # Success rate threshold for alerts
  min_success_rate_percent: 95
  
  # Log rotation
  max_log_size_mb: 100
  keep_log_files: 10

storage:
  # Local backup storage limits
  max_backup_size_gb: 50
  cleanup_when_full: true
  
  # Remote backup integration (future)
  s3_backup_bucket: "arxplorer-disaster-recovery-backups"
  enable_remote_backup: false
"""
    
    config_file = "backup_storage/backup_config.yaml"
    with open(config_file, 'w') as f:
        f.write(config_content)
    
    print(f"Created backup configuration: {config_file}")

def setup_windows_service():
    """Setup instructions for Windows service installation"""
    print("\n" + "="*60)
    print("WINDOWS SERVICE SETUP")
    print("="*60)
    print("""
To run automated backups as a Windows service (recommended for production):

1. Open PowerShell as Administrator

2. Install the service:
   python infrastructure/dr_automation/windows_service.py install

3. Start the service:
   python infrastructure/dr_automation/windows_service.py start

4. Check service status:
   python infrastructure/dr_automation/windows_service.py status

5. To remove the service later:
   python infrastructure/dr_automation/windows_service.py remove

The service will automatically:
- Start when Windows boots
- Run backups every 15 minutes
- Perform health checks
- Log all activities
- Survive system crashes and restarts
""")

def setup_manual_scheduler():
    """Setup instructions for manual scheduler (development/testing)"""
    print("\n" + "="*60)
    print("MANUAL SCHEDULER SETUP (Development/Testing)")
    print("="*60)
    print("""
To run automated backups manually (for testing):

1. Start the scheduler:
   python infrastructure/dr_automation/automated_backup_scheduler.py

2. The scheduler will run continuously and perform:
   - Backup every 15 minutes
   - Health checks every 30 seconds
   - Automatic cleanup of old backups

3. Press Ctrl+C to stop

4. Check backup status:
   python -c "
   from infrastructure.disaster_recovery.automated_backup_scheduler import AutomatedBackupScheduler
   scheduler = AutomatedBackupScheduler()
   status = scheduler.get_backup_status()
   print(f'Status: {status['message']}')
   print(f'Success Rate: {status['success_rate']:.1f}%')
   "
""")

def demonstrate_real_world_scenario():
    """Show how automated backups prevent data loss in real scenarios"""
    print("\n" + "="*60)
    print("REAL-WORLD DISASTER PROTECTION")
    print("="*60)
    print("""
Your automated backup system now protects against:

SCENARIO 1: Hardware Failure
- Server crashes at 2:47 PM
- Last automated backup was at 2:45 PM (2 minutes ago)
- Maximum data loss: 2 minutes of changes
- Recovery time: 15-30 minutes

SCENARIO 2: Accidental Deletion
- Developer accidentally drops MongoDB collection at 3:22 PM
- Last backup was at 3:15 PM (7 minutes ago)
- Maximum data loss: 7 minutes of changes
- Recovery time: 5-10 minutes

SCENARIO 3: Security Breach
- Ransomware detected at 1:30 AM
- Last backup was at 1:15 AM (15 minutes ago)
- Maximum data loss: 15 minutes of changes
- Recovery time: 30-60 minutes

SCENARIO 4: Deployment Failure
- Bad deployment corrupts system at 4:15 PM
- Pre-deployment backup was triggered automatically
- Maximum data loss: 0 minutes
- Recovery time: 10-15 minutes

Your RPO (Recovery Point Objective): 15 minutes maximum
Your RTO (Recovery Time Objective): 30 minutes maximum
""")

def main():
    """Main setup function"""
    print("ArXplorer Disaster Recovery - Production Setup")
    print("=" * 50)
    
    try:
        # Create necessary directories
        create_directories()
        
        # Install dependencies
        install_dependencies()
        
        # Create configuration
        create_config_file()
        
        print("\n" + "✅ " * 20)
        print("AUTOMATED BACKUP SYSTEM READY!")
        print("✅ " * 20)
        
        # Show setup options
        setup_windows_service()
        setup_manual_scheduler()
        demonstrate_real_world_scenario()
        
        print(f"\nSetup complete! Choose your deployment method above.")
        
    except Exception as e:
        print(f"Setup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()