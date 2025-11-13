#!/usr/bin/env python3
"""
ArXplorer Disaster Recovery - Automated Backup Scheduler
Production-grade continuous backup system with scheduling and monitoring
"""

import asyncio
import schedule
import time
import threading
import logging
from datetime import datetime, timedelta
import json
import os
import subprocess
import sys
from pathlib import Path

# Import our existing backup system
import importlib.util
spec = importlib.util.spec_from_file_location(
    "backup_module", 
    str(Path(__file__).parent / "01_pre_disaster_backup.py")
)
backup_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backup_module)
PreDisasterBackup = backup_module.PreDisasterBackup

class AutomatedBackupScheduler:
    def __init__(self):
        """Initialize automated backup scheduler"""
        self.setup_logging()
        self.backup_system = PreDisasterBackup()
        self.backup_history = []
        self.is_running = False
        
    def setup_logging(self):
        """Configure logging for backup operations"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('backup_storage/logs/backup_scheduler.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def backup_job(self, backup_type="scheduled"):
        """Execute backup job with error handling and logging"""
        try:
            self.logger.info(f"Starting {backup_type} backup...")
            
            # Create timestamped backup
            backup_result = {
                'timestamp': datetime.now().isoformat(),
                'type': backup_type,
                'status': 'running',
                'components': {}
            }
            
            # Execute backup components
            try:
                s3_result = self.backup_system.backup_aws_s3_state()
                backup_result['components']['s3'] = {'status': 'success', 'buckets': len(s3_result['buckets'])}
            except Exception as e:
                backup_result['components']['s3'] = {'status': 'failed', 'error': str(e)}
                self.logger.error(f"S3 backup failed: {e}")
            
            try:
                mongo_result = self.backup_system.backup_mongodb_state()
                backup_result['components']['mongodb'] = {'status': 'success', 'collections': len(mongo_result['collections'])}
            except Exception as e:
                backup_result['components']['mongodb'] = {'status': 'failed', 'error': str(e)}
                self.logger.error(f"MongoDB backup failed: {e}")
            
            try:
                config_result = self.backup_system.backup_application_config()
                backup_result['components']['config'] = {'status': 'success'}
            except Exception as e:
                backup_result['components']['config'] = {'status': 'failed', 'error': str(e)}
                self.logger.error(f"Config backup failed: {e}")
            
            try:
                tf_result = self.backup_system.backup_terraform_state()
                backup_result['components']['terraform'] = {'status': 'success'}
            except Exception as e:
                backup_result['components']['terraform'] = {'status': 'failed', 'error': str(e)}
                self.logger.error(f"Terraform backup failed: {e}")
            
            # Determine overall status
            component_statuses = [comp['status'] for comp in backup_result['components'].values()]
            if all(status == 'success' for status in component_statuses):
                backup_result['status'] = 'success'
                self.logger.info(f"{backup_type} backup completed successfully")
            elif any(status == 'success' for status in component_statuses):
                backup_result['status'] = 'partial'
                self.logger.warning(f"{backup_type} backup completed with errors")
            else:
                backup_result['status'] = 'failed'
                self.logger.error(f"{backup_type} backup failed completely")
            
            # Store backup history
            self.backup_history.append(backup_result)
            self.save_backup_history()
            
            # Cleanup old backups (keep last 30 days)
            self.cleanup_old_backups()
            
            return backup_result
            
        except Exception as e:
            self.logger.error(f"Critical error in backup job: {e}")
            backup_result = {
                'timestamp': datetime.now().isoformat(),
                'type': backup_type,
                'status': 'critical_failure',
                'error': str(e)
            }
            self.backup_history.append(backup_result)
            return backup_result
    
    def setup_schedules(self):
        """Configure backup schedules for different frequencies"""
        
        # CRITICAL: Every 15 minutes (for high-change periods)
        schedule.every(15).minutes.do(self.backup_job, "critical-15min")
        
        # HIGH: Every hour (for normal operations)
        schedule.every().hour.do(self.backup_job, "hourly")
        
        # DAILY: Every day at 2 AM (for full system backup)
        schedule.every().day.at("02:00").do(self.backup_job, "daily-full")
        
        # WEEKLY: Every Sunday at 1 AM (for long-term retention)
        schedule.every().sunday.at("01:00").do(self.backup_job, "weekly-archive")
        
        # BEFORE DEPLOYMENTS: Event-driven backup
        # This would be triggered by CI/CD pipeline or manual deployment
        
        self.logger.info("Backup schedules configured:")
        self.logger.info("- Critical: Every 15 minutes")
        self.logger.info("- Hourly: Every hour")
        self.logger.info("- Daily: Every day at 2 AM")
        self.logger.info("- Weekly: Every Sunday at 1 AM")
    
    def save_backup_history(self):
        """Save backup history to disk"""
        history_file = "backup_storage/logs/backup_history.json"
        os.makedirs(os.path.dirname(history_file), exist_ok=True)
        
        with open(history_file, 'w') as f:
            json.dump(self.backup_history, f, indent=2, default=str)
    
    def cleanup_old_backups(self):
        """Remove backups older than retention period"""
        cutoff_date = datetime.now() - timedelta(days=30)
        
        backup_dir = Path("disaster_recovery/backups")
        if backup_dir.exists():
            for backup_folder in backup_dir.iterdir():
                if backup_folder.is_dir():
                    try:
                        # Parse timestamp from folder name
                        folder_time = datetime.strptime(backup_folder.name, "%Y%m%d_%H%M%S")
                        
                        if folder_time < cutoff_date:
                            self.logger.info(f"Removing old backup: {backup_folder.name}")
                            import shutil
                            shutil.rmtree(backup_folder)
                            
                    except ValueError:
                        # Skip folders that don't match timestamp pattern
                        continue
    
    def health_check(self):
        """Perform system health check and backup if needed"""
        try:
            # Check if last backup is too old
            if self.backup_history:
                last_backup = datetime.fromisoformat(self.backup_history[-1]['timestamp'])
                if datetime.now() - last_backup > timedelta(minutes=20):
                    self.logger.warning("Last backup is too old, triggering emergency backup")
                    self.backup_job("emergency-health-check")
            else:
                self.logger.warning("No backup history found, triggering initial backup")
                self.backup_job("initial-health-check")
                
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
    
    def run_scheduler(self):
        """Main scheduler loop - runs in background thread"""
        self.is_running = True
        self.logger.info("Automated backup scheduler started")
        
        # Initial health check
        self.health_check()
        
        # Run scheduler
        while self.is_running:
            schedule.run_pending()
            time.sleep(30)  # Check every 30 seconds
            
        self.logger.info("Automated backup scheduler stopped")
    
    def start_background_scheduler(self):
        """Start scheduler in background thread"""
        self.setup_schedules()
        
        scheduler_thread = threading.Thread(target=self.run_scheduler, daemon=True)
        scheduler_thread.start()
        
        return scheduler_thread
    
    def stop_scheduler(self):
        """Stop the background scheduler"""
        self.is_running = False
        schedule.clear()
        self.logger.info("Backup scheduler shutdown requested")
    
    def get_backup_status(self):
        """Get current backup system status"""
        if not self.backup_history:
            return {
                'status': 'no_backups',
                'message': 'No backups have been performed yet',
                'last_backup': None
            }
        
        last_backup = self.backup_history[-1]
        last_time = datetime.fromisoformat(last_backup['timestamp'])
        time_since = datetime.now() - last_time
        
        # Determine health status
        if time_since > timedelta(hours=2):
            status = 'critical'
            message = f"Last backup was {time_since} ago - CRITICAL"
        elif time_since > timedelta(minutes=30):
            status = 'warning'
            message = f"Last backup was {time_since} ago - WARNING"
        else:
            status = 'healthy'
            message = f"Last backup was {time_since} ago - OK"
        
        return {
            'status': status,
            'message': message,
            'last_backup': last_backup,
            'total_backups': len(self.backup_history),
            'success_rate': len([b for b in self.backup_history if b['status'] == 'success']) / len(self.backup_history) * 100
        }

def main():
    """Main entry point for automated backup scheduler"""
    scheduler = AutomatedBackupScheduler()
    
    try:
        # Start background scheduler
        scheduler_thread = scheduler.start_background_scheduler()
        
        print("ArXplorer Automated Backup Scheduler")
        print("====================================")
        print("Background backup scheduler is now running...")
        print("Press Ctrl+C to stop")
        print()
        
        # Keep main thread alive and show status
        while True:
            status = scheduler.get_backup_status()
            print(f"\rStatus: {status['message']}", end="", flush=True)
            time.sleep(10)
            
    except KeyboardInterrupt:
        print("\n\nShutting down backup scheduler...")
        scheduler.stop_scheduler()
        print("Backup scheduler stopped.")

if __name__ == "__main__":
    main()