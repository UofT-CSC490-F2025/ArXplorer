#!/usr/bin/env python3
"""
ArXplorer Disaster Recovery - Windows Service Wrapper
Runs automated backups as a Windows background service
"""

import win32serviceutil
import win32service
import win32event
import servicemanager
import sys
import os
from pathlib import Path

# Add the disaster recovery directory to path
sys.path.append(str(Path(__file__).parent))
from automated_backup_scheduler import AutomatedBackupScheduler

class ArXplorerBackupService(win32serviceutil.ServiceFramework):
    """Windows service for automated backup scheduler"""
    
    _svc_name_ = "ArXplorerBackupService"
    _svc_display_name_ = "ArXplorer Automated Backup Service"
    _svc_description_ = "Continuous automated backup service for ArXplorer disaster recovery"

    def __init__(self, args):
        win32serviceutil.ServiceFramework.__init__(self, args)
        self.hWaitStop = win32event.CreateEvent(None, 0, 0, None)
        self.scheduler = None
        self.is_alive = True

    def SvcStop(self):
        """Stop the service"""
        self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
        win32event.SetEvent(self.hWaitStop)
        self.is_alive = False
        if self.scheduler:
            self.scheduler.stop_scheduler()

    def SvcDoRun(self):
        """Run the service"""
        servicemanager.LogMsg(
            servicemanager.EVENTLOG_INFORMATION_TYPE,
            servicemanager.PYS_SERVICE_STARTED,
            (self._svc_name_, '')
        )
        
        try:
            # Change to the correct working directory
            os.chdir("C:/Users/Admin/Desktop/CSC490/ArXplorer")
            
            # Initialize and start backup scheduler
            self.scheduler = AutomatedBackupScheduler()
            scheduler_thread = self.scheduler.start_background_scheduler()
            
            # Wait for stop signal
            while self.is_alive:
                # Wait for stop event (timeout every 30 seconds to check health)
                if win32event.WaitForSingleObject(self.hWaitStop, 30000) == win32event.WAIT_OBJECT_0:
                    break
                
                # Perform periodic health checks
                if self.scheduler:
                    self.scheduler.health_check()
            
        except Exception as e:
            servicemanager.LogErrorMsg(f"Service error: {e}")
        
        servicemanager.LogMsg(
            servicemanager.EVENTLOG_INFORMATION_TYPE,
            servicemanager.PYS_SERVICE_STOPPED,
            (self._svc_name_, '')
        )

if __name__ == '__main__':
    if len(sys.argv) == 1:
        servicemanager.Initialize()
        servicemanager.PrepareToHostSingle(ArXplorerBackupService)
        servicemanager.StartServiceCtrlDispatcher()
    else:
        win32serviceutil.HandleCommandLine(ArXplorerBackupService)