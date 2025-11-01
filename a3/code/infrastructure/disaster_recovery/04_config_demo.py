#!/usr/bin/env python3
"""
ArXplorer Disaster Recovery - Demonstration Script
Shows the different configuration destruction modes
"""

import sys
import os
import importlib.util

def load_disaster_simulation():
    """Load the DisasterSimulation class from 02_disaster_simulation.py"""
    script_path = os.path.join(os.path.dirname(__file__), '02_disaster_simulation.py')
    spec = importlib.util.spec_from_file_location("disaster_simulation", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.DisasterSimulation

def demonstrate_config_options():
    """Demonstrate different configuration destruction modes"""
    print("ArXplorer Disaster Recovery - Configuration Destruction Demo")
    print("=" * 70)
    print()
    
    DisasterSimulation = load_disaster_simulation()
    # Note: We don't instantiate here to avoid config loading issues
    
    print("📋 Available Configuration Destruction Modes:")
    print()
    print("1. CORRUPTION MODE:")
    print("   - Creates corrupted config file with disaster markers")
    print("   - File still exists but is unusable")
    print("   - Shows as 'CORRUPTED' in results")
    print("   - Can be restored from backup")
    print()
    
    print("2. DESTRUCTION MODE:")
    print("   - Completely deletes the configuration file")
    print("   - File no longer exists")
    print("   - Shows as 'DESTROYED' in results")  
    print("   - Requires recreation from scratch or backup restoration")
    print()
    
    print("🔍 Current Configuration Status:")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    config_file = os.path.join(project_root, 'config.yaml')
    
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            content = f.read(50)  # First 50 chars
        
        if 'CORRUPTED CONFIG FILE' in content:
            print(f"   Status: CORRUPTED (disaster marker found)")
        else:
            print(f"   Status: HEALTHY (normal configuration)")
        print(f"   Location: {config_file}")
        print(f"   Size: {os.path.getsize(config_file)} bytes")
    else:
        print(f"   Status: DESTROYED (file does not exist)")
        print(f"   Expected Location: {config_file}")
    
    print()
    print("💡 Recovery Options:")
    print("   - Restore from backup: config.yaml.original.TIMESTAMP")
    print("   - Create new config: python 03_create_new_config.py")
    print("   - Manual recreation: Copy from template")

if __name__ == "__main__":
    demonstrate_config_options()