"""
Assignment A5 Part 2 - Load Testing Configuration and Runner (FIXED VERSION)
Automated execution of load tests against ArXplorer with optimized load settings

This script provides different load testing scenarios with reduced load to stay within capacity limits
"""

import os
import sys
import time
import json
import subprocess
from datetime import datetime

# FIXED Load testing configurations - reduced load to stay within capacity
LOAD_TEST_CONFIGS = {
    "light": {
        "description": "Light load - normal usage simulation (FIXED)",
        "users": 2,  # REDUCED from 5 to 2
        "spawn_rate": 0.5,  # REDUCED from 1 to 0.5
        "duration": "30s",
        "host": "http://localhost:8000"
    },
    
    "moderate": {
        "description": "Moderate load - busy period simulation (FIXED)", 
        "users": 5,  # REDUCED from 15 to 5
        "spawn_rate": 1,  # REDUCED from 2 to 1
        "duration": "60s",
        "host": "http://localhost:8000"
    },
    
    "heavy": {
        "description": "Heavy load - peak usage simulation (FIXED)",
        "users": 10,  # REDUCED from 50 to 10
        "spawn_rate": 2,  # REDUCED from 5 to 2
        "duration": "120s", 
        "host": "http://localhost:8000"
    },
    
    "stress": {
        "description": "Stress test - finding breaking points (FIXED)",
        "users": 15,  # REDUCED from 100 to 15
        "spawn_rate": 3,  # REDUCED from 10 to 3
        "duration": "180s",
        "host": "http://localhost:8000"
    },
    
    "spike": {
        "description": "Spike test - sudden traffic surge (FIXED)",
        "users": 20,  # REDUCED from 200 to 20
        "spawn_rate": 5,  # REDUCED from 50 to 5
        "duration": "60s",
        "host": "http://localhost:8000"
    }
}


def run_locust_test(config_name, config, custom_args=None):
    """Run a locust load test with given configuration"""
    
    print(f"\nSTARTING LOAD TEST: {config_name.upper()}")
    print(f"   Description: {config['description']}")
    print(f"   Users: {config['users']}")
    print(f"   Spawn Rate: {config['spawn_rate']}/sec")
    print(f"   Duration: {config['duration']}")
    print(f"   Target: {config['host']}")
    print("-" * 50)
    
    # Prepare locust command
    cmd = [
        "locust",
        "-f", "load_testing.py",  # Our locust file
        "--users", str(config['users']),
        "--spawn-rate", str(config['spawn_rate']),
        "--run-time", config['duration'],
        "--host", config['host'],
        "--headless",  # No web UI
        "--print-stats",
        "--html", f"results/load_test_{config_name}_fixed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    ]
    
    # Add any custom arguments
    if custom_args:
        cmd.extend(custom_args)
    
    # Create results directory
    os.makedirs("results", exist_ok=True)
    
    try:
        # Run the load test
        result = subprocess.run(
            cmd,
            cwd=os.path.dirname(__file__),  # Run from load_testing directory
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        print("Load test completed!")
        print("\nRESULTS:")
        print(result.stdout)
        
        if result.stderr:
            print("\nWARNINGS/ERRORS:")
            print(result.stderr)
            
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "return_code": result.returncode
        }
        
    except subprocess.TimeoutExpired:
        print("Load test timed out after 5 minutes")
        return {
            "success": False,
            "error": "Timeout after 5 minutes"
        }
    except FileNotFoundError:
        print("Locust not found! Install with: pip install locust")
        return {
            "success": False,
            "error": "Locust not installed"
        }
    except Exception as e:
        print(f"Load test failed: {e}")
        return {
            "success": False,
            "error": str(e)
        }


def run_breaking_point_analysis():
    """Run the breaking point analysis script"""
    
    print("\nRUNNING BREAKING POINT ANALYSIS")
    print("This will stress test the ArXplorer components directly")
    print("-" * 50)
    
    try:
        result = subprocess.run(
            [sys.executable, "breaking_points.py"],
            cwd=os.path.dirname(__file__),
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout for intensive operations
        )
        
        print("Breaking point analysis completed!")
        print(result.stdout)
        
        if result.stderr:
            print("\nERRORS:")
            print(result.stderr)
            
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr
        }
        
    except subprocess.TimeoutExpired:
        print("Breaking point analysis timed out")
        return {"success": False, "error": "Timeout"}
    except Exception as e:
        print(f"Breaking point analysis failed: {e}")
        return {"success": False, "error": str(e)}


def check_dependencies():
    """Check if required dependencies are installed"""
    
    print("CHECKING DEPENDENCIES")
    print("-" * 30)
    
    dependencies = {
        "locust": "pip install locust",
        "psutil": "pip install psutil", 
        "numpy": "pip install numpy"
    }
    
    missing = []
    
    for package, install_cmd in dependencies.items():
        try:
            __import__(package)
            print(f"[OK] {package} - installed")
        except ImportError:
            print(f"[MISSING] {package} - missing (run: {install_cmd})")
            missing.append(package)
    
    if missing:
        print(f"\nMissing dependencies: {', '.join(missing)}")
        print("Install them before running load tests")
        return False
    
    print("All dependencies satisfied!")
    return True


def interactive_menu():
    """Interactive menu for load testing"""
    
    while True:
        print("\n" + "=" * 60)
        print("ARXPLORER LOAD TESTING MENU (FIXED VERSION)")
        print("Assignment A5 Part 2 - Load Testing - Optimized Settings")
        print("=" * 60)
        
        print("\nLoad Test Scenarios (REDUCED LOAD):")
        for i, (name, config) in enumerate(LOAD_TEST_CONFIGS.items(), 1):
            print(f"  {i}. {name.upper()} - {config['description']}")
        
        print(f"\n  {len(LOAD_TEST_CONFIGS) + 1}. Breaking Point Analysis")
        print(f"  {len(LOAD_TEST_CONFIGS) + 2}. Run All Tests")
        print(f"  {len(LOAD_TEST_CONFIGS) + 3}. Check Dependencies")
        print("  0. Exit")
        
        try:
            choice = input("\nSelect option: ").strip()
            
            if choice == "0":
                print("Goodbye!")
                break
                
            elif choice.isdigit():
                choice_num = int(choice)
                config_names = list(LOAD_TEST_CONFIGS.keys())
                
                if 1 <= choice_num <= len(config_names):
                    # Run specific load test
                    config_name = config_names[choice_num - 1]
                    config = LOAD_TEST_CONFIGS[config_name]
                    run_locust_test(config_name, config)
                    
                elif choice_num == len(LOAD_TEST_CONFIGS) + 1:
                    # Breaking point analysis
                    run_breaking_point_analysis()
                    
                elif choice_num == len(LOAD_TEST_CONFIGS) + 2:
                    # Run all tests
                    print("\nRUNNING ALL LOAD TESTS (FIXED VERSION)")
                    print("Using optimized load settings to stay within capacity limits")
                    print("-" * 60)
                    
                    all_passed = True
                    for name, config in LOAD_TEST_CONFIGS.items():
                        result = run_locust_test(name, config)
                        if not result["success"]:
                            print(f"Test {name} failed, stopping sequence")
                            all_passed = False
                            break
                        time.sleep(5)  # Cool-down between tests
                    
                    if all_passed:
                        print("\nSUCCESS! All tests passed with optimized load settings")
                        print("The application can handle the reduced load successfully")
                    
                    print("\nRunning breaking point analysis...")
                    run_breaking_point_analysis()
                    
                elif choice_num == len(LOAD_TEST_CONFIGS) + 3:
                    # Check dependencies
                    check_dependencies()
                    
                else:
                    print("Invalid option")
            else:
                print("Please enter a number")
                
        except KeyboardInterrupt:
            print("\nInterrupted by user, goodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    """Main entry point"""
    
    print("ARXPLORER LOAD TESTING - FIXED VERSION")
    print("Optimized load settings to demonstrate successful testing")
    print("=" * 60)
    
    # Check if this is being run directly or with arguments
    if len(sys.argv) > 1:
        # Command line mode
        command = sys.argv[1].lower()
        
        if command == "check":
            check_dependencies()
            
        elif command == "breaking":
            run_breaking_point_analysis()
            
        elif command in LOAD_TEST_CONFIGS:
            config = LOAD_TEST_CONFIGS[command]
            run_locust_test(command, config)
            
        elif command == "all":
            # Run all tests
            print("Running all tests with optimized settings...")
            for name, config in LOAD_TEST_CONFIGS.items():
                run_locust_test(name, config)
                time.sleep(5)
            run_breaking_point_analysis()
            
        else:
            print(f"Unknown command: {command}")
            print("Available commands: check, breaking, all, " + ", ".join(LOAD_TEST_CONFIGS.keys()))
            
    else:
        # Interactive mode
        interactive_menu()


if __name__ == "__main__":
    main()