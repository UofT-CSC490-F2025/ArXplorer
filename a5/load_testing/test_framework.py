"""
Simple test to verify our load testing framework works
"""

import sys
import os

# Add project root to path  
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, ROOT)

print("🧪 TESTING LOAD TESTING FRAMEWORK")
print("=" * 40)

# Test 1: Import all modules
try:
    from breaking_points import SystemMonitor, AttackScenario
    print("✅ Breaking points module imported successfully")
except Exception as e:
    print(f"❌ Breaking points import failed: {e}")

# Test 2: Test system monitoring
try:
    monitor = SystemMonitor()
    print("✅ SystemMonitor created successfully")
    
    # Quick monitoring test
    monitor.start_monitoring()
    import time
    time.sleep(2)  # Monitor for 2 seconds
    stats = monitor.stop_monitoring()
    
    if stats:
        print(f"✅ Collected {len(stats)} monitoring samples")
        print(f"   Sample CPU: {stats[0]['cpu_percent']:.1f}%")
        print(f"   Sample Memory: {stats[0]['memory_percent']:.1f}%")
    else:
        print("❌ No monitoring stats collected")
        
except Exception as e:
    print(f"❌ SystemMonitor test failed: {e}")

# Test 3: Test simple attack scenario
try:
    class TestAttack(AttackScenario):
        def __init__(self):
            super().__init__("Simple Test Attack")
            
        def execute_attack(self):
            print("   Executing simple test attack...")
            import time
            time.sleep(1)  # Simple 1-second operation
            
            self.results.update({
                'test_metric': 42,
                'attack_successful': True
            })
    
    attack = TestAttack()
    result = attack.run_attack()
    
    if result and 'test_metric' in result:
        print("✅ Attack scenario framework working")
        print(f"   Duration: {result['duration_seconds']:.2f}s")
    else:
        print("❌ Attack scenario test failed")
        
except Exception as e:
    print(f"❌ Attack scenario test failed: {e}")

# Test 4: Check runner module
try:
    import runner
    print("✅ Runner module imported successfully")
    
    # Test configuration loading
    configs = runner.LOAD_TEST_CONFIGS
    print(f"✅ Found {len(configs)} load test configurations")
    for name in configs.keys():
        print(f"   - {name}")
        
except Exception as e:
    print(f"❌ Runner module test failed: {e}")

print("\n🎯 FRAMEWORK TEST COMPLETE")
print("If all tests passed, the framework is ready to use!")