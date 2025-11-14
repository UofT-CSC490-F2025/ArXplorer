# Assignment A5 Part 2 - Load Testing Framework

This directory contains a comprehensive load testing framework for ArXplorer, designed to identify breaking points and system vulnerabilities.

## 🎯 Objective

Red Team vs Blue Team approach:
- **Red Team**: Break the ArXplorer system using systematic stress testing
- **Blue Team**: Implement fixes and improvements based on findings
- **Goal**: Record video demonstration of breaking and fixing the application

## 📁 File Structure

```
load_testing/
├── load_testing.py       # Main Locust-based load testing framework
├── breaking_points.py    # Direct component stress testing
├── runner.py             # Test execution runner with configurations
├── README.md             # This documentation
└── results/              # Generated test reports (auto-created)
```

## 🔧 Setup

### 1. Install Dependencies

```bash
pip install locust psutil numpy
```

### 2. Check Dependencies

```bash
python runner.py check
```

### 3. Ensure ArXplorer is Running

Make sure your ArXplorer API is running on `http://localhost:8000` before running load tests.

## 🚀 Quick Start

### Interactive Mode

```bash
python runner.py
```

This opens an interactive menu with all testing options.

### Command Line Mode

```bash
# Run specific load test
python runner.py light
python runner.py moderate  
python runner.py heavy
python runner.py stress
python runner.py spike

# Run breaking point analysis
python runner.py breaking

# Run all tests
python runner.py all
```

## 📊 Load Test Scenarios

### 1. Light Load
- **Users**: 5 concurrent users
- **Duration**: 30 seconds
- **Purpose**: Baseline performance testing

### 2. Moderate Load  
- **Users**: 15 concurrent users
- **Duration**: 60 seconds
- **Purpose**: Normal busy period simulation

### 3. Heavy Load
- **Users**: 50 concurrent users
- **Duration**: 120 seconds
- **Purpose**: Peak usage simulation

### 4. Stress Test
- **Users**: 100 concurrent users
- **Duration**: 180 seconds
- **Purpose**: Finding system limits

### 5. Spike Test
- **Users**: 200 concurrent users (rapid spawn)
- **Duration**: 60 seconds
- **Purpose**: Sudden traffic surge testing

## 💥 Breaking Point Analysis

The `breaking_points.py` script directly stress tests ArXplorer components:

### Attack Scenarios

1. **Embedding Generation Bomb**
   - Overwhelms embedding generation with concurrent requests
   - Tests: CPU usage, memory consumption, response times
   - Target: SciBERT embedding bottleneck (210.9s from profiling)

2. **Memory Exhaustion Attack**
   - Attempts to exhaust available system memory
   - Tests: Memory allocation limits, OOM handling
   - Progressive allocation until system limits

3. **CPU Exhaustion Attack** 
   - Saturates all CPU cores with intensive computations
   - Tests: CPU scheduling, thermal throttling
   - Matrix multiplication workloads

4. **Index Building Bomb**
   - Overwhelms FAISS index construction
   - Tests: Large-scale vector indexing performance
   - Target: build_index bottleneck (2.391s from profiling)

## 📈 Metrics Collected

### Load Testing (Locust)
- Response times (min/max/average/percentiles)
- Request rates (requests per second)
- Error rates and types
- Concurrent user simulation

### Breaking Point Analysis
- System resource usage (CPU/Memory/Disk/Network)
- Peak resource consumption
- Time to failure
- Exception tracking
- Performance degradation curves

## 🎥 Red Team vs Blue Team Process

### Phase 1: Red Team (Breaking)
1. Run all load tests to identify performance bottlenecks
2. Execute breaking point analysis to find failure modes
3. Document all breaking points and vulnerabilities
4. Record demonstration of system failures

### Phase 2: Blue Team (Fixing)  
1. Analyze failure patterns and resource constraints
2. Implement fixes:
   - Connection pooling
   - Rate limiting
   - Caching improvements
   - Resource optimization
3. Re-run tests to validate improvements
4. Record demonstration of fixed system

### Phase 3: Documentation
1. Create comprehensive report of findings
2. Document all implemented fixes
3. Provide before/after performance comparisons
4. Submit video demonstration

## 🔍 Interpreting Results

### Load Test Results
- **HTML Reports**: Generated in `results/` directory
- **Success Criteria**: 
  - Response times < 2 seconds for 95% of requests
  - Error rate < 1%
  - System stability throughout test

### Breaking Point Indicators
- **Memory**: System fails when >90% memory usage
- **CPU**: Performance degrades when >80% sustained usage  
- **Errors**: Any unhandled exceptions indicate breaking points
- **Timeouts**: Request timeouts indicate system overload

## 🛠️ Customization

### Adding New Load Tests

Edit `LOAD_TEST_CONFIGS` in `runner.py`:

```python
"custom_test": {
    "description": "Custom test description",
    "users": 25,
    "spawn_rate": 3,
    "duration": "90s",
    "host": "http://localhost:8000"
}
```

### Adding New Attack Scenarios

Extend `breaking_points.py` by creating new `AttackScenario` subclasses:

```python
class CustomAttack(AttackScenario):
    def __init__(self):
        super().__init__("Custom Attack Name")
    
    def execute_attack(self):
        # Implement your attack logic
        pass
```

## ⚠️ Safety Considerations

- **Resource Usage**: Breaking point tests may consume significant system resources
- **Data Safety**: Tests use mock data, but ensure no production data exposure
- **System Impact**: High-intensity tests may temporarily affect system performance
- **Cool-down**: Built-in cool-down periods prevent system overheating

## 📋 Assignment Requirements Checklist

- ✅ Load testing framework with multiple scenarios
- ✅ Direct component stress testing
- ✅ System resource monitoring
- ✅ Automated test execution
- ✅ Comprehensive result reporting
- ✅ Red Team attack scenarios
- ⏳ Blue Team fix implementation (Phase 2)
- ⏳ Video demonstration (Phase 3)

## 🚨 Troubleshooting

### Common Issues

1. **"Locust not found"**
   ```bash
   pip install locust
   ```

2. **"Connection refused"**
   - Ensure ArXplorer API is running on localhost:8000
   - Check if port is available: `netstat -an | find "8000"`

3. **"Memory allocation failed"**
   - Expected behavior during memory exhaustion tests
   - Indicates successful breaking point detection

4. **High CPU usage**
   - Expected during CPU exhaustion tests
   - Monitor system temperature and fan activity

### Performance Optimization Tips

1. **Reduce test intensity** if system becomes unresponsive
2. **Monitor system resources** during tests
3. **Use staged approach** - start with light tests
4. **Cool-down periods** between intensive tests

## 📊 Expected Outcomes

Based on A5 Part 1 profiling results:

### Likely Breaking Points
1. **Embedding Generation**: ~210 seconds for model download
2. **Memory**: Large batch processing may exhaust RAM
3. **CPU**: Multiple concurrent embedding requests
4. **Index Building**: Large vector sets may timeout

### Success Metrics
- System handles 50+ concurrent users under normal load
- Graceful degradation under stress
- Clear identification of resource bottlenecks
- Successful implementation of fixes (Phase 2)