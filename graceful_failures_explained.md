# Training Pipeline: Graceful Failure Behaviors Explained

## 🔧 **GRACEFUL FAILURES** - What They Mean & How They Work

### 🧮 **Preprocessing Stage Graceful Failures**

#### **Scaling Fails → Continue without standardization**
**What this means:**
- Standardization (StandardScaler) normalizes features to have mean=0, std=1
- Sometimes fails due to: constant features, numerical precision issues, or corrupted data

**Example Failure Scenarios:**
```python
# Scenario 1: All features have identical values
embeddings = [[0.5, 0.5, 0.5]] * 100  # StandardScaler can't scale constant features

# Scenario 2: Extreme outliers causing numerical issues
embeddings[0] = [1e20, 1e-20, np.nan]  # Causes scaling to fail

# Scenario 3: Memory issues with very large matrices
embeddings = np.random.rand(100000, 1000)  # Too large for available memory
```

**What happens when scaling fails:**
```python
try:
    scaler = StandardScaler()
    processed_embeddings = scaler.fit_transform(embeddings)
    logging.info("✓ Standardization applied successfully")
except Exception as e:
    # GRACEFUL FAILURE: Continue without scaling
    processed_embeddings = embeddings.copy()  # Use original embeddings
    warning_msg = f"Scaling failed: {str(e)} - proceeding without standardization"
    self.training_warnings.append(warning_msg)
    logging.warning("⚠️ %s", warning_msg)
```

**Impact:** HDBSCAN can still work with non-standardized embeddings, but results might be affected by feature scale differences.

#### **PCA Fails → Continue with original dimensions**
**What this means:**
- PCA reduces dimensionality (e.g., 384D → 50D) to speed up clustering
- Sometimes fails due to: insufficient variance, numerical instability, or memory issues

**Example Failure Scenarios:**
```python
# Scenario 1: Insufficient variance for PCA
embeddings = np.ones((100, 384)) * 0.001  # Very low variance, PCA can't find components

# Scenario 2: More components requested than available
pca = PCA(n_components=500)  # But embeddings only have 384 dimensions

# Scenario 3: Numerical instability
embeddings = np.random.rand(50, 1000) * 1e-15  # Very small values cause numerical issues
```

**What happens when PCA fails:**
```python
try:
    pca = PCA(n_components=50, random_state=42)
    processed_embeddings = pca.fit_transform(embeddings)
    logging.info("✓ PCA applied: %d → %d dimensions", original_dims, 50)
except Exception as e:
    # GRACEFUL FAILURE: Use original dimensions
    processed_embeddings = embeddings.copy()  # Keep original embeddings
    warning_msg = f"PCA failed: {str(e)} - proceeding with original dimensions"
    self.training_warnings.append(warning_msg)
    logging.warning("⚠️ %s", warning_msg)
```

**Impact:** Training continues with higher-dimensional data, which may be slower but often more accurate.

#### **Memory Issues → Monitor and warn but continue**
**What this means:**
- System monitors memory usage during preprocessing
- Warns if memory usage is getting high but doesn't stop unless critical

**Example Memory Monitoring:**
```python
import psutil

def monitor_memory_usage(operation_name):
    memory_percent = psutil.virtual_memory().percent
    
    if memory_percent > 85:
        warning_msg = f"High memory usage ({memory_percent:.1f}%) during {operation_name}"
        logging.warning("⚠️ %s", warning_msg)
        
        if memory_percent > 95:
            # Only stop if critically low memory
            raise MemoryError(f"Critical memory usage: {memory_percent:.1f}%")
```

---

## 🎯 **Training Stage RETRY STRATEGY** - Parameter Fallback Logic

### **Understanding "Too Much Noise" → More Lenient Parameters**

When HDBSCAN finds "too much noise" (>90% of points classified as noise), it means the parameters are too strict. Here's how the fallback strategy works:

#### **Primary Parameters Fail: "Too much noise detected"**
```python
# Primary attempt - STRICT parameters
primary_params = {
    "min_cluster_size": 15,  # Large clusters required
    "min_samples": 5,        # High density required
    "metric": "euclidean"
}

# Result: 45/50 points classified as noise (90% noise) → TOO STRICT
```

#### **Fallback 1: More Lenient Cluster Size**
```python
# Make clusters smaller (easier to form)
fallback_1 = {
    "min_cluster_size": 8,   # REDUCED: Smaller clusters allowed
    "min_samples": 3,        # REDUCED: Lower density requirement
    "metric": "euclidean"
}

# Result: 20/50 points classified as noise (40% noise) → BETTER
```

#### **Fallback 2: Even More Lenient**
```python
# Further relax requirements
fallback_2 = {
    "min_cluster_size": 5,   # FURTHER REDUCED: Very small clusters OK
    "min_samples": 2,        # FURTHER REDUCED: Low density OK
    "metric": "cosine"       # DIFFERENT METRIC: May find different patterns
}

# Result: 8/50 points classified as noise (16% noise) → ACCEPTABLE
```

#### **Fallback 3: Last Resort**
```python
# Maximum leniency
fallback_3 = {
    "min_cluster_size": 3,   # MINIMUM: Tiny clusters allowed
    "min_samples": 1,        # MINIMUM: Single points can form core
    "metric": "manhattan"    # DIFFERENT METRIC: Different distance calculation
}
```

### **Parameter Adjustment Logic for Different Failure Types**

#### **"No clusters found" → More Lenient Parameters**
```python
# Problem: Parameters too strict, everything is noise
# Solution: Reduce min_cluster_size and min_samples

if "No clusters found" in error_message:
    next_params = {
        "min_cluster_size": max(3, current_params["min_cluster_size"] // 2),
        "min_samples": max(1, current_params["min_samples"] // 2),
        "metric": current_params["metric"]
    }
```

#### **"Too much noise" → More Lenient Parameters**
```python
# Problem: >90% points classified as noise
# Solution: Reduce density requirements

if noise_ratio > 0.9:
    next_params = {
        "min_cluster_size": max(3, current_params["min_cluster_size"] - 2),
        "min_samples": max(1, current_params["min_samples"] - 1),
        "metric": current_params["metric"]
    }
```

#### **"Convergence failed" → Different Metrics/Parameters**
```python
# Problem: Algorithm can't converge with current metric
# Solution: Try different distance metrics

if "convergence" in error_message.lower():
    metric_alternatives = {
        "euclidean": "cosine",
        "cosine": "manhattan", 
        "manhattan": "euclidean"
    }
    next_params = {
        "min_cluster_size": current_params["min_cluster_size"],
        "min_samples": current_params["min_samples"],
        "metric": metric_alternatives[current_params["metric"]]
    }
```

### **Complete Fallback Strategy Example**

```python
def create_adaptive_fallback_params(dataset_size, primary_failure_reason):
    """Create fallback parameters based on why primary failed"""
    
    base_cluster_size = max(3, dataset_size // 10)  # Adaptive to dataset size
    
    if "too much noise" in primary_failure_reason.lower():
        # Make clustering more lenient
        return [
            {"min_cluster_size": base_cluster_size - 2, "min_samples": 2, "metric": "euclidean"},
            {"min_cluster_size": base_cluster_size - 3, "min_samples": 1, "metric": "cosine"},
            {"min_cluster_size": 3, "min_samples": 1, "metric": "manhattan"}
        ]
    
    elif "no clusters" in primary_failure_reason.lower():
        # Drastically reduce requirements
        return [
            {"min_cluster_size": max(3, base_cluster_size // 2), "min_samples": 1, "metric": "euclidean"},
            {"min_cluster_size": 3, "min_samples": 1, "metric": "cosine"},
            {"min_cluster_size": 2, "min_samples": 1, "metric": "euclidean"}  # Last resort
        ]
    
    elif "convergence" in primary_failure_reason.lower():
        # Try different metrics with same strictness
        return [
            {"min_cluster_size": base_cluster_size, "min_samples": 3, "metric": "cosine"},
            {"min_cluster_size": base_cluster_size, "min_samples": 3, "metric": "manhattan"},
            {"min_cluster_size": base_cluster_size - 1, "min_samples": 2, "metric": "euclidean"}
        ]
```

## 🎯 **Key Insight: Why Graceful Failures Matter**

### **Without Graceful Failures:**
```
❌ Scaling failed → TRAINING STOPS
❌ PCA failed → TRAINING STOPS  
❌ Primary params failed → TRAINING STOPS
Result: 80% of training attempts fail due to minor issues
```

### **With Graceful Failures:**
```
⚠️ Scaling failed → Continue without scaling
⚠️ PCA failed → Continue with original dimensions
🔄 Primary params failed → Try fallback 1
🔄 Fallback 1 failed → Try fallback 2
✅ Fallback 2 succeeded → Training successful
Result: 95% of training attempts succeed with acceptable quality
```

The graceful failure system ensures that **minor technical issues don't prevent successful clustering**, while still providing visibility into what went wrong so you can improve the process over time.