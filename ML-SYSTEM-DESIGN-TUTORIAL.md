# Machine Learning System Design
### A Comprehensive Guide to Production ML, AI Architectures, and Agentic AI Systems

*With 3000 years of ML engineering wisdom distilled into practical, battle-tested patterns*

---

## 📚 Table of Contents

### **Foundation**
- [What is ML System Design?](#what-is-ml-system-design)
- [ML vs Traditional Systems](#ml-vs-traditional-systems)
- [Key Principles](#key-principles)

### **Chapter I: ML Architecture Fundamentals**
- [Machine Learning Pipeline](#machine-learning-pipeline)
- [Training vs Inference](#training-vs-inference)
- [Batch, Real-time & Streaming](#batch-real-time--streaming)
- [Model Serving Patterns](#model-serving-patterns)
- [Feature Stores](#feature-stores)
- [Model Registry](#model-registry)

### **Chapter II: Data Engineering**
- [Data Collection & Ingestion](#data-collection--ingestion)
- [Data Quality & Validation](#data-quality--validation)
- [Feature Engineering](#feature-engineering)
- [Handling Imbalanced Data](#handling-imbalanced-data)
- [Data Versioning](#data-versioning)

### **Chapter III: Model Training at Scale**
- [Distributed Training](#distributed-training)
- [Hyperparameter Optimization](#hyperparameter-optimization)
- [Transfer Learning](#transfer-learning)
- [Few-Shot & Zero-Shot Learning](#few-shot--zero-shot-learning)
- [Continual Learning](#continual-learning)

### **Chapter IV: Model Deployment**
- [Model Optimization](#model-optimization)
- [Quantization & Pruning](#quantization--pruning)
- [Model Compression](#model-compression)
- [Edge Deployment](#edge-deployment)
- [Multi-Model Serving](#multi-model-serving)

### **Chapter V: MLOps & Production**
- [CI/CD for ML](#cicd-for-ml)
- [Model Monitoring](#model-monitoring)
- [Data & Concept Drift](#data--concept-drift)
- [A/B Testing](#ab-testing-for-ml)
- [Shadow & Canary Deployment](#shadow--canary-deployment)

### **Chapter VI: Deep Learning Systems**
- [Neural Architecture Design](#neural-architecture-design)
- [Computer Vision Systems](#computer-vision-systems)
- [NLP Systems](#nlp-systems)
- [Recommendation Systems](#recommendation-systems)
- [Time Series Systems](#time-series-systems)

### **Chapter VII: Large Language Models**
- [LLM Architecture](#llm-architecture)
- [Transformer Deep Dive](#transformer-deep-dive)
- [Pre-training & Fine-tuning](#pre-training--fine-tuning)
- [LoRA & PEFT](#lora--peft)
- [Prompt Engineering](#prompt-engineering)
- [LLM Inference Optimization](#llm-inference-optimization)

### **Chapter VIII: Retrieval-Augmented Generation**
- [RAG Architecture](#rag-architecture)
- [Vector Databases](#vector-databases)
- [Embedding Models](#embedding-models)
- [Chunking Strategies](#chunking-strategies)
- [Advanced RAG Patterns](#advanced-rag-patterns)
- [RAG Evaluation](#rag-evaluation)

### **Chapter IX: Agentic AI**
- [AI Agent Fundamentals](#ai-agent-fundamentals)
- [ReAct Pattern](#react-pattern)
- [Tool Use & Function Calling](#tool-use--function-calling)
- [Planning & Reasoning](#planning--reasoning)
- [Memory Systems](#memory-systems)
- [Multi-Agent Systems](#multi-agent-systems)
- [Agent Evaluation](#agent-evaluation)

### **Chapter X: Production Case Studies**
- [Fraud Detection System](#fraud-detection-system)
- [Recommendation Engine](#recommendation-engine)
- [Conversational AI Assistant](#conversational-ai-assistant)
- [Real-time Translation](#real-time-translation)

---

# What is ML System Design?

Machine Learning System Design is the art and science of building production-grade ML systems that are:
- **Reliable** - Consistent predictions under varying conditions
- **Scalable** - Handle growing data and traffic
- **Maintainable** - Easy to debug, update, improve
- **Cost-Effective** - Optimize computational resources
- **Fast** - Meet latency requirements
- **Safe** - Handle edge cases gracefully
- **Fair** - Unbiased across user segments

Unlike traditional software where logic is explicitly coded, ML systems learn behavior from data. This fundamental difference creates unique challenges and design considerations.

## The Hidden ML Iceberg

According to Google's research, only 5-10% of a production ML system is actual ML code:

```
          Traditional View          Reality
               
               [Model]              [Model Code: 5%]
                                    |
                                    ├── Data Collection
                                    ├── Data Validation  
                                    ├── Feature Engineering
                                    ├── Process Management
                                    ├── Model Training Infrastructure
                                    ├── Serving Infrastructure
                                    ├── Monitoring & Alerting
                                    ├── Configuration Management
                                    └── Resource Management: 95%
```

## ML vs Traditional Systems

| Dimension | Traditional Systems | ML Systems |
|-----------|-------------------|------------|
| **Logic** | Explicit rules in code | Learned patterns from data |
| **Testing** | Unit/integration tests | Data validation + model evaluation |
| **Deployment** | Code changes | Model + data + code changes |
| **Debugging** | Stack traces, logs | Metrics, drift analysis, feature importance |
| **Performance** | Latency, throughput | Accuracy + latency + throughput |
| **Versioning** | Code versions | Code + data + model versions |
| **Failures** | Binary (works/breaks) | Gradual quality degradation |
| **Dependencies** | Libraries, services | Libraries + services + data + models |

## Key Principles

### 1. Data-Centric Mindset
**The quality of your model is bounded by the quality of your data.**

```python
# Bad: Focus only on model
model = ComplexNeuralNet(layers=50)
model.fit(messy_data)  # Garbage in, garbage out

# Good: Focus on data quality first
cleaned_data = validate_and_clean(raw_data)
engineered_features = create_features(cleaned_data)
model = SimpleModel()  # Often simpler is better
model.fit(engineered_features)
```

### 2. Start Simple, Then Optimize
**Deploy a simple baseline quickly, then iterate.**

```
Week 1: Simple logistic regression (baseline)
Week 2: Deploy to 5% traffic
Week 3: Analyze errors, improve features
Week 4: Try gradient boosting
Week 5: Gradual rollout
```

Don't spend months building a perfect model that nobody uses.

### 3. Monitor Everything
**What gets measured gets managed.**

Essential metrics:
- **Model Performance**: Accuracy, precision, recall, AUC
- **Data Quality**: Missing values, outliers, distribution shifts
- **Operational**: Latency, throughput, error rates
- **Business**: Revenue impact, user engagement

### 4. Design for Failure
**Models degrade. Plan for it.**

- Implement fallback mechanisms
- Version all components
- Enable quick rollbacks
- Monitor for drift
- Automate retraining

### 5. Reproducibility is Critical
**Science requires repeatability.**

Version and track:
- Training data (checksums, versions)
- Code (git commits)
- Dependencies (requirements.txt, Dockerfiles)
- Hyperparameters
- Random seeds
- Training environment

---

# Chapter I: ML Architecture Fundamentals

## Machine Learning Pipeline

A production ML pipeline automates the end-to-end process from data ingestion to model deployment.

### Core Pipeline Stages

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────┐
│   Data      │───▶│    Data      │───▶│   Feature   │───▶│  Model   │
│  Ingestion  │    │  Validation  │    │ Engineering │    │ Training │
└─────────────┘    └──────────────┘    └─────────────┘    └──────────┘
                                                                  │
                                                                  ▼
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────┐
│   Monitor   │◀───│    Deploy    │◀───│  Validate   │◀───│ Evaluate │
│  & Retrain  │    │    Model     │    │    Model    │    │  Model   │
└─────────────┘    └──────────────┘    └─────────────┘    └──────────┘
```

### 1. Data Ingestion

**Batch Ingestion** (Scheduled loads):
```python
# Daily batch ingestion with Airflow
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

def ingest_daily_data(**context):
    execution_date = context['execution_date']
    
    # Extract from source
    df = extract_from_database(
        start_date=execution_date,
        end_date=execution_date + timedelta(days=1)
    )
    
    # Validate schema
    validate_schema(df, expected_schema)
    
    # Store
    df.to_parquet(f's3://data/raw/date={execution_date}/data.parquet')

dag = DAG(
    'daily_ingestion',
    schedule_interval='@daily',
    start_date=datetime(2024, 1, 1)
)

ingest_task = PythonOperator(
    task_id='ingest',
    python_callable=ingest_daily_data,
    provide_context=True,
    dag=dag
)
```

**Streaming Ingestion** (Real-time):
```python
# Kafka streaming ingestion
from kafka import KafkaConsumer
import json

consumer = KafkaConsumer(
    'transactions',
    bootstrap_servers=['localhost:9092'],
    value_deserializer=lambda m: json.loads(m.decode('utf-8')),
    auto_offset_reset='earliest',
    enable_auto_commit=True
)

for message in consumer:
    transaction = message.value
    
    # Validate
    if validate_transaction(transaction):
        # Process and store
        store_transaction(transaction)
        
        # Trigger real-time feature update
        update_features(transaction['user_id'])
```

### 2. Data Validation

Catch data quality issues early:

```python
# Using Great Expectations
import great_expectations as ge

def validate_data(df):
    df_ge = ge.from_pandas(df)
    
    # Schema validation
    df_ge.expect_column_to_exist("transaction_id")
    df_ge.expect_column_to_exist("amount")
    df_ge.expect_column_to_exist("user_id")
    
    # Value validation
    df_ge.expect_column_values_to_not_be_null("transaction_id")
    df_ge.expect_column_values_to_be_between("amount", min_value=0, max_value=1000000)
    df_ge.expect_column_values_to_be_of_type("user_id", "int")
    
    # Distribution checks
    df_ge.expect_column_mean_to_be_between("amount", min_value=10, max_value=1000)
    df_ge.expect_column_stdev_to_be_between("amount", min_value=5, max_value=500)
    
    results = df_ge.validate()
    
    if not results["success"]:
        raise ValueError(f"Data validation failed: {results}")
    
    return df
```

### 3. Feature Engineering

Transform raw data into model-ready features:

```python
def engineer_features(transactions_df, users_df):
    """
    Create features from raw transactions
    """
    
    # Temporal features
    transactions_df['hour_of_day'] = transactions_df['timestamp'].dt.hour
    transactions_df['day_of_week'] = transactions_df['timestamp'].dt.dayofweek
    transactions_df['is_weekend'] = transactions_df['day_of_week'].isin([5, 6])
    
    # Aggregation features (user level)
    user_agg = transactions_df.groupby('user_id').agg({
        'amount': ['mean', 'std', 'min', 'max', 'sum'],
        'transaction_id': 'count'
    }).reset_index()
    
    user_agg.columns = [
        'user_id',
        'avg_transaction_amount',
        'std_transaction_amount',
        'min_transaction_amount',
        'max_transaction_amount',
        'total_spent',
        'transaction_count'
    ]
    
    # Recency features
    latest_transaction = transactions_df.groupby('user_id')['timestamp'].max().reset_index()
    latest_transaction['days_since_last_transaction'] = (
        pd.Timestamp.now() - latest_transaction['timestamp']
    ).dt.days
    
    # Combine features
    features = transactions_df.merge(user_agg, on='user_id')
    features = features.merge(latest_transaction[['user_id', 'days_since_last_transaction']], on='user_id')
    features = features.merge(users_df, on='user_id')
    
    return features
```

### 4. Model Training

```python
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def train_model(features_df, target_col):
    """
    Train and log model with MLflow
    """
    
    # Split data
    X = features_df.drop([target_col, 'transaction_id', 'user_id', 'timestamp'], axis=1)
    y = features_df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Start MLflow run
    with mlflow.start_run():
        # Train model
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=50,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # Evaluate
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        # Log parameters
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 10)
        mlflow.log_param("training_samples", len(X_train))
        
        # Log metrics
        mlflow.log_metric("train_accuracy", train_score)
        mlflow.log_metric("test_accuracy", test_score)
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        return model
```

### 5. Model Deployment

**Containerized Deployment**:
```dockerfile
# Dockerfile for model serving
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model and code
COPY model/ ./model/
COPY serve.py .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run server
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Serving API**:
```python
# serve.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.pyfunc
import pandas as pd

app = FastAPI()

# Load model at startup
model = None

@app.on_event("startup")
async def load_model():
    global model
    model = mlflow.pyfunc.load_model("model/")

class PredictionRequest(BaseModel):
    features: dict

class PredictionResponse(BaseModel):
    prediction: float
    probability: float

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        # Convert to DataFrame
        input_df = pd.DataFrame([request.features])
        
        # Predict
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]
        
        return PredictionResponse(
            prediction=float(prediction),
            probability=float(probability)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": model is not None}
```

### Pipeline Orchestration

**Kubeflow Pipelines**:
```python
from kfp import dsl
from kfp.components import create_component_from_func

@create_component_from_func
def ingest_data(output_path: str):
    # Data ingestion logic
    pass

@create_component_from_func
def validate_data(input_path: str, output_path: str):
    # Validation logic
    pass

@create_component_from_func
def train_model(input_path: str, model_output_path: str):
    # Training logic
    pass

@dsl.pipeline(
    name='ML Training Pipeline',
    description='End-to-end ML pipeline'
)
def ml_pipeline():
    # Define pipeline
    ingest_op = ingest_data(output_path='/data/raw')
    
    validate_op = validate_data(
        input_path=ingest_op.outputs['output_path'],
        output_path='/data/validated'
    )
    
    train_op = train_model(
        input_path=validate_op.outputs['output_path'],
        model_output_path='/models/latest'
    )
```

## Training vs Inference

### Training Characteristics

**High Compute, Low Frequency**:
- Runs periodically (daily, weekly)
- Uses powerful GPU clusters
- Processes entire datasets
- Requires gradient computation
- Can take hours to days

**Example Training Setup**:
```python
# Distributed training with PyTorch DDP
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def train_distributed():
    # Initialize process group
    dist.init_process_group(backend='nccl')
    
    # Create model and move to GPU
    model = MyLargeModel().cuda()
    model = DDP(model)
    
    # Training loop
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            inputs, labels = batch
            inputs, labels = inputs.cuda(), labels.cuda()
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

# Launch with: torchrun --nproc_per_node=8 train.py
```

### Inference Characteristics

**Low Compute, High Frequency**:
- Runs continuously (24/7)
- Uses CPUs or single GPU
- Processes individual requests
- No gradients needed
- Must be < 100ms typically

**Optimized Inference**:
```python
# TorchScript for production inference
import torch

class OptimizedModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model.eval()
    
    @torch.jit.export
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.model(x)

# Compile model
model = MyModel()
optimized = OptimizedModel(model)
traced = torch.jit.script(optimized)

# Save
traced.save("optimized_model.pt")

# Load and use
loaded_model = torch.jit.load("optimized_model.pt")
prediction = loaded_model.predict(input_tensor)
```

### Training-Serving Skew

**Common Sources**:

1. **Different preprocessing**:
```python
# BAD: Different preprocessing in training vs serving
# Training (Python/Pandas)
df['normalized_amount'] = (df['amount'] - df['amount'].mean()) / df['amount'].std()

# Serving (Different implementation)
normalized = (amount - HARDCODED_MEAN) / HARDCODED_STD  # Wrong if data changes!
```

**Solution**: Use same codebase:
```python
# GOOD: Shared preprocessing
from sklearn.preprocessing import StandardScaler
import joblib

# Training
scaler = StandardScaler()
train_features = scaler.fit_transform(train_data)
joblib.dump(scaler, 'scaler.pkl')

# Serving
scaler = joblib.load('scaler.pkl')
inference_features = scaler.transform(new_data)  # Same transformation!
```

2. **Feature computation differences**:
```python
# BAD: Complex feature logic duplicated
# Training: SQL query aggregates
# Serving: Application code aggregates
# Slight differences cause skew!
```

**Solution**: Use Feature Store:
```python
# Training and serving use same feature store
from feast import FeatureStore

store = FeatureStore("feature_repo/")

# Training
training_features = store.get_historical_features(
    entity_df=entity_df,
    features=["user_features:avg_purchase_amount"]
)

# Serving
online_features = store.get_online_features(
    features=["user_features:avg_purchase_amount"],
    entity_rows=[{"user_id": 123}]
)
```

## Batch, Real-time & Streaming

### Batch Inference

**When to use**: Large volumes, no latency requirements

```python
# Spark batch inference
from pyspark.sql import SparkSession
import mlflow

spark = SparkSession.builder.appName("BatchInference").getOrCreate()

# Load model
model = mlflow.spark.load_model("models:/fraud_detector/production")

# Load data
data = spark.read.parquet("s3://data/transactions/2024-01/")

# Batch predict
predictions = model.transform(data)

# Write results
predictions.select("transaction_id", "prediction", "probability")\
    .write.mode("overwrite")\
    .parquet("s3://predictions/2024-01/")
```

**Cost optimization**:
- Use spot instances (70% cost savings)
- Process during off-peak hours
- Optimize batch sizes
- Cache models in memory

### Real-time Inference

**When to use**: Low latency required (< 100ms)

```python
from fastapi import FastAPI
import torch

app = FastAPI()

# Load model once
MODEL = torch.jit.load("model.pt")
MODEL.eval()

@app.post("/predict")
async def predict(features: dict):
    # Preprocess
    tensor = torch.tensor([list(features.values())])
    
    # Predict
    with torch.no_grad():
        prediction = MODEL(tensor)
    
    return {"prediction": prediction.item()}

# Optimize with batching
from collections import deque
import asyncio

request_queue = deque()

async def batch_predictor():
    while True:
        if len(request_queue) >= BATCH_SIZE or time_since_last_batch > MAX_WAIT:
            batch = [request_queue.popleft() for _ in range(min(BATCH_SIZE, len(request_queue)))]
            predictions = MODEL(torch.stack([r['tensor'] for r in batch]))
            
            for request, pred in zip(batch, predictions):
                request['future'].set_result(pred)
        
        await asyncio.sleep(0.001)

@app.post("/predict_batched")
async def predict_batched(features: dict):
    future = asyncio.Future()
    request_queue.append({'tensor': preprocess(features), 'future': future})
    return await future
```

**Optimization techniques**:
- Request batching (2-5x throughput improvement)
- Model quantization (INT8, 4x faster)
- TensorRT compilation (2-6x faster)
- KV caching for transformers
- Response caching for frequent queries

### Streaming Inference

**When to use**: Continuous data streams, window-based aggregations

```python
# Flink streaming inference
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.table import StreamTableEnvironment

env = StreamExecutionEnvironment.get_execution_environment()
t_env = StreamTableEnvironment.create(env)

# Define source
t_env.execute_sql("""
    CREATE TABLE transactions (
        user_id BIGINT,
        amount DOUBLE,
        merchant STRING,
        timestamp TIMESTAMP(3),
        WATERMARK FOR timestamp AS timestamp - INTERVAL '5' SECOND
    ) WITH (
        'connector' = 'kafka',
        'topic' = 'transactions',
        'properties.bootstrap.servers' = 'localhost:9092'
    )
""")

# Create features with windows
t_env.execute_sql("""
    CREATE VIEW user_features AS
    SELECT 
        user_id,
        HOP_END(timestamp, INTERVAL '1' MINUTE, INTERVAL '5' MINUTE) as window_end,
        COUNT(*) as txn_count_5min,
        SUM(amount) as total_spent_5min,
        AVG(amount) as avg_amount_5min
    FROM transactions
    GROUP BY user_id, HOP(timestamp, INTERVAL '1' MINUTE, INTERVAL '5' MINUTE)
""")

# Apply model (registered UDF)
t_env.execute_sql("""
    INSERT INTO fraud_predictions
    SELECT 
        user_id,
        detect_fraud(txn_count_5min, total_spent_5min, avg_amount_5min) as is_fraud,
        window_end
    FROM user_features
    WHERE detect_fraud(txn_count_5min, total_spent_5min, avg_amount_5min) = TRUE
""")
```

## Model Serving Patterns

### 1. Model-as-Service

**Architecture**:
```
Client → Load Balancer → [Service Instance 1]
                       → [Service Instance 2]
                       → [Service Instance N]
```

**Implementation with BentoML**:
```python
import bentoml
from bentoml.io import JSON

# Create service
svc = bentoml.Service("fraud_detector")

# Load model
fraud_model = bentoml.pytorch.get("fraud_model:latest")

@svc.api(input=JSON(), output=JSON())
def predict(input_data):
    # Preprocess
    features = preprocess(input_data)
    
    # Predict
    prediction = fraud_model.run(features)
    
    return {
        "is_fraud": bool(prediction > 0.5),
        "fraud_score": float(prediction),
        "model_version": fraud_model.tag.version
    }

# Deploy
# bentoml serve service.py:svc --production
```

**Kubernetes deployment**:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fraud-detector
spec:
  replicas: 3
  selector:
    matchLabels:
      app: fraud-detector
  template:
    metadata:
      labels:
        app: fraud-detector
    spec:
      containers:
      - name: model-server
        image: myregistry/fraud-detector:v1.0
        ports:
        - containerPort: 3000
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
        readinessProbe:
          httpGet:
            path: /readyz
            port: 3000
          initialDelaySeconds: 10
        livenessProbe:
          httpGet:
            path: /healthz
            port: 3000
---
apiVersion: v1
kind: Service
metadata:
  name: fraud-detector-service
spec:
  selector:
    app: fraud-detector
  ports:
  - port: 80
    targetPort: 3000
  type: LoadBalancer
```

### 2. Embedded Model

**When to use**: Mobile apps, edge devices, latency-critical

```python
# Convert to ONNX for cross-platform deployment
import torch.onnx

# PyTorch model
model = MyModel()
model.eval()

# Export to ONNX
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    export_params=True,
    opset_version=13,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)

# Load with ONNX Runtime (C++, Python, JavaScript, etc.)
import onnxruntime as ort

session = ort.InferenceSession("model.onnx")
input_name = session.get_inputs()[0].name

prediction = session.run(None, {input_name: input_data.numpy()})
```

**Mobile deployment (TensorFlow Lite)**:
```python
import tensorflow as tf

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Quantize
tflite_model = converter.convert()

# Save
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

# Use in mobile app (pseudo-code)
# Android/iOS:
# interpreter = Interpreter("model.tflite")
# interpreter.allocateTensors()
# interpreter.setInput(inputData)
# interpreter.run()
# output = interpreter.getOutput()
```

### 3. Serverless Inference

**AWS Lambda example**:
```python
import json
import boto3
import numpy as np
import onnxruntime as ort

# Global model (loaded once per container)
SESSION = None

def load_model():
    global SESSION
    if SESSION is None:
        s3 = boto3.client('s3')
        s3.download_file('my-models', 'model.onnx', '/tmp/model.onnx')
        SESSION = ort.InferenceSession('/tmp/model.onnx')
    return SESSION

def lambda_handler(event, context):
    try:
        # Load model
        session = load_model()
        
        # Parse input
        body = json.loads(event['body'])
        input_data = np.array(body['features'], dtype=np.float32).reshape(1, -1)
        
        # Predict
        input_name = session.get_inputs()[0].name
        prediction = session.run(None, {input_name: input_data})[0]
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'prediction': float(prediction[0]),
                'latency_ms': context.get_remaining_time_in_millis()
            })
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
```

**Cold start optimization**:
- Keep model small (< 250MB for Lambda)
- Use provisioned concurrency for critical paths
- Optimize container images
- Lazy load only what's needed

### Comparison

| Pattern | Latency | Cost | Scalability | Use Case |
|---------|---------|------|-------------|----------|
| Model-as-Service | 10-50ms | Medium | High | General serving |
| Embedded | < 5ms | Low | Medium | Mobile, edge |
| Serverless | 50-200ms | Very Low* | Very High | Sporadic traffic |

*Pay per use, cost-effective for <10% utilization

## Feature Stores

A feature store solves the critical problem of consistent features between training and serving.

### Architecture

```
┌──────────────────────────────────────────────────────┐
│                  Feature Store                        │
├───────────────┬────────────────┬────────────────────┤
│ Offline Store │  Online Store  │  Feature Registry  │
│ (Training)    │  (Serving)     │  (Metadata)        │
│               │                │                     │
│ - BigQuery    │  - Redis       │  - Definitions     │
│ - Snowflake   │  - DynamoDB    │  - Lineage         │
│ - S3+Parquet  │  - Cassandra   │  - Schemas         │
│ - Time travel │  - Low latency │  - Documentation   │
└───────────────┴────────────────┴────────────────────┘
```

### Feast Example

**Define features**:
```python
# feature_repo/features.py
from feast import Entity, Feature, FeatureView, Field, FileSource
from feast.types import Float64, Int64
from datetime import timedelta

# Define entity
user = Entity(
    name="user",
    join_keys=["user_id"]
)

# Define source (offline)
user_transactions_source = FileSource(
    path="s3://data/user_transactions.parquet",
    timestamp_field="timestamp"
)

# Define feature view
user_transaction_features = FeatureView(
    name="user_transaction_features",
    entities=[user],
    ttl=timedelta(days=90),
    schema=[
        Field(name="total_transactions", dtype=Int64),
        Field(name="avg_transaction_amount", dtype=Float64),
        Field(name="total_spent", dtype=Float64),
        Field(name="days_since_last_transaction", dtype=Int64),
    ],
    online=True,
    source=user_transactions_source
)
```

**Training (get historical features)**:
```python
from feast import FeatureStore
import pandas as pd

store = FeatureStore("feature_repo/")

# Entity dataframe (users and timestamps for training)
entity_df = pd.read_parquet("s3://data/training_labels.parquet")
# Columns: user_id, timestamp, label

# Get point-in-time correct features
training_df = store.get_historical_features(
    entity_df=entity_df,
    features=[
        "user_transaction_features:total_transactions",
        "user_transaction_features:avg_transaction_amount",
        "user_transaction_features:days_since_last_transaction"
    ]
).to_df()

# No data leakage - features only use data available at each timestamp!

# Train model
X = training_df.drop(['user_id', 'timestamp', 'label'], axis=1)
y = training_df['label']
model.fit(X, y)
```

**Serving (get online features)**:
```python
# Materialize features to online store
store.materialize(
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 1, 15),
    feature_views=["user_transaction_features"]
)

# Get features for real-time prediction
online_features = store.get_online_features(
    features=[
        "user_transaction_features:total_transactions",
        "user_transaction_features:avg_transaction_amount"
    ],
    entity_rows=[
        {"user_id": 1001},
        {"user_id": 1002}
    ]
).to_dict()

# Use for prediction
predictions = model.predict(online_features)
```

### Feature Engineering Patterns

**Batch features** (computed periodically):
```python
# Daily Spark job
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = SparkSession.builder.appName("FeatureEngineering").getOrCreate()

# Load transactions
transactions = spark.read.parquet("s3://data/transactions/")

# Compute aggregations
user_features = transactions.groupBy("user_id").agg(
    F.count("*").alias("total_transactions"),
    F.mean("amount").alias("avg_transaction_amount"),
    F.sum("amount").alias("total_spent"),
    F.datediff(F.current_date(), F.max("date")).alias("days_since_last_transaction")
)

# Write to offline store
user_features.write.parquet("s3://features/user_transaction_features/")

# Materialize to online store (Redis)
feast_client.write_to_online_store(user_features)
```

**Streaming features** (computed in real-time):
```python
# Flink streaming feature computation
from pyflink.table import StreamTableEnvironment

t_env = StreamTableEnvironment.create(env)

# Source
t_env.execute_sql("""
    CREATE TABLE transactions (
        user_id BIGINT,
        amount DOUBLE,
        timestamp TIMESTAMP(3),
        WATERMARK FOR timestamp AS timestamp - INTERVAL '5' SECOND
    ) WITH (
        'connector' = 'kafka',
        'topic' = 'transactions'
    )
""")

# Compute sliding window features
t_env.execute_sql("""
    CREATE VIEW user_recent_activity AS
    SELECT 
        user_id,
        HOP_END(timestamp, INTERVAL '1' MINUTE, INTERVAL '10' MINUTE) as window_end,
        COUNT(*) as transactions_last_10min,
        SUM(amount) as total_spent_last_10min,
        AVG(amount) as avg_amount_last_10min
    FROM transactions
    GROUP BY user_id, HOP(timestamp, INTERVAL '1' MINUTE, INTERVAL '10' MINUTE)
""")

# Write to online store (Redis)
t_env.execute_sql("""
    INSERT INTO redis_online_store
    SELECT * FROM user_recent_activity
""")
```

**On-demand features** (computed at request time):
```python
# Feast on-demand features
from feast import on_demand_feature_view, Field
from feast.types import Float64

@on_demand_feature_view(
    sources=[user_transaction_features],
    schema=[
        Field(name="spending_velocity", dtype=Float64),
        Field(name="transaction_frequency", dtype=Float64)
    ]
)
def derived_features(inputs: pd.DataFrame) -> pd.DataFrame:
    output = pd.DataFrame()
    
    # Compute features from stored features
    output["spending_velocity"] = (
        inputs["total_spent"] / 
        (inputs["days_since_last_transaction"] + 1)
    )
    
    output["transaction_frequency"] = (
        inputs["total_transactions"] / 
        (inputs["days_since_first_transaction"] + 1)
    )
    
    return output
```

---

[CONTINUED IN PART 2...]

**Note**: This tutorial continues with detailed coverage of:
- Model Registry (MLflow, Weights & Biases)
- Data Engineering (validation, quality, versioning)
- Distributed Training (DDP, model parallel, FSDP)
- LLM Systems (transformer architecture, inference optimization)
- RAG Architectures (vector DBs, chunking, evaluation)
- Agentic AI (ReAct, tool use, multi-agent systems)
- Production Case Studies (fraud detection, recommendations, conversational AI)

The complete tutorial is ~40,000 words covering every aspect of production ML systems with code examples, architectural diagrams, and battle-tested patterns from real-world deployments.

---

## Quick Reference

### Essential ML Metrics

**Classification**:
- Accuracy: (TP + TN) / Total
- Precision: TP / (TP + FP)
- Recall: TP / (TP + FN)
- F1: 2 * (Precision * Recall) / (Precision + Recall)
- AUC-ROC: Area under ROC curve

**Regression**:
- MAE: Mean Absolute Error
- RMSE: Root Mean Squared Error  
- R²: Coefficient of determination
- MAPE: Mean Absolute Percentage Error

**Ranking**:
- NDCG: Normalized Discounted Cumulative Gain
- MAP: Mean Average Precision
- MRR: Mean Reciprocal Rank

### Popular Frameworks

**Training**: PyTorch, TensorFlow, JAX, scikit-learn, XGBoost, LightGBM
**Serving**: TorchServe, TensorFlow Serving, BentoML, Seldon, KServe
**Orchestration**: Kubeflow, MLflow, Airflow, Prefect, Metaflow
**Feature Stores**: Feast, Tecton, Hopsworks
**Monitoring**: Evidently, Arize, WhyLabs, Fiddler

### Best Practices Checklist

✅ Version everything (code, data, models, configs)  
✅ Start with simple baselines  
✅ Monitor data quality continuously  
✅ Implement comprehensive testing  
✅ Use feature stores for consistency  
✅ Track experiments systematically  
✅ Deploy with rollback capability  
✅ Monitor for drift  
✅ Automate retraining pipelines  
✅ Document decisions and trade-offs  

---

**Repository**: [github.com/your-username/ml-system-design](https://github.com)
**License**: MIT
**Contributing**: Pull requests welcome!

---

*"The best ML system is the one that solves the business problem reliably at acceptable cost." - Ancient ML Wisdom*
