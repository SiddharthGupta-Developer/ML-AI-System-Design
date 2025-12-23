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
- [Recommendation Engine](#recommendation-engine)


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
# Chapter I: ML Architecture Fundamentals

## Machine Learning Pipeline

An ML pipeline automates the complete workflow from raw data to production predictions. It encompasses data ingestion, feature engineering, model training, deployment, and monitoring with built-in version control and reproducibility.

**Core Components:**

**Data Ingestion** - Extracts data from sources (databases, APIs, streams), validates schemas, performs initial quality checks, and routes to storage. Must be idempotent to enable safe retries.

**Data Processing** - Transforms raw data into model-ready features through cleaning, normalization, encoding, and aggregation. Critical requirement: identical transformation logic for training and serving to prevent training-serving skew.

**Training Layer** - Manages experiments, hyperparameter optimization, distributed training, and model validation. Tracks all metadata (data versions, hyperparameters, metrics) for reproducibility.

**Model Registry** - Central repository storing model artifacts, metadata, lineage, and deployment status. Implements versioning and lifecycle management (experimental → validated → staging → production → archived).

**Serving Layer** - Deploys models for inference with load balancing, request batching, caching, and monitoring. Optimized for latency and throughput requirements.

**Monitoring Layer** - Tracks model performance, data quality, system health, and detects drift. Triggers alerts and automated retraining when degradation occurs.

**Pipeline Orchestration** uses Directed Acyclic Graphs (DAGs) where nodes represent processing steps and edges represent dependencies. Orchestration systems handle scheduling, retry logic, resource allocation, and parallel execution. Every run logs comprehensive metadata: data versions, code commits, hyperparameters, metrics, and execution timing.

---

## Training vs Inference

### Training Phase

**Computational Characteristics:**

Training is throughput-oriented, processing large batches (32-512+ samples) to maximize GPU utilization. Requires GPUs/TPUs for parallel matrix operations. Jobs run for hours to weeks. Memory bandwidth is often the bottleneck.

**Memory Consumption:**

Total memory = Model Parameters + Gradients + Optimizer States + Activations + Batch Data

For a 1B parameter model with Adam optimizer:
- Parameters: 4GB (float32)
- Gradients: 4GB (same size as parameters)
- Optimizer states: 8GB (2x parameters for Adam's momentum and variance)
- Activations: 8-16GB (grows with batch size and depth)
- Total: ~24-32GB

**Optimization Techniques:**

**Mixed Precision Training** - Uses float16 for computation and float32 for accumulation. Reduces memory by ~50% and speeds up training on modern GPUs with tensor cores. Applies loss scaling to prevent underflow.

**Gradient Checkpointing** - Trades computation for memory by not storing all activations. Recomputes activations during backward pass. Enables training larger models or bigger batches at cost of ~30% slower training.

**Distributed Training** - Splits work across multiple devices:
- Data Parallelism: Different devices process different data batches with same model
- Model Parallelism: Different devices hold different model parts
- Pipeline Parallelism: Splits model into stages, each on different device
- ZeRO (Zero Redundancy Optimizer): Partitions optimizer states, gradients, and parameters across devices

**Gradient Accumulation** - Simulates larger batches by accumulating gradients over multiple forward-backward passes before updating weights. Useful when memory limits batch size.

**Learning Rate Scheduling** - Adjusts learning rate during training:
- Warm-up: Gradually increases LR from zero to avoid early instability
- Cosine Annealing: Smoothly decreases LR following cosine curve
- Step Decay: Reduces LR at fixed intervals
- ReduceLROnPlateau: Decreases when metrics plateau

### Inference Phase

**Computational Characteristics:**

Latency-oriented, processing single samples or small batches (1-32). Often runs on CPUs with optimizations. Latency ranges from <10ms (ad serving) to seconds (content moderation) depending on use case.

**Memory:** Only needs model parameters and current batch activations. Same 1B parameter model requires 4GB (float32) or 2GB (float16) or 0.5GB (int8 quantized) plus 100MB-1GB activations. Total can be <1GB with optimizations.

**Inference Optimization Techniques:**

**Quantization** - Reduces numerical precision of weights and activations:
- Post-Training Quantization: Converts trained float32 model to int8 without retraining. Simple but slight accuracy loss.
- Quantization-Aware Training: Simulates quantization during training for better accuracy. Model learns to be robust to reduced precision.
- Dynamic Quantization: Quantizes weights ahead of time, activations dynamically at runtime.
- Common formats: int8 (4x smaller), int4 (8x smaller), mixed-precision (critical layers stay float16)

**Pruning** - Removes unnecessary model parameters:
- Unstructured Pruning: Removes individual weights below threshold. High compression but requires specialized hardware.
- Structured Pruning: Removes entire neurons, filters, or attention heads. Lower compression but efficient on standard hardware.
- Magnitude-based: Prunes smallest weights
- Iterative Pruning: Gradually increases sparsity with fine-tuning between steps

**Knowledge Distillation** - Trains smaller "student" model to mimic larger "teacher". Student learns from teacher's soft predictions (probability distributions) not just hard labels. Captures dark knowledge - nuanced decision boundaries teacher learned. Typical compression: 10-100x smaller with <5% accuracy loss.

**Operator Fusion** - Combines multiple operations into single kernel. Example: Fusing matrix multiplication + bias addition + activation into one GPU kernel. Reduces memory transfers between operations.

**KV-Cache Optimization** - For autoregressive models (GPT, LLama):
- Problem: Each new token generation requires recomputing attention over all previous tokens
- Solution: Cache key and value matrices from previous tokens
- Memory trade-off: Stores KV cache (grows with sequence length) to avoid redundant computation
- Multi-Query Attention (MQA): Shares key/value heads across attention heads to reduce KV cache size
- Grouped-Query Attention (GQA): Middle ground between full and multi-query attention

**Batch Inference** - Groups multiple requests. Dynamic Batching accumulates requests arriving within time window. Increases throughput significantly on GPUs. Adaptive batch sizing based on model, hardware, and latency requirements.

**Model Compilation** - Optimizes model graph by fusing operators, removing redundant computations, optimizing memory allocation, and generating hardware-specific code (TensorRT for NVIDIA, OpenVINO for Intel).

**Early Exit** - Adds intermediate classifiers. Simple inputs exit early from shallow layers while complex inputs use full model depth. Reduces average inference time while maintaining accuracy.

---

## Batch, Real-time, and Streaming Processing

### Batch Processing

Processes complete datasets accumulated over time windows. Data collected then processed all at once, typically scheduled (hourly/daily/weekly).

**Characteristics:** High latency (hours to days), high throughput (petabytes), complete view, cost-effective (spot instances, off-peak processing).

**Use Cases:** Model training on historical data, customer lifetime value, daily product recommendations, segmentation models, bulk scoring for campaigns.

### Real-Time Processing

Handles individual requests synchronously with immediate responses (milliseconds to seconds).

**Characteristics:** Low latency (sub-second to seconds), request-response pattern, fresh context, always-on infrastructure.

**Use Cases:** Fraud detection, search ranking, ad serving, chatbots, page-load recommendations, content moderation.

**Latency Breakdown:** Network (5-20ms) + Feature retrieval (5-30ms) + Model inference (10-100ms) + Postprocessing (1-5ms) = Total 20-150ms typical.

### Streaming Processing

Processes continuous event flows with stateful computations over time windows. Combines low latency with temporal aggregation.

**Characteristics:** Near real-time (seconds), stateful, continuous, temporal context.

**Use Cases:** Click-through rate calculation, trending topics, continuous fraud scoring, IoT sensor analysis, real-time user profiling.

**Windowing Strategies:**

**Tumbling Windows** - Fixed, non-overlapping intervals. Each event in exactly one window. For periodic aggregations (hourly counts, daily sums).

**Sliding Windows** - Overlapping windows advancing incrementally. Events in multiple windows. For moving averages, continuous trends.

**Session Windows** - Activity-based with gaps defining boundaries. Windows adapt to behavior. For user sessions, transaction bursts.

**Event Time vs Processing Time:**

Event time: When event actually occurred (embedded in event)
Processing time: When system processes event
Watermarks track event time progress, determining when windows close. Late events handled via allowed lateness or side outputs.

### Lambda Architecture

Combines batch and streaming for accuracy with low latency.

**Components:**
- Batch Layer: Complete historical data, accurate comprehensive results (high latency, high accuracy)
- Speed Layer: Recent data only, approximate real-time results (low latency, eventual accuracy)
- Serving Layer: Merges results, prioritizes speed layer for recent data

**Example:** Recommendation system uses daily batch computation of user-item affinity from all interactions plus real-time session click tracking, serving blends both.

---

## Model Serving Patterns

### Embedded Model

Model packaged within application, loaded at startup, inference in-process.

**When to Use:** Mobile apps (TensorFlow Lite, ONNX Runtime), edge devices, microservices needing <1ms latency, privacy-sensitive applications, offline functionality.

**Advantages:** Zero network latency, offline capability, complete privacy, simple deployment, no separate servers.

**Limitations:** Model size constraints, updates require app redeployment, resource competition, difficult A/B testing.

### Model as Service

Models deployed as independent services accessed via REST/gRPC APIs.

**When to Use:** Multiple applications share model, complex models need GPUs, frequent model updates, sophisticated monitoring, A/B testing, high-traffic scenarios.

**Advantages:** Independent scaling, easy updates, shared expensive hardware, centralized monitoring, version control, canary deployments.

**Limitations:** Network latency (5-50ms), operational complexity, potential single point of failure, infrastructure costs.

**Optimization Techniques:**

**Dynamic Batching** - Accumulates requests within timeout window (e.g., 10ms), processes as batch. Increases GPU throughput. Trade-off: Slight latency for better efficiency.

**Model Warmup** - Pre-loads model and runs dummy predictions before traffic. Avoids cold-start latency.

**Multi-Model Serving** - Single server hosts multiple models, shares resources. For A/B testing or ensembles.

**Prediction Caching** - Stores results for duplicate requests with TTL. Effective for popular queries.

### Batch Serving

Precomputes predictions for known entities, stores in database, serves via lookup.

**When to Use:** Finite entity set (all users/products), staleness tolerable (hours/days), expensive inference, predictable requests.

**Use Cases:** Daily user recommendations, monthly churn predictions, email engagement scoring, content pre-ranking.

**Advantages:** Ultra-low serving latency (1-5ms lookup), cost-effective computation amortization, simple infrastructure, enables prediction review.

**Limitations:** Stale predictions, storage requirements, cannot handle new entities immediately, wasted computation for unused predictions.

**Hybrid Pattern:** Batch provides baseline (daily updates) + Real-time adjusts for current session = Final score combines both (e.g., 0.7 * batch + 0.3 * realtime).

### Streaming Serving

Models process event streams, enriching events with predictions real-time.

**When to Use:** Events need ML enrichment, real-time fraud/anomaly detection, event-driven architectures, requires temporal context.

**Use Cases:** Transaction fraud scoring, live stream content moderation, IoT predictive maintenance, ad bidding, network security monitoring.

**State Management:** Processor maintains feature state (e.g., user's last 10 transactions). State in embedded databases (RocksDB) with periodic checkpointing. Enables stateful features like running averages, distinct counts, sequence patterns.

---

## Feature Stores

Centralized repository managing ML features, solving training-serving skew by ensuring consistent feature computation across training and inference.

### Core Problem: Training-Serving Skew

Features computed differently during training (Python/Pandas) vs serving (Java/Go) cause model degradation despite good offline metrics. Feature Store uses identical transformation logic for both.

### Architecture Components

**Feature Registry** - Catalog of feature definitions: name, type, entity, transformation logic, data sources, update schedule, owner, version.

**Offline Store** - Historical feature storage for training (S3, BigQuery, Hive). Point-in-time correct values. Large-scale batch reads.

**Online Store** - Low-latency serving storage (Redis, DynamoDB, Cassandra). Latest feature values. Single-key lookups <10ms p99.

**Transformation Engine** - Computes features from raw data. Same code generates features for offline (training) and online (serving).

### Point-in-Time Correctness

Features must reflect values as they existed at each training example's timestamp, not current/future values. Prevents data leakage.

Example: Training example at 2024-01-15 predicting churn.
- Correct: user_transaction_count_30d uses transactions from 2023-12-16 to 2024-01-15
- Incorrect: Using current count includes future data, inflates offline metrics, fails in production

### Feature Types

**Batch Features** - Computed periodically (daily/hourly) from large datasets. Examples: user_total_purchases, product_avg_rating. Stored offline, synced to online.

**Streaming Features** - Computed real-time from event streams. Examples: user_clicks_last_hour, session_page_views. Directly updated online.

**On-Demand Features** - Computed at request time from context. Examples: time_since_last_login, is_weekend. Computed during serving, not stored.

### Feature Engineering Patterns

**Entity-Centric:** User (age, country, total_purchases), Product (category, price, rating), Transaction (amount, payment_method).

**Time-Windowed Aggregations:** user_purchases_count_7d / 30d / 90d, product_revenue_sum_24h.

**Ratio Features:** conversion_rate = purchases / views, cart_abandonment_rate = (carts - orders) / carts.

### Workflows

**Training:** Define features → Backfill historical values → Request features with entity IDs and timestamps → Point-in-time joins → Return dataset → Train model.

**Serving:** Receive request with entity IDs → Query online store (GET user:123:features) → Return in <10ms → Run inference → Return prediction.

### Feature Versioning

Features evolve: user_ltv_v1 (sum of purchases) → v2 (purchases minus refunds) → v3 (new formula with retention). Registry tracks which model uses which feature version.

### Feature Monitoring

**Data Quality:** Null rate, schema compliance, value distribution shifts.
**Freshness:** Last update timestamp, lag from source, staleness alerts.
**Consistency:** Online-offline comparison, training-serving parity tests.

---

## Model Registry

Central catalog managing ML model lifecycle: artifacts, metadata, versioning, lineage, deployment tracking.

### Core Functions

**Model Versioning** - Semantic versioning (major.minor.patch) for each trained model. Tracks complete lineage.

**Artifact Storage** - Model files (weights, configs), preprocessing artifacts (scalers, encoders, tokenizers), postprocessing logic. Supports TensorFlow SavedModel, PyTorch, ONNX, pickle.

**Metadata Management** - Training data version, feature definitions, hyperparameters, metrics (accuracy, loss curves), validation results, training duration, compute resources, framework versions, owner/team.

**Lifecycle Management** - Stages: Experimental (initial training) → Validated (passed evaluation) → Staging (test environment) → Production (live traffic) → Archived (deprecated).

**Access Control** - Permissions for registration, stage promotion, production deployment, artifact downloads.

### Model Registration Workflow

Training completes → If metrics meet thresholds, register via API → Registry assigns version, stores artifacts (S3), records metadata → Model enters "Experimental" → Review and promote to "Validated" → Deploy to staging ("Staging") → After validation, promote to "Production" → Old production becomes "Archived".

### Lineage Tracking

Maps model to training dataset (version, location, schema), feature definitions, code repository (commit hash), parent models (for fine-tuning), hyperparameters. Answers: "What data trained this?" "Which models use this feature?" "What changed between versions?"

### Model Comparison

Compare models across metrics (accuracy, precision, recall, F1, AUC, KPIs), latency (inference time p50/p95/p99), resource usage (memory, CPU/GPU), training cost (compute hours, cloud spend), data requirements (size, feature counts). Supports A/B testing decisions.

### Deployment Metadata

Tracks deployment environment (staging/production/edge), deployment timestamp, traffic percentage (gradual rollouts), performance metrics (online accuracy, latency, throughput, errors), rollback status (automatic rollback if metrics degrade).

### Integration

**Training:** Framework auto-registers models post-training via SDK/API.
**CI/CD:** Deployment pipelines query registry for latest validated model. Automated tests before promotion.
**Monitoring:** Production systems report metrics back. Registry tracks online vs offline divergence. Triggers retraining on degradation.

### Model Cards

Documentation stored in registry: model purpose, training data characteristics (size, distribution, biases), performance metrics and limitations, ethical considerations, maintenance schedule, contact information. Essential for governance, compliance, collaboration.

# Chapter II: Data Engineering for ML

## Data Collection & Ingestion

### Data Sources

**Structured Data** - Relational databases (PostgreSQL, MySQL), data warehouses (Snowflake, BigQuery). Defined schemas, typed columns, relational integrity.

**Semi-Structured Data** - JSON from APIs, XML, logs, CSV. Flexible schemas, nested structures, varying field presence.

**Unstructured Data** - Text, images, videos, audio. No inherent structure, requires feature extraction.

**Streaming Data** - Real-time events from Kafka/Kinesis, IoT sensors, clickstreams. Continuous flow, temporal ordering.

### Ingestion Patterns

**Batch Ingestion** - Scheduled periodic extraction. Uses ETL tools (Airflow, Prefect). Suitable for historical datasets.

**Change Data Capture (CDC)** - Tracks and streams database changes. Uses transaction logs (binlog, WAL). Captures inserts/updates/deletes. Tools: Debezium, Maxwell.

**API Ingestion** - Pulls data from REST/GraphQL APIs. Implements rate limiting, retry logic, pagination.

**Event Streaming** - Consumes real-time events from message queues. Uses consumer groups for parallel processing.

### Schema Management

**Schema Evolution** - Handles changes: adding fields (backward compatible), removing fields (forward compatible), changing types (breaking). Uses schema registry (Avro, Protobuf).

**Schema Validation** - Validates incoming data against expected schema. Checks data types, required fields, value ranges.

---

## Data Quality & Validation

### Data Quality Dimensions

**Completeness** - Presence of required data. Metrics: null rate, missing value percentage.

**Accuracy** - Correctness of values. Checks: value ranges, format validation, referential integrity.

**Consistency** - Agreement across sources. Validates: cross-field dependencies, temporal consistency.

**Timeliness** - Data freshness. Monitors: ingestion lag, update frequency.

**Uniqueness** - Absence of duplicates. Detects exact and fuzzy duplicates.

### Validation Techniques

**Statistical Validation** - Compares distribution to historical baselines. Monitors mean, median, standard deviation, percentiles. Outlier detection using IQR or z-scores.

**Rule-Based Validation** - Applies business rules: age between 0-120, email regex pattern, positive amounts, referential integrity.

**Constraint Validation** - Enforces NOT NULL, UNIQUE, CHECK constraints, foreign keys.

### Handling Data Quality Issues

**Missing Values:**
- Deletion: Remove records (risk: reduces size, introduces bias)
- Mean/Median Imputation: Fill with central tendency (assumes MAR - Missing At Random)
- Mode Imputation: Most frequent value for categorical
- Forward Fill: Use previous value (time-series)
- Model-Based Imputation: MICE (Multiple Imputation by Chained Equations)
- Missingness Indicator: Binary flag preserving missingness pattern signal

**Outliers:**
- Capping/Winsorizing: Replace beyond thresholds with threshold values
- Transformation: Log/sqrt to compress range
- Removal: Delete extremes (loses information)
- Separate Model: Different models for outliers vs normal

**Duplicates:**
- Exact: Remove based on key columns
- Fuzzy: Similarity matching (Levenshtein distance, Jaccard similarity)

**Data Drift Detection:**
- Monitors distribution changes over time
- Statistical tests: Kolmogorov-Smirnov, Chi-square
- Divergence metrics: KL divergence, JS divergence, Wasserstein distance
- Triggers retraining when drift exceeds threshold

---

## Feature Engineering

### Numeric Transformations

**Log Transform** - Handles skewed distributions. log(x + 1) handles zeros.

**Power Transform** - Box-Cox (positive values), Yeo-Johnson (handles negative).

**Polynomial Features** - x², x³, interactions (x1 * x2).

**Binning/Discretization** - Converts continuous to categorical. Equal-width or equal-frequency bins.

### Categorical Encoding

**Label Encoding** - Maps categories to integers. For ordinal variables only.

**One-Hot Encoding** - Binary column per category. Use when <20 categories.

**Target Encoding** - Replaces category with mean target for that category. Risk: leakage, needs regularization.

**Frequency Encoding** - Replaces with frequency/proportion in dataset.

**Binary Encoding** - Converts to binary representation. Efficient for high cardinality.

**Hashing Trick** - Hashes categories to fixed-size vector. Handles unknown categories.

### Text Features

**Bag-of-Words** - Count of each word. Sparse, high-dimensional.

**TF-IDF** - Term Frequency - Inverse Document Frequency. Down-weights common words.

**N-grams** - Captures sequences (bigrams, trigrams). "machine learning" as single feature.

**Word Embeddings** - Dense vectors (Word2Vec, GloVe, FastText). Captures semantic similarity.

**Sentence Embeddings** - BERT, Sentence-BERT for full text representation.

### Temporal Features

**Date Components** - Year, month, day, day_of_week, hour, quarter.

**Cyclical Encoding** - Sin/cos transformation for cyclic features. hour_sin = sin(2π * hour / 24).

**Time Since Event** - Days since last purchase, hours since signup.

**Time-Based Aggregations** - Rolling windows (7-day average), expanding windows (all-time total).

**Lag Features** - Previous values (t-1, t-7, t-30) for time-series.

### Aggregation Features

**Group Statistics** - Per-entity aggregations: user_avg_purchase_amount, product_total_views.

**Cross-Feature Aggregations** - user_purchases_per_category, merchant_transactions_by_country.

**Ratio Features** - conversion_rate = purchases / views, average_basket_size = items / transactions.

### Feature Scaling

**Normalization (Min-Max Scaling)** - Scales to [0, 1]: x_scaled = (x - min) / (max - min). Sensitive to outliers.

**Standardization (Z-score)** - Zero mean, unit variance: x_scaled = (x - mean) / std. Robust to outliers.

**Robust Scaling** - Uses median and IQR: x_scaled = (x - median) / IQR. Very robust to outliers.

**Max Abs Scaling** - Scales by maximum absolute value. Preserves zero entries, useful for sparse data.

---

## Handling Imbalanced Data

Class imbalance occurs when one class significantly outnumbers others. Common in fraud detection (99% legitimate, 1% fraud), medical diagnosis, anomaly detection.

### Evaluation Metrics for Imbalanced Data

**Accuracy Paradox** - High accuracy misleading with imbalance. Model predicting all negative achieves 99% accuracy on 99:1 dataset but useless.

**Better Metrics:**
- Precision: TP / (TP + FP) - Of predicted positives, how many correct?
- Recall: TP / (TP + FN) - Of actual positives, how many caught?
- F1-Score: Harmonic mean of precision and recall
- AUC-ROC: Area Under Receiver Operating Characteristic curve
- AUC-PR: Area Under Precision-Recall curve (better for severe imbalance)

### Resampling Techniques

**Random Undersampling** - Removes majority class samples. Simple but loses information. Risk: discards potentially useful data.

**Random Oversampling** - Duplicates minority class samples. Risk: overfitting to specific minority examples.

**SMOTE (Synthetic Minority Over-sampling Technique)** - Creates synthetic minority samples by interpolating between existing minority samples. For each minority sample, finds k nearest neighbors, randomly selects one, creates new sample along line between them.

**ADASYN (Adaptive Synthetic Sampling)** - Extension of SMOTE. Generates more synthetic samples for minority samples that are harder to learn (near decision boundary).

**Tomek Links** - Identifies and removes pairs of opposite-class samples that are close together. Cleans overlap between classes.

**Edited Nearest Neighbors (ENN)** - Removes majority samples whose class differs from majority of k nearest neighbors. Cleans noisy majority samples.

**Combination Approaches** - SMOTE + Tomek or SMOTE + ENN. Oversample minority then clean overlap.

### Algorithmic Approaches

**Class Weights** - Assigns higher penalty to misclassifying minority class. Most algorithms support class_weight parameter. Auto-balancing: weight = n_samples / (n_classes * class_count).

**Threshold Moving** - Adjusts decision threshold based on business cost. Default 0.5 may not be optimal for imbalanced data.

**Ensemble Methods** - Combine multiple models trained on different balanced subsets.

**BalancedBagging** - Undersamples majority class for each ensemble member.

**BalancedRandomForest** - Balances bootstrap samples when building trees.

**EasyEnsemble** - Creates multiple balanced subsets, trains classifier on each, aggregates predictions.

**Anomaly Detection Approaches** - For extreme imbalance (99.9% negative), treat as anomaly detection. One-Class SVM, Isolation Forest, Autoencoders.

---

## Data Versioning

### Why Version Data

**Reproducibility** - Retrain model with exact same data.

**Debugging** - Identify which data version caused model degradation.

**Compliance** - Audit trail for regulated industries.

**Experimentation** - Compare model trained on different data versions.

### Data Versioning Strategies

**Snapshot Versioning** - Store complete copy at each version. Simple but storage-intensive. Suitable for small datasets.

**Delta/Incremental Versioning** - Store only changes between versions. Storage-efficient. Used by Git-like systems (DVC, lakeFS).

**Hash-Based Versioning** - Content-addressable storage. Same content, same hash. Deduplicates automatically.

**Time-Based Versioning** - Query data as it existed at specific timestamp. Requires temporal tables or append-only logs.

### Data Versioning Tools

**DVC (Data Version Control)** - Git-like interface for data. Stores data remotely (S3, GCS), tracks with Git. Commands: dvc add, dvc push, dvc pull. Integrates with ML pipelines.

**lakeFS** - Git for data lakes. Provides branches, commits, merges for data. ACID guarantees. Format-agnostic (Parquet, CSV, images).

**Delta Lake** - ACID transactions on data lakes. Time travel via versioning. Handles schema evolution. Built on Parquet.

**Pachyderm** - Data versioning with pipeline automation. Provenance tracking. Containerized data transformations.

### Best Practices

**Version Datasets, Not Individual Files** - Version logical dataset (all files for training) not individual files.

**Immutable Data** - Never modify versioned data. Create new version instead.

**Semantic Versioning** - Major.Minor.Patch. Major: breaking changes, Minor: backward-compatible additions, Patch: bug fixes.

**Metadata Tracking** - Record schema, row count, statistics, data quality metrics per version.

**Automatic Snapshotting** - Version data automatically when pipeline runs. Link model to data version used for training.
