# Machine Learning System Design
### A Comprehensive Guide to Production ML, AI Architectures, and Agentic AI Systems


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

# Chapter III: Model Training at Scale

## Distributed Training

Training large models on single GPU becomes infeasible due to memory and time constraints. Distributed training splits work across multiple devices.

### Data Parallelism

**Concept** - Same model replicated across devices. Each device processes different data batch. Gradients averaged across devices, weights updated synchronously.

**Workflow:**
1. Model replicated to all devices
2. Each device gets different data batch
3. Each computes forward pass and gradients
4. Gradients aggregated (averaged) across devices
5. Each device updates weights with averaged gradients
6. Repeat for next batch

**Synchronous Data Parallel** - All devices synchronize after each batch. Consistent but slow (waits for slowest device).

**Asynchronous Data Parallel** - Devices update central parameter server independently. Faster but can lead to stale gradients and convergence issues.

**Advantages** - Easy to implement. Linear speedup with number of devices (in theory). No model changes needed.

**Limitations** - Communication overhead for gradient aggregation. Batch size scales with devices (may affect convergence). All-reduce operation bottleneck.

**Implementations** - PyTorch DistributedDataParallel (DDP), TensorFlow MirroredStrategy, Horovod.

### Model Parallelism

**Concept** - Model split across devices. Each device holds different model layers. Data passes through devices sequentially.

**Types:**

**Tensor Parallelism** - Splits individual layers across devices. Example: Large matrix multiplication split into smaller operations. Each device computes subset of neurons. Requires frequent communication between devices.

**Pipeline Parallelism** - Splits model into sequential stages. Each device holds consecutive layers. Data flows through pipeline. Micro-batching prevents pipeline bubbles (idle time).

**Workflow:**
1. Split model into stages (layers 1-10 on GPU0, 11-20 on GPU1)
2. Divide batch into micro-batches
3. GPU0 processes first micro-batch, passes activations to GPU1
4. While GPU1 processes first micro-batch, GPU0 starts second
5. Pipeline stays full, reduces bubble time

**Advantages** - Enables training models larger than single device memory. No batch size increase required.

**Limitations** - Pipeline bubbles reduce efficiency. Complex implementation. Requires careful layer partitioning.

**Implementations** - GPipe, PipeDream, Megatron-LM.

### Hybrid Parallelism

Combines data and model parallelism. Model split across devices (model parallel), each partition replicated (data parallel). Common for largest models (GPT-3, GPT-4).

Example: 8 GPUs, 4-way model parallel, 2-way data parallel. Model split into 4 parts, each replicated twice.

### ZeRO (Zero Redundancy Optimizer)

**Problem** - Optimizer states (Adam) consume 2-3x parameter memory. Each GPU stores full copy in data parallelism.

**ZeRO Stages:**

**ZeRO-1 (Optimizer State Partitioning)** - Partitions optimizer states across devices. Each device stores 1/N of optimizer states. 4x memory reduction for optimizer states.

**ZeRO-2 (Gradient Partitioning)** - Also partitions gradients. Each device stores 1/N of gradients. 8x memory reduction combined.

**ZeRO-3 (Parameter Partitioning)** - Also partitions parameters. Each device stores 1/N of parameters. Communicates parameters as needed. Near-linear memory scaling with devices.

**ZeRO-Offload** - Offloads optimizer computation and states to CPU. Trades compute for GPU memory.

**ZeRO-Infinity** - Offloads to NVMe storage. Enables trillion-parameter models.

**Implementations** - DeepSpeed ZeRO, FairScale.

---

## Hyperparameter Optimization

Hyperparameters control learning process but aren't learned from data. Examples: learning rate, batch size, number of layers, hidden units, dropout rate.

### Search Strategies

**Grid Search** - Exhaustively tries all combinations in predefined grid. Example: learning_rate = [0.001, 0.01, 0.1], batch_size = [32, 64]. Tests all 6 combinations. Simple but exponentially expensive. Not recommended for more than 3-4 hyperparameters.

**Random Search** - Samples random combinations. Often finds good solutions faster than grid search. Can explore larger search space with same budget. Recommended over grid search.

**Bayesian Optimization** - Builds probabilistic model of objective function. Uses acquisition function to select next hyperparameters. Balances exploration (uncertain regions) and exploitation (promising regions). Efficient for expensive evaluations. Tools: Optuna, Hyperopt, Ax.

**Acquisition Functions:**
- Expected Improvement (EI): Expected improvement over current best
- Probability of Improvement (PI): Probability of improving over current best
- Upper Confidence Bound (UCB): Mean + uncertainty bonus

**Population-Based Training (PBT)** - Evolves population of models simultaneously. Periodically copies hyperparameters from better to worse models. Mutates hyperparameters during training. Enables online hyperparameter adaptation.

**Successive Halving** - Allocates small budget to many configs. Eliminates bottom 50% after each round. Doubles budget for survivors. Continues until one remains.

**Hyperband** - Extension of successive halving. Runs multiple successive halving brackets with different budget allocations. Robust to unknown optimal budget.

**ASHA (Asynchronous Successive Halving)** - Asynchronous version for distributed optimization. Promotes/stops configurations asynchronously.

### Hyperparameter Types

**Continuous** - Learning rate, dropout rate, regularization strength. Search in log space for learning rate (0.0001 to 0.1).

**Integer** - Number of layers, hidden units, batch size. Often powers of 2 for efficiency.

**Categorical** - Optimizer (Adam, SGD, RMSprop), activation function (ReLU, GELU, Swish).

**Conditional** - Depend on other hyperparameters. Example: momentum only relevant if optimizer=SGD.

### Early Stopping

Monitors validation metric during training. Stops if no improvement for patience epochs. Saves best model checkpoint. Prevents overfitting and saves computation.

**Patience** - Number of epochs to wait without improvement. Too low: stops before convergence. Too high: wastes computation.

### Multi-Fidelity Optimization

Evaluates configurations with varying resource budgets. Low fidelity: Few epochs, small dataset subset. High fidelity: Full training. Quickly eliminates bad configurations with low fidelity. Invests computation in promising ones.

---

## Transfer Learning

Leverages knowledge from pre-trained model (source task) for new task (target task). Particularly effective when target task has limited data.

### Why Transfer Learning Works

**Feature Reusability** - Early layers learn general features (edges, textures for images; syntax for NLP). Later layers learn task-specific features. Source and target tasks share underlying feature representations.

**Low-Resource Settings** - Training from scratch requires large datasets. Transfer learning achieves good performance with small target datasets.

### Transfer Learning Strategies

**Feature Extraction (Frozen Features)** - Freeze all pre-trained layers. Train only new output layer on target task. Fast, requires minimal data. Use when target dataset very small (<1000 samples) and similar to source domain.

**Fine-Tuning** - Initialize with pre-trained weights. Continue training on target task. Updates all or subset of layers. More powerful but requires more target data.

**Fine-Tuning Strategies:**
- Full Fine-Tuning: Update all layers. Use when sufficient target data (>10k samples).
- Gradual Unfreezing: Start with frozen layers, gradually unfreeze from top to bottom. Prevents catastrophic forgetting.
- Discriminative Fine-Tuning: Use different learning rates per layer. Lower LR for early layers (general features), higher for late layers (task-specific).
- Chain-Thaw: Unfreeze and train one layer at a time from top to bottom.

**Layer-Wise Learning Rates** - Early layers: LR = base_lr / 100 (small updates preserve general features). Middle layers: LR = base_lr / 10. Final layers: LR = base_lr (allow significant adaptation).

### Domain Adaptation

Source and target domains differ in distribution. Techniques bridge domain gap.

**Feature-Level Adaptation** - Learns domain-invariant features. Domain Adversarial Neural Network (DANN): Adds domain classifier. Trains feature extractor to fool domain classifier. Features become domain-invariant.

**Instance Weighting** - Reweights source samples to match target distribution. Assigns higher weight to source samples similar to target.

**Self-Training** - Trains on source data. Generates pseudo-labels for target data with high confidence predictions. Retrains with source + pseudo-labeled target data. Iterates until convergence.

### Multi-Task Learning

Trains single model on multiple related tasks simultaneously. Shared layers learn common representations. Task-specific layers capture task-specific patterns.

**Hard Parameter Sharing** - All tasks share hidden layers. Each task has separate output layer. Simple, prevents overfitting through shared representation.

**Soft Parameter Sharing** - Each task has own parameters. Regularization encourages parameter similarity across tasks.

**Advantages** - Improved generalization through multi-task regularization. Data-efficient (leverages data from all tasks). Single model serves multiple predictions.

---

## Few-Shot & Zero-Shot Learning

### Few-Shot Learning

Learns from very few examples per class (1-shot, 5-shot). Critical when data labeling expensive or rare classes exist.

**Meta-Learning (Learning to Learn)** - Trains model to quickly adapt to new tasks. Model-Agnostic Meta-Learning (MAML): Finds initialization that adapts quickly with few gradient steps. Training alternates between support set (few examples for adaptation) and query set (evaluation).

**Prototypical Networks** - Learns embedding space. Computes prototype (mean embedding) per class from support set. Classifies query by nearest prototype. Simple, effective for few-shot classification.

**Siamese Networks** - Learns similarity between pairs. Trains on pairs with contrastive loss (similar pairs close, dissimilar far). At test, compares query to support examples.

**Matching Networks** - Attention mechanism over support set. Classifies query based on weighted combination of support labels. Weights from attention scores.

**Data Augmentation** - Critical for few-shot. Generates additional training examples. Task-specific augmentations (rotation, crop for images; paraphrasing for text).

### Zero-Shot Learning

Classifies without any labeled examples of target classes. Relies on auxiliary information (class descriptions, attributes, semantic embeddings).

**Semantic Embeddings** - Maps images and class names to shared embedding space. Example: Word2Vec embeddings for class names. Model trained to align image features with class name embeddings.

**Attribute-Based** - Defines classes by attributes. Example: "has stripes", "has four legs" for animals. Model predicts attributes, infers class from attribute combination.

**Generative Models** - Generates synthetic examples for unseen classes. Conditional GAN with class embeddings. Generates images for unseen classes, trains classifier.

**Vision-Language Models** - CLIP, ALIGN pre-trained on image-text pairs. Zero-shot: Provide text descriptions of classes as prompts. Model scores image-text similarity for classification.

---

## Continual Learning

Model learns from stream of tasks/data over time. Challenges: catastrophic forgetting (new task overwrites old knowledge), task boundaries may be unknown.

### Strategies

**Regularization-Based** - Penalizes changes to important parameters from previous tasks.

**Elastic Weight Consolidation (EWC)** - Computes parameter importance (Fisher Information) after each task. Adds regularization term penalizing changes to important parameters. Less important parameters free to adapt.

**Synaptic Intelligence** - Tracks parameter importance online during training. Importance based on contribution to loss reduction.

**Memory-Based (Replay)** - Stores subset of old task data. Replays during new task training. Prevents forgetting by interleaving old and new data.

**Experience Replay** - Stores raw samples from previous tasks. Randomly samples from memory during training.

**Generative Replay** - Trains generative model on previous tasks. Generates pseudo-samples instead of storing real ones. More memory-efficient.

**Parameter Isolation** - Allocates separate parameters for each task.

**Progressive Neural Networks** - Adds new column (sub-network) per task. Previous columns frozen. New column can use features from previous via lateral connections.

**PackNet** - Trains on first task. Prunes less important weights. "Packs" second task into remaining capacity. Iterates for more tasks.

**Dynamic Architectures** - Grows network as needed. Dynamically Expandable Network (DEN) adds neurons when capacity saturated. Selective retraining updates relevant parameters.

**Task-Specific Heads** - Shared feature extractor, task-specific output layers. Requires task identity at test time.

### Continual Learning Metrics

**Average Accuracy** - Mean accuracy across all tasks after learning all tasks.

**Forgetting** - Difference between peak accuracy on task and final accuracy after learning subsequent tasks. Measures catastrophic forgetting.

**Forward Transfer** - Performance on new task benefiting from previous learning. Positive: new task learns faster. Negative: previous learning hurts.

**Backward Transfer** - How learning new task affects previous task performance. Positive: new learning improves old tasks. Negative: new learning degrades old tasks.

# Chapter IV: Model Deployment

## Model Optimization

Optimizing models for production deployment focuses on reducing latency, memory footprint, and computational requirements while maintaining accuracy.

### Model Quantization

**Quantization** reduces numerical precision of weights and activations from float32 (32 bits) to lower bit representations.

**Post-Training Quantization (PTQ):**
- Converts trained float32 model to int8 without retraining
- Simple: Single function call in most frameworks
- 4x model size reduction, 2-4x speedup on CPUs
- Slight accuracy loss (typically <1%)
- Static Quantization: Calibrates on representative dataset to determine scale factors
- Dynamic Quantization: Weights quantized statically, activations quantized dynamically at runtime

**Quantization-Aware Training (QAT):**
- Simulates quantization during training using fake quantization nodes
- Model learns to be robust to reduced precision
- Better accuracy than PTQ, especially for aggressive quantization
- Adds quantization operations to forward pass
- Backward pass still in float32
- Requires retraining but achieves near-float32 accuracy

**Quantization Formats:**
- int8: 8-bit integers, 4x compression, standard for deployment
- int4: 4-bit integers, 8x compression, more accuracy loss
- Mixed Precision: Critical layers (first/last) stay float16, others int8
- Per-Channel Quantization: Different scale per output channel, better accuracy than per-tensor

**Symmetric vs Asymmetric Quantization:**
- Symmetric: Range [-α, α], simpler hardware, zero-point always zero
- Asymmetric: Range [α, β], more accurate for asymmetric distributions, requires zero-point offset

### Model Pruning

**Pruning** removes unnecessary parameters to reduce model size and computation.

**Unstructured Pruning:**
- Removes individual weights below threshold
- High sparsity possible (90%+ for CNNs)
- Requires sparse matrix operations for speedup
- Irregular sparsity pattern
- Magnitude-Based: Prunes smallest absolute value weights
- Tools: TensorFlow Model Optimization, PyTorch pruning utilities

**Structured Pruning:**
- Removes entire structures: neurons, filters, attention heads, layers
- Lower compression than unstructured but efficient on standard hardware
- Regular sparsity pattern
- Filter Pruning: Removes entire convolutional filters
- Attention Head Pruning: Removes attention heads in transformers
- Layer Dropping: Removes entire layers

**Iterative Pruning:**
1. Train model to convergence
2. Prune small percentage (e.g., 20%)
3. Fine-tune to recover accuracy
4. Repeat until target sparsity
- Gradual pruning prevents drastic accuracy loss

**Lottery Ticket Hypothesis:**
- Claims winning subnetworks exist within larger networks
- These subnetworks, when trained in isolation from initialization, match full network performance
- Finding winning tickets enables training smaller models directly

### Knowledge Distillation

**Concept** - Transfer knowledge from large "teacher" model to smaller "student" model.

**Standard Distillation:**
- Student trained on soft targets (teacher's probability distribution) not just hard labels
- Loss = α * distillation_loss + (1-α) * student_loss
- Distillation loss: KL divergence between student and teacher outputs
- Temperature parameter (T) softens probabilities: softmax(logits / T)
- Higher T → softer distributions → more information in relative probabilities

**Why It Works:**
- Dark knowledge: Teacher's incorrect class probabilities contain information
- Example: Cat image, teacher outputs [0.9 cat, 0.08 dog, 0.02 car]
- "Dog" probability (though wrong) indicates visual similarity
- Student learns these nuanced relationships

**Self-Distillation:**
- Teacher and student same architecture
- Iteratively distills from generation N to N+1
- Improves accuracy even without compression

**Multi-Teacher Distillation:**
- Multiple teachers (ensemble) distill to single student
- Student learns from diverse knowledge sources
- Outperforms single-teacher distillation

**Progressive Distillation:**
- Gradually reduces student size through multiple distillation stages
- Teacher → Medium Student → Small Student → Tiny Student
- Each stage easier than direct large-to-small distillation

### Neural Architecture Search (NAS)

**Concept** - Automatically discovers optimal neural architectures for given task and constraints.

**Search Space:**
- Defines possible architectures
- Micro-search: Cell structure, repeated throughout network
- Macro-search: Entire network topology
- Limits: Layer types (conv, pool, skip), connections, hyperparameters

**Search Strategy:**
- Random Search: Samples random architectures
- Reinforcement Learning: Controller RNN generates architectures, reward is validation accuracy
- Evolutionary Algorithm: Mutates and crosses over architectures, selects fittest
- Gradient-Based: DARTS (Differentiable Architecture Search), continuous relaxation makes search space differentiable

**Performance Estimation:**
- Full training per candidate expensive
- Early Stopping: Train few epochs, estimate final performance
- Weight Sharing: All candidate architectures share supernetwork weights
- Proxy Tasks: Smaller dataset or model for faster evaluation

**Hardware-Aware NAS:**
- Optimizes for target hardware (mobile, edge device)
- Objective includes latency, energy, memory
- Produces Pareto-optimal architectures (accuracy vs efficiency)

---

## Edge Deployment

Deploying models on resource-constrained devices (phones, IoT, embedded systems).

### Constraints

**Memory** - Limited RAM (1-4GB on phones, <1GB on IoT). Model must fit with application.

**Compute** - CPU-only or limited GPU/NPU. Lower FLOPS than servers.

**Power** - Battery-powered devices. Energy efficiency critical.

**Latency** - Real-time requirements (camera apps, AR). Network calls add latency and require connectivity.

### Optimization Techniques

**Quantization** - Essential for edge. int8 reduces size 4x, speeds up CPU inference.

**Pruning** - Removes parameters. 50-90% sparsity common without accuracy loss.

**Efficient Architectures** - Designed for mobile: MobileNet (depthwise separable convolutions), EfficientNet (compound scaling), SqueezeNet (fire modules).

**Depthwise Separable Convolution:**
- Standard convolution: K×K×C_in×C_out parameters
- Depthwise: K×K×C_in parameters (operates per channel)
- Pointwise: 1×1×C_in×C_out parameters (combines channels)
- Total: K×K×C_in + C_in×C_out vs K×K×C_in×C_out
- 8-9x fewer parameters for 3×3 kernels

**Knowledge Distillation** - Distill large server model to small edge model.

### Mobile Frameworks

**TensorFlow Lite:**
- Optimized for mobile/embedded
- Supports quantization, GPU delegate
- Pre-optimized operations for ARM processors
- Model size typically <50MB

**PyTorch Mobile:**
- Mobile runtime for PyTorch models
- Supports quantization, model optimization
- Smaller binary size than full PyTorch

**ONNX Runtime:**
- Cross-platform inference engine
- Supports multiple frameworks (PyTorch, TensorFlow)
- Hardware acceleration (CoreML on iOS, NNAPI on Android)

**Core ML (iOS):**
- Apple's ML framework
- Optimized for Apple Silicon (Neural Engine)
- Supports on-device training

**TensorFlow.js:**
- Runs models in browsers
- WebGL acceleration
- Model conversion from TensorFlow/Keras

### On-Device vs Cloud Hybrid

**Fully On-Device:**
- Advantages: No latency, privacy, works offline
- Disadvantages: Limited model complexity, deployment updates difficult

**Cloud-Only:**
- Advantages: Complex models, easy updates, unlimited compute
- Disadvantages: Latency, privacy concerns, requires connectivity

**Hybrid:**
- Simple models on-device for low-latency inference
- Complex models in cloud for high-quality results
- Fallback: Cloud model when high confidence needed
- Example: Voice assistant does wake-word detection on-device, speech recognition in cloud

### Model Updates

**Over-the-Air (OTA):**
- Download model updates via network
- Enables frequent model improvements without app updates
- Challenges: Download size, versioning, fallback on download failure

**App Store Updates:**
- Bundle model with application
- Requires app store approval and user update
- Guarantees model-app compatibility

**Delta Updates:**
- Only download changed parameters
- Reduces bandwidth for model updates
- Requires efficient diff algorithm

---

## Multi-Model Serving

Serving multiple models simultaneously in production.

### Use Cases

**A/B Testing:**
- Compare model versions
- Route portion of traffic to each variant
- Measure performance metrics
- Promote winner to full traffic

**Ensemble Serving:**
- Multiple models make prediction
- Combine predictions (averaging, voting, stacking)
- Often more accurate than single model
- Trade-off: Higher latency and cost

**Multi-Tenant:**
- Serve different models per customer/tenant
- Each tenant has customized model
- Shared infrastructure, isolated models

**Model Specialization:**
- Different models for different segments
- Example: Fraud detection model per geographic region
- Each model specialized for segment characteristics

### Serving Architectures

**Separate Services:**
- Each model in independent service
- Pros: Isolation, independent scaling
- Cons: Resource duplication, management overhead

**Multi-Model Server:**
- Single server hosts multiple models
- Models loaded/unloaded dynamically based on demand
- Pros: Resource sharing, operational simplicity
- Cons: Resource contention, harder isolation

**Model Routing:**
- Router directs requests to appropriate model
- Routing based on: Model version, customer ID, feature flags, request characteristics
- Enables canary deployments, gradual rollouts

### Model Loading Strategies

**Eager Loading:**
- Load all models at startup
- Pros: Zero cold-start latency
- Cons: High memory usage, slow startup

**Lazy Loading:**
- Load models on first request
- Pros: Lower memory baseline
- Cons: First request latency spike

**Dynamic Loading/Unloading:**
- Load frequently-used models
- Unload idle models to free memory
- LRU (Least Recently Used) eviction policy
- Monitors model access patterns

### Resource Management

**Model Batching:**
- Batch requests across multiple models
- Share GPU across models
- Trade-off: Increased latency for better throughput

**GPU Sharing:**
- Multiple models share single GPU
- CUDA Multi-Process Service (MPS)
- Time-slicing or spatial partitioning

**Priority Queues:**
- Critical models get higher priority
- Production models prioritized over experimental
- SLA-based prioritization

### Monitoring Multi-Model Systems

**Per-Model Metrics:**
- Latency, throughput, error rate per model
- Resource usage (CPU, memory, GPU) per model
- Request distribution across models

**Cross-Model Metrics:**
- Total system throughput
- Resource utilization
- Cost per prediction

**A/B Testing Metrics:**
- Statistical significance of metric differences
- Sample size per variant
- Confidence intervals

### Challenges

**Version Compatibility:**
- Models may require different library versions
- Containerization isolates dependencies
- Model server must handle version conflicts

**Cold Start:**
- Loading large models takes time (seconds to minutes)
- Warm pools of pre-loaded models
- Predict which models likely needed next

**Resource Allocation:**
- Balancing resources across models
- Elastic scaling per model based on load
- Preventing one model from starving others

**Cost Optimization:**
- Some models more expensive than others
- Route simple requests to cheaper models
- Reserve expensive models for complex cases

- # Chapter V: MLOps & Production

## CI/CD for ML

Continuous Integration and Continuous Deployment adapted for ML systems. Traditional CI/CD insufficient due to data dependencies, model artifacts, and statistical nature of ML.

### Components

**Continuous Integration (CI):**
- Code changes trigger automated pipeline
- Runs unit tests on code
- Validates data schemas
- Checks data quality
- Tests model training pipeline
- Validates model performance on validation set
- Compares metrics against baseline

**Continuous Delivery (CD):**
- Automated deployment to staging
- Integration tests in staging environment
- A/B testing configuration
- Gradual rollout (canary deployment)
- Automated rollback on metric degradation

**Continuous Training (CT):**
- Unique to ML systems
- Automatically retrains models on new data
- Triggers: Schedule (daily/weekly), data drift detection, performance degradation
- Validates retrained model before deployment

**Continuous Monitoring (CM):**
- Monitors model performance in production
- Tracks data distribution shifts
- Alerts on anomalies
- Feeds back to CT triggers

### ML Pipeline Testing

**Data Tests:**
- Schema validation: correct types, required fields present
- Distribution tests: mean, variance within expected ranges
- Freshness tests: data not stale
- Volume tests: row counts within expectations
- Referential integrity: foreign keys valid

**Model Tests:**
- Training convergence: loss decreases, no NaN/inf values
- Prediction quality: accuracy > threshold on validation set
- Inference latency: <100ms p99 (example threshold)
- Model size: <500MB for deployment
- Behavioral tests: model behaves correctly on edge cases

**Example Edge Case Tests:**
- Invariance: Prediction unchanged when adding irrelevant features
- Directional expectation: Increasing feature X should increase/decrease prediction
- Minimum functionality: Accuracy >80% on critical subset

**Integration Tests:**
- End-to-end pipeline execution
- Feature store integration
- Model registry registration
- Serving infrastructure deployment
- Monitoring dashboard updates

### Infrastructure as Code

**Model Training:**
- Training infrastructure defined in code (Terraform, CloudFormation)
- Compute resources (GPU clusters)
- Data storage (S3 buckets, databases)
- Networking (VPCs, security groups)
- Version controlled, repeatable deployments

**Model Serving:**
- Serving infrastructure as code
- Load balancers, autoscaling groups
- Container orchestration (Kubernetes manifests)
- Monitoring and logging configuration

### Versioning

**Code Versioning:**
- Git for training code, preprocessing, serving
- Tags for production releases

**Data Versioning:**
- DVC, lakeFS for training data
- Feature store tracks feature versions

**Model Versioning:**
- Model registry assigns versions
- Links model to code and data versions

**Environment Versioning:**
- Docker images with dependencies
- Requirements.txt or environment.yml pinned versions

---

## Model Monitoring

Continuous tracking of model performance and health in production.

### Metrics to Monitor

**Model Quality Metrics:**
- Classification: Accuracy, precision, recall, F1, AUC
- Regression: MAE, RMSE, R²
- Ranking: NDCG, MAP
- Requires ground truth labels (often delayed)

**Operational Metrics:**
- Prediction latency (p50, p95, p99)
- Throughput (requests per second)
- Error rate (5xx errors, timeouts)
- Resource utilization (CPU, memory, GPU)

**Business Metrics:**
- Click-through rate (CTR)
- Conversion rate
- Revenue per prediction
- User engagement

**Data Quality Metrics:**
- Missing value rate
- Out-of-vocabulary tokens (NLP)
- Invalid value percentage
- Schema violations

### Monitoring Architecture

**Logging:**
- Log all predictions with features, timestamp, model version
- Sample or aggregate for high-volume systems
- Store in data warehouse for analysis

**Metrics Collection:**
- Real-time metrics to time-series database (Prometheus, InfluxDB)
- Aggregated metrics (per minute, per hour)
- Dashboards for visualization (Grafana, Kibana)

**Alerting:**
- Define thresholds for critical metrics
- Multi-level alerts: warning, critical
- On-call rotation for critical alerts
- Alert fatigue prevention: tune thresholds, suppress flapping

### Ground Truth Collection

**Challenge:** Ground truth often delayed or unavailable in real-time.

**Delayed Labels:**
- Fraud detection: True label days/weeks later after investigation
- Churn prediction: Label known only after observation period
- Loan default: Known only after loan matures

**Strategies:**
- Human labeling: Annotators label sample of predictions
- Natural labels: User behavior indicates quality (clicks, purchases)
- Proxy metrics: Correlated metrics available immediately

**Feedback Loops:**
- Collect ground truth when available
- Join predictions with delayed labels
- Compute actual performance metrics
- Trigger retraining if degradation detected

---

## Data & Concept Drift

Model performance degrades when data distribution changes from training distribution.

### Types of Drift

**Covariate Shift (Data Drift):**
- Input feature distribution P(X) changes
- Relationship P(Y|X) unchanged
- Example: Recommendation system trained on desktop users, deployed to mobile users
- Detection: Compare feature distributions between training and production

**Prior Probability Shift:**
- Label distribution P(Y) changes
- P(X|Y) unchanged
- Example: Fraud rate increases from 1% to 5%
- Detection: Monitor label distribution

**Concept Drift:**
- Relationship P(Y|X) changes
- Input distribution may stay same
- Example: User preferences change over time, economic conditions shift
- Detection: Monitor model performance metrics

**Types of Concept Drift:**
- Sudden: Abrupt change (new product launch, policy change)
- Gradual: Slow evolution over time (seasonal trends)
- Incremental: Step-wise changes
- Recurring: Cyclic patterns (daily, weekly, seasonal)

### Drift Detection Methods

**Statistical Tests:**

**Kolmogorov-Smirnov Test:**
- Tests if two samples from same distribution
- Compares cumulative distribution functions
- Works for continuous features
- Null hypothesis: same distribution

**Chi-Square Test:**
- For categorical features
- Compares observed vs expected frequencies
- Detects changes in category proportions

**Population Stability Index (PSI):**
- Measures distribution shift
- PSI = Σ (actual% - expected%) * ln(actual% / expected%)
- PSI < 0.1: No significant change
- 0.1 < PSI < 0.2: Moderate change
- PSI > 0.2: Significant change, investigate

**Divergence Metrics:**

**KL Divergence:**
- Measures how one distribution differs from reference
- Asymmetric: KL(P||Q) ≠ KL(Q||P)
- KL(P||Q) = Σ P(x) * log(P(x) / Q(x))

**Jensen-Shannon Divergence:**
- Symmetric version of KL divergence
- Bounded between 0 and 1
- JS(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M), M = 0.5*(P+Q)

**Wasserstein Distance:**
- "Earth Mover's Distance"
- Minimum cost to transform one distribution to another
- Works well for continuous distributions

### Handling Drift

**Model Retraining:**
- Retrain on recent data
- Full retraining: Complete retraining from scratch
- Incremental learning: Update model with new data (not all algorithms support)
- Trigger: Scheduled (weekly/monthly) or drift-based

**Ensemble Methods:**
- Weighted ensemble of models from different time periods
- Recent models get higher weight
- Adapts to drift while retaining historical knowledge

**Online Learning:**
- Continuously update model as new data arrives
- Suitable for streaming scenarios
- Challenges: Catastrophic forgetting, concept drift

**Feature Engineering:**
- Add temporal features capturing time trends
- Seasonality features (day of week, month)
- Trend features (moving averages)

**Data Windowing:**
- Train on recent time window only
- Window size balance: Too small (overfits recent), too large (includes outdated patterns)
- Sliding window: Retrain as window moves forward

---

## A/B Testing for ML

Compares two model versions to determine which performs better.

### Experimental Design

**Hypothesis:**
- Null hypothesis (H0): New model = Old model
- Alternative hypothesis (H1): New model > Old model
- Define success metric (e.g., conversion rate increase)

**Randomization:**
- Randomly assign users/requests to control (A) or treatment (B)
- Ensures groups comparable
- Avoids selection bias

**Sample Size:**
- Determine required sample size for statistical power
- Factors: Expected effect size, significance level (α, typically 0.05), power (1-β, typically 0.8)
- Online calculators or: n = (Z_α/2 + Z_β)² * (p1*(1-p1) + p2*(1-p2)) / (p1-p2)²

**Duration:**
- Run long enough to account for weekly patterns
- Minimum 1 week, often 2-4 weeks
- Account for user overlap (same user sees both variants)

### Implementation

**Traffic Splitting:**
- User-level: Consistent experience per user
- Request-level: Each request randomly assigned
- Session-level: Consistent within session
- Hash-based assignment: hash(user_id) % 100 < 50 → Group A

**Gradual Rollout:**
- Start small (1-5% treatment)
- Increase if metrics favorable
- Stops at any sign of degradation
- Reduces blast radius of poor models

**Holdout Group:**
- Keep small percentage (1-5%) on old model permanently
- Long-term comparison
- Detects delayed effects

### Analysis

**Statistical Significance:**
- T-test or Z-test for metric differences
- P-value < 0.05: Reject null hypothesis (models differ)
- Confidence intervals show range of true effect

**Multiple Testing Correction:**
- Testing multiple metrics increases false positive rate
- Bonferroni correction: α_adjusted = α / number_of_tests
- False Discovery Rate (FDR) control

**Practical Significance:**
- Statistical significance ≠ practical significance
- Effect size: Is improvement large enough to matter?
- Cost-benefit: Does improvement justify deployment cost?

**Segment Analysis:**
- Performance may differ across user segments
- Analyze by demographics, device type, geography
- Heterogeneous treatment effects

### Common Pitfalls

**Peeking:**
- Checking results before planned end date
- Increases false positive rate
- Solution: Pre-commit to sample size and duration

**Novelty Effect:**
- Users initially react to change itself, not underlying quality
- Effect diminishes over time
- Solution: Run test longer (2-4 weeks)

**Seasonality:**
- Metrics vary by day of week, time of year
- Solution: Run complete weeks, compare same day-of-week

**Network Effects:**
- Treatment affects control group (social platforms)
- Spillover violates independence assumption
- Solution: Cluster randomization (assign entire communities)

---

## Shadow & Canary Deployment

Deployment strategies that reduce risk of introducing new models.

### Shadow Deployment

**Concept:**
- Deploy new model alongside existing production model
- New model receives all traffic but predictions not served to users
- Compare new model predictions to old model and ground truth
- Zero user impact

**Workflow:**
1. Deploy new model in shadow mode
2. All requests sent to both models
3. Old model predictions served to users
4. New model predictions logged for analysis
5. Compare performance metrics
6. If satisfactory, promote new model to production

**Benefits:**
- Safe validation in production environment
- Real production traffic and data
- No user impact during evaluation
- Can run indefinitely for comprehensive comparison

**Costs:**
- Doubles infrastructure cost (running two models)
- Doubles latency (both models must complete)
- May need asynchronous shadow calls to avoid latency impact

**Use Cases:**
- High-risk deployments (finance, healthcare)
- Major model architecture changes
- Validating improved model before full rollout

### Canary Deployment

**Concept:**
- Deploy new model to small percentage of traffic (1-10%)
- Monitor closely for issues
- Gradually increase traffic if metrics acceptable
- Roll back immediately if problems detected

**Workflow:**
1. Deploy new model to 5% of traffic
2. Monitor metrics for 24-48 hours
3. If metrics acceptable, increase to 25%
4. Continue gradual increase: 25% → 50% → 100%
5. Roll back at any sign of degradation

**Canary Metrics:**
- Error rate: Should not increase
- Latency: Should remain within SLA
- Business metrics: Should not degrade
- User complaints: Monitor support tickets

**Benefits:**
- Limits blast radius of bad deployments
- Real user feedback at small scale
- Quick rollback if issues detected
- Gradual confidence building

**Costs:**
- Requires traffic splitting infrastructure
- Longer deployment cycle
- Complex monitoring and automation

**Automated Canary:**
- Automated metric comparison
- Automatic promotion if metrics pass thresholds
- Automatic rollback if metrics fail
- Reduces manual oversight

### Blue-Green Deployment

**Concept:**
- Two identical environments: Blue (current) and Green (new)
- Deploy new model to Green environment
- Switch traffic from Blue to Green atomically
- Keep Blue as instant rollback option

**Workflow:**
1. Blue environment serves production traffic
2. Deploy new model to Green environment
3. Test Green environment with smoke tests
4. Switch router/load balancer to Green
5. Monitor for issues
6. If problems, switch back to Blue instantly
7. If stable, Blue becomes next deployment target

**Benefits:**
- Zero-downtime deployment
- Instant rollback capability
- Clear separation of old and new versions

**Costs:**
- Requires double infrastructure
- Coordination of traffic switching
- State synchronization between environments (for stateful systems)

### Comparison

**Shadow Deployment:**
- Risk: Zero (no user impact)
- Cost: High (runs two models)
- Duration: Can run indefinitely
- Use: Major changes, risk-averse scenarios

**Canary Deployment:**
- Risk: Low (limited traffic)
- Cost: Medium (gradual infrastructure increase)
- Duration: Days to weeks
- Use: Standard deployments, gradual validation

**Blue-Green Deployment:**
- Risk: Medium (all traffic switches at once)
- Cost: High (double infrastructure at switchover)
- Duration: Minutes to hours
- Use: Fast rollback requirement, clear version separation

- # Chapter VI: Deep Learning Systems

## Neural Architecture Design

### Core Building Blocks

**Fully Connected (Dense) Layers:**
- Each neuron connects to all neurons in previous layer
- Parameters: weights (input_dim × output_dim) + biases (output_dim)
- Use: Final layers for classification, small input dimensions
- Limitations: Massive parameters for high-dimensional inputs

**Convolutional Layers:**
- Applies filters across spatial dimensions
- Parameters: filter_size × filter_size × input_channels × output_channels
- Weight sharing: Same filter applied across entire input
- Translation invariance: Detects features regardless of position
- Use: Images, sequences, any grid-like data

**Pooling Layers:**
- Downsamples spatial dimensions
- Max Pooling: Takes maximum value in window
- Average Pooling: Takes average value in window
- Reduces parameters, provides translation invariance
- No learnable parameters

**Recurrent Layers (RNN, LSTM, GRU):**
- Processes sequential data with hidden state
- LSTM: Long Short-Term Memory, handles long-range dependencies via gates (forget, input, output)
- GRU: Gated Recurrent Unit, simpler than LSTM with fewer parameters (reset, update gates)
- Use: Time-series, sequences, but largely replaced by Transformers

**Attention Mechanisms:**
- Weighs importance of different input parts
- Self-Attention: Attends to different positions of same sequence
- Cross-Attention: Attends from one sequence to another
- Scaled Dot-Product Attention: Q, K, V matrices, Attention(Q,K,V) = softmax(QK^T / √d_k)V

**Normalization Layers:**

**Batch Normalization:**
- Normalizes across batch dimension
- Reduces internal covariate shift
- Acts as regularizer
- Parameters: learnable scale (γ) and shift (β)
- Challenge: Behavior differs between training (uses batch statistics) and inference (uses running statistics)

**Layer Normalization:**
- Normalizes across feature dimension
- Independent of batch size
- Used in Transformers
- Stable across batch sizes

**Group Normalization:**
- Divides channels into groups, normalizes within groups
- Middle ground between Layer Norm and Instance Norm
- Effective for small batches

**Activation Functions:**
- ReLU: max(0, x), simple, fast, but dead neurons problem
- Leaky ReLU: max(αx, x), prevents dead neurons
- GELU: x * Φ(x), smooth, used in Transformers
- Swish/SiLU: x * sigmoid(x), smooth, self-gated
- Tanh: (-1, 1) range, used in RNNs
- Sigmoid: (0, 1) range, used in output layers for probabilities

**Residual Connections (Skip Connections):**
- Adds input directly to output: y = F(x) + x
- Enables training very deep networks (ResNet)
- Gradient flows directly through skip connections
- Mitigates vanishing gradient problem

---

## Computer Vision Systems

### Image Classification

**Architecture Evolution:**

**LeNet (1998):** Conv → Pool → Conv → Pool → FC → FC. Early CNN, simple structure.

**AlexNet (2012):** Deeper, ReLU activation, dropout, data augmentation. Won ImageNet 2012.

**VGG (2014):** Very deep (16-19 layers), small 3×3 filters throughout. Simple, modular design.

**GoogLeNet/Inception (2014):** Inception modules with parallel convolutions of different sizes (1×1, 3×3, 5×5). Efficient parameter usage.

**ResNet (2015):** Residual connections enable training 50-152 layer networks. Solves vanishing gradient. Identity mappings preserve gradients.

**EfficientNet (2019):** Compound scaling (depth, width, resolution). Optimizes accuracy-efficiency trade-off. Neural Architecture Search.

**Vision Transformers (ViT, 2020):** Applies Transformers to images. Splits image into patches, treats as sequence. Competitive with CNNs, especially with large datasets.

### Object Detection

**Goal:** Locate and classify multiple objects in image.

**Two-Stage Detectors:**

**R-CNN:** Region proposal (selective search) → CNN feature extraction → SVM classification. Slow, separate stages.

**Fast R-CNN:** Single CNN, RoI pooling, end-to-end training. Faster than R-CNN.

**Faster R-CNN:** Replaces selective search with Region Proposal Network (RPN). RPN predicts objectness and bounding boxes. Fully learnable.

**One-Stage Detectors:**

**YOLO (You Only Look Once):** Divides image into grid. Each cell predicts bounding boxes and class probabilities. Single forward pass. Real-time inference.

**SSD (Single Shot Detector):** Multiple feature maps at different scales. Predicts boxes from each scale. Handles different object sizes.

**RetinaNet:** Focal Loss addresses class imbalance (many background regions). Feature Pyramid Network for multi-scale detection.

**Components:**

**Anchor Boxes:** Pre-defined boxes of various sizes/aspect ratios. Detector predicts offsets from anchors.

**Non-Maximum Suppression (NMS):** Removes duplicate detections. Suppresses boxes with high IoU (Intersection over Union) with higher-confidence boxes.

**IoU (Intersection over Union):** Overlap metric. IoU = Area of Overlap / Area of Union. Measures prediction accuracy.

### Semantic Segmentation

**Goal:** Classify each pixel into categories.

**FCN (Fully Convolutional Network):** Removes fully connected layers. Uses transpose convolutions for upsampling. Produces pixel-wise predictions.

**U-Net:** Encoder-decoder with skip connections. Encoder downsamples, decoder upsamples. Skip connections preserve spatial information. Popular for medical imaging.

**DeepLab:** Atrous (Dilated) convolutions for larger receptive field without losing resolution. Atrous Spatial Pyramid Pooling (ASPP) captures multi-scale context.

**PSPNet:** Pyramid Pooling Module aggregates context at multiple scales. Global context helps disambiguation.

### Instance Segmentation

**Goal:** Detect and segment each object instance individually.

**Mask R-CNN:** Extends Faster R-CNN. Adds mask prediction branch parallel to bounding box branch. RoIAlign for precise spatial localization.

---

## NLP Systems

### Text Representation

**One-Hot Encoding:** Binary vector, one dimension per vocabulary word. Sparse, no semantic similarity.

**Word Embeddings:**

**Word2Vec:** CBOW (predict word from context) or Skip-gram (predict context from word). Dense vectors, captures semantic similarity. Similar words have similar vectors.

**GloVe:** Global Vectors. Matrix factorization on word co-occurrence counts. Combines global statistics with local context.

**FastText:** Extends Word2Vec with subword information (character n-grams). Handles out-of-vocabulary words. Effective for morphologically rich languages.

**Contextualized Embeddings:**

**ELMo:** Bidirectional LSTM. Produces context-dependent embeddings. Same word different embeddings in different contexts.

**BERT:** Bidirectional Transformer. Masked Language Modeling (predicts masked words) and Next Sentence Prediction. Pre-trained on large corpus, fine-tuned for tasks.

**GPT:** Unidirectional Transformer. Autoregressive language modeling (predicts next word). Pre-trained, fine-tuned.

### Text Classification

**Approaches:**
- Bag-of-Words + Classical ML (Naive Bayes, Logistic Regression, SVM)
- CNN for text: 1D convolutions over word embeddings, captures local patterns (n-grams)
- RNN/LSTM: Processes sequences, captures long-range dependencies
- Transformers (BERT): State-of-art, bidirectional context, fine-tune pre-trained model

**Architecture:**
Input text → Embedding → Encoder (CNN/LSTM/Transformer) → Pooling (max/average) → Dense → Output

### Named Entity Recognition (NER)

**Goal:** Identify entities (person, organization, location) in text.

**Sequence Labeling:**
- BIO tagging: B-PER (Begin Person), I-PER (Inside Person), O (Outside)
- Each token assigned tag
- Model: BiLSTM-CRF or Transformer

**BiLSTM-CRF:**
- BiLSTM encodes context from both directions
- CRF (Conditional Random Field) layer enforces tag sequence constraints (e.g., I-PER must follow B-PER)
- Jointly optimizes entire tag sequence

**Transformer-based:**
- BERT + token classification head
- Fine-tune on NER dataset
- State-of-art performance

### Machine Translation

**Seq2Seq with Attention:**
- Encoder processes source sentence into context
- Decoder generates target sentence
- Attention weighs relevant source positions for each target position
- Handles variable-length sequences

**Transformer for Translation:**
- Encoder-decoder architecture
- Multi-head self-attention in encoder and decoder
- Cross-attention from decoder to encoder
- Parallel processing (vs sequential in RNN)
- State-of-art: T5, mBART, NLLB

**Evaluation Metrics:**
- BLEU: Compares n-gram overlap with reference translations
- METEOR: Considers synonyms and stemming
- chrF: Character-level F-score

---

## Recommendation Systems

### Approaches

**Collaborative Filtering:**

**User-Based:** Find similar users, recommend items they liked. Similarity: Cosine, Pearson correlation.

**Item-Based:** Find similar items, recommend similar items to what user liked. More stable than user-based (items change less).

**Matrix Factorization:** Decompose user-item matrix into user and item latent factors. Users and items represented as vectors in latent space. Prediction: dot product of user and item vectors. ALS (Alternating Least Squares), SVD.

**Content-Based Filtering:**
- Recommends items similar to those user liked
- Uses item features (genre, keywords, attributes)
- TF-IDF for text, embeddings for images
- Doesn't require other users' data

**Hybrid Systems:**
- Combines collaborative and content-based
- Netflix: Collaborative filtering + content features
- Overcomes cold-start for new items

### Deep Learning for Recommendations

**Neural Collaborative Filtering (NCF):**
- Replaces matrix factorization dot product with neural network
- Learns non-linear user-item interactions
- Embeddings for users and items fed to MLP

**Two-Tower Models:**
- Separate towers (networks) for users and items
- User tower: user features → embedding
- Item tower: item features → embedding
- Similarity: dot product or cosine of embeddings
- Efficient candidate generation (item embeddings pre-computed)

**Sequence-Based:**
- Models user interaction sequences
- RNN/LSTM/Transformer over user's past interactions
- Predicts next item
- Captures temporal dynamics

**Wide & Deep:**
- Wide component: Memorizes specific feature interactions (linear model)
- Deep component: Generalizes with neural network
- Combines memorization and generalization
- Used at Google Play

### Ranking

**Two-Stage Approach:**

**Candidate Generation:**
- Retrieves hundreds to thousands of candidates from millions of items
- Fast, approximate methods (collaborative filtering, nearest neighbors, trending)
- Recall-focused

**Ranking:**
- Scores and ranks candidates
- Complex model considering many features
- Precision-focused
- Outputs top-N items

**Features:**
- User features: demographics, history, preferences
- Item features: category, popularity, recency
- Context features: time, device, location
- Cross features: user-item interactions

**Learning to Rank:**
- Pointwise: Predicts relevance score per item
- Pairwise: Learns which item preferred over another
- Listwise: Optimizes ranking of entire list

---

## Time Series Systems

### Time Series Characteristics

**Trend:** Long-term increase or decrease.

**Seasonality:** Repeating patterns at fixed intervals (daily, weekly, yearly).

**Cyclicality:** Fluctuations not at fixed periods.

**Noise:** Random variations.

### Classical Methods

**ARIMA (AutoRegressive Integrated Moving Average):**
- AR: Regression on past values
- I: Differencing to make stationary
- MA: Regression on past forecast errors
- Requires stationarity (constant mean, variance)

**Seasonal ARIMA (SARIMA):**
- Extends ARIMA with seasonal components
- Handles seasonal patterns

**Exponential Smoothing:**
- Weighted average of past observations
- Recent observations weighted more
- Simple, Holt (trend), Holt-Winters (trend + seasonality)

### Deep Learning for Time Series

**RNN/LSTM/GRU:**
- Processes sequences naturally
- Maintains hidden state capturing history
- Many-to-one: Sequence → single prediction (classification)
- Many-to-many: Sequence → sequence (forecasting multiple steps)

**Temporal Convolutional Networks (TCN):**
- 1D convolutions over time
- Dilated convolutions for large receptive fields
- Causal convolutions (no future information)
- Parallelizable (unlike RNN)

**Transformers for Time Series:**
- Attention over time steps
- Captures long-range dependencies
- Informer, Autoformer for long sequences

**N-BEATS (Neural Basis Expansion Analysis for Time Series):**
- Stack of blocks with residual connections
- Each block predicts basis functions
- Forecasts aggregated from blocks
- Interpretable (trend and seasonality blocks)

**DeepAR:**
- Autoregressive RNN for probabilistic forecasting
- Learns across multiple related time series
- Outputs distribution, not point estimate
- Handles uncertainty quantification

### Features for Time Series

**Lag Features:** Values from previous time steps (t-1, t-7, t-30).

**Rolling Statistics:** Moving average, moving std, min, max over windows.

**Time-Based Features:** Hour, day_of_week, month, is_weekend, is_holiday.

**Fourier Features:** Captures periodic patterns, sin/cos transformations.

### Evaluation Metrics

**MAE (Mean Absolute Error):** Average absolute difference.

**RMSE (Root Mean Squared Error):** Square root of average squared difference. Penalizes large errors more.

**MAPE (Mean Absolute Percentage Error):** Percentage error, scale-independent.

**SMAPE (Symmetric MAPE):** Symmetric version, handles zero values better.

### Challenges

**Non-Stationarity:** Mean/variance changes over time. Solution: Differencing, transformations.

**Missing Data:** Gaps in time series. Solution: Interpolation, forward fill, model-based imputation.

**Multiple Seasonalities:** Daily + weekly + yearly patterns. Solution: Seasonal decomposition, Fourier features.

**Long-Range Dependencies:** Predictions depend on distant past. Solution: Transformers, dilated convolutions.

**Irregularly Sampled:** Observations at non-uniform intervals. Solution: Continuous-time models, interpolation.

# Chapter VII: Large Language Models

## LLM Architecture

Large Language Models are autoregressive transformers trained to predict next token given previous tokens. Scale (parameters, data, compute) drives performance improvements.

### Key Characteristics

**Autoregressive:** Generates tokens sequentially, each conditioned on previous tokens.

**Causal Masking:** Attention only to previous positions, not future. Maintains autoregressive property.

**Large Scale:** Billions to trillions of parameters. GPT-3: 175B, GPT-4: estimated >1T, Llama 2: 7B-70B.

**Pre-training Objective:** Next token prediction on massive text corpora (web pages, books, code).

**Emergent Abilities:** Capabilities appearing only at scale (few-shot learning, reasoning, instruction following).

---

## Transformer Deep Dive

### Architecture Components

**Token Embedding:**
- Maps tokens to dense vectors
- Embedding matrix: vocab_size × d_model
- Learned during training

**Positional Encoding:**
- Injects position information (transformers have no inherent order)
- Sinusoidal: PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
- Learned: Trainable position embeddings
- RoPE (Rotary Positional Embedding): Rotates embeddings based on position, used in Llama, captures relative positions better

**Multi-Head Attention:**

Single attention head:
1. Query (Q), Key (K), Value (V) projections from input: Q = XW_Q, K = XW_K, V = XW_V
2. Attention scores: Attention(Q, K, V) = softmax(QK^T / √d_k)V
3. √d_k scaling prevents large dot products (which cause gradients to vanish after softmax)

Multi-head splits into h heads:
- Each head: different Q, K, V projections, learns different relationships
- Concatenate head outputs, project: MultiHead(Q,K,V) = Concat(head_1, ..., head_h)W_O
- Typical: 12-96 heads

**Feed-Forward Network (FFN):**
- Applied after attention
- Two linear layers with activation: FFN(x) = ReLU(xW_1 + b_1)W_2 + b_2 (ReLU often replaced with GELU/SiLU)
- Dimension expansion: d_model → 4*d_model → d_model
- Processes each position independently

**Layer Normalization:**
- Applied before attention and FFN (Pre-LN) or after (Post-LN)
- Pre-LN more stable for deep networks
- Normalizes across feature dimension

**Residual Connections:**
- Around attention and FFN blocks
- Output = LayerNorm(x + Attention(x))
- Enables deep networks, gradient flow

**Complete Transformer Block:**
```
x = x + MultiHeadAttention(LayerNorm(x))
x = x + FFN(LayerNorm(x))
```

### Attention Mechanisms

**Self-Attention:** Attends to different positions within same sequence. Captures dependencies between words.

**Causal Self-Attention (Decoder):** Masked attention, position i only attends to positions ≤ i. Prevents looking at future tokens.

**Cross-Attention (Encoder-Decoder):** Decoder attends to encoder outputs. Used in translation, not in GPT-style LLMs.

**Attention Patterns:**
- Local patterns: Adjacent token attention (syntax)
- Long-range: Distant token attention (coreference, themes)
- Broadcast: All positions attend to specific token (e.g., [CLS], special tokens)

### Computational Complexity

**Attention Complexity:** O(n² * d), where n = sequence length, d = model dimension.

**Problem:** Quadratic in sequence length. 1000 tokens → 1M attention computations. 10k tokens → 100M.

**Solutions for Long Contexts:**

**Sparse Attention:** Attend to subset of positions. Patterns: local window, strided, global tokens.

**Linear Attention:** Approximates attention in O(n) complexity. Linearizes softmax operation.

**Flash Attention:** Algorithm optimization, not architecture change. Reduces memory I/O, enables longer contexts with same memory.

---

## Pre-training & Fine-tuning

### Pre-training

**Objective:** Maximize likelihood of next token: P(x_t | x_1, ..., x_{t-1})

**Dataset:** Trillions of tokens from diverse sources (web, books, code, academic papers). Curated for quality, filtered for toxicity.

**Scale:**
- Data: GPT-3 trained on ~500B tokens
- Compute: Months on thousands of GPUs/TPUs
- Cost: Millions to tens of millions of dollars

**Pre-training Benefits:**
- Learns language structure, world knowledge, reasoning patterns
- Transfers to downstream tasks
- Foundation for specialization

### Fine-tuning

**Supervised Fine-Tuning (SFT):**
- Continues training on task-specific data
- Input-output pairs (e.g., instruction-response)
- Small dataset (thousands to millions of examples)
- Lower learning rate than pre-training

**Instruction Tuning:**
- Fine-tunes on instructions and desired responses
- Teaches model to follow instructions
- Example datasets: FLAN, Alpaca, Dolly

**Domain Adaptation:**
- Fine-tunes on domain-specific data (medical, legal, code)
- Specializes language and knowledge

**Multi-Task Fine-Tuning:**
- Fine-tunes on multiple tasks simultaneously
- Improves generalization
- Example: T5 trained on diverse tasks with text-to-text format

### RLHF (Reinforcement Learning from Human Feedback)

**Problem:** Next token prediction doesn't align with human preferences (helpfulness, harmlessness, honesty).

**Process:**

**Step 1: Collect Comparisons**
- Humans rank multiple model responses
- "Response A better than Response B"
- Thousands to hundreds of thousands of comparisons

**Step 2: Train Reward Model**
- Model predicts human preference score
- Trained on comparison data
- Inputs: prompt + response, Output: scalar reward

**Step 3: RL Fine-Tuning (PPO)**
- Optimize policy (LLM) to maximize reward model scores
- PPO (Proximal Policy Optimization): Stable RL algorithm
- KL penalty: Keeps policy close to SFT model, prevents over-optimization

**Outcome:** Model responses aligned with human preferences. Used in ChatGPT, Claude, and other assistants.

### DPO (Direct Preference Optimization)

Alternative to RLHF. Directly optimizes on preference data without separate reward model.

**Advantages:**
- Simpler: No reward model, no RL
- Stable: Avoids RL instabilities
- Efficient: Single-stage training

**Process:** Trains LLM to increase probability of preferred responses, decrease probability of dis-preferred.

---

## LoRA & PEFT

**Problem:** Fine-tuning entire LLM expensive. Billions of parameters to update and store.

### LoRA (Low-Rank Adaptation)

**Concept:** Freeze pre-trained weights, inject trainable low-rank matrices.

**Method:**
- Original weight W ∈ R^(d×k)
- LoRA adds: ΔW = BA, where B ∈ R^(d×r), A ∈ R^(r×k), r << min(d,k)
- Forward pass: h = Wx + BAx = Wx + ΔWx
- Only train B and A (few parameters)

**Example:** d=4096, k=4096, W has 16M parameters. r=8: B has 32K, A has 32K. Total LoRA: 64K parameters (0.4% of original).

**Benefits:**
- Drastically fewer trainable parameters (10-100x reduction)
- Faster training, less memory
- Multiple LoRA adapters can be trained for different tasks, swapped at inference
- Original model remains frozen, preserving general capabilities

**Inference:** Can merge LoRA weights into original: W_merged = W + BA. No inference overhead.

### QLoRA (Quantized LoRA)

Combines LoRA with quantization.

**Approach:**
- Quantize frozen base model to 4-bit
- Train LoRA adapters in float16/bfloat16
- Drastically reduces memory requirements

**Enables:** Fine-tuning 65B model on single GPU (48GB VRAM).

### Other PEFT Methods

**Prefix Tuning:** Prepends learnable prefix tokens to each layer. Prefix guides model behavior. Frozen model, trainable prefixes.

**Prompt Tuning:** Learns soft prompt (continuous embeddings) prepended to input. Similar to prefix tuning but only input layer.

**Adapter Layers:** Inserts small bottleneck modules between transformer layers. Original layers frozen, adapters trained.

**IA³ (Infused Adapter by Inhibiting and Amplifying):** Learns vectors that scale activations. Very few parameters (10K-100K for billion-parameter models).

---

## Prompt Engineering

Crafting inputs to elicit desired model behavior.

### Basic Techniques

**Zero-Shot:** Direct instruction without examples.
Example: "Translate to French: Hello world"

**Few-Shot:** Provide examples in prompt.
Example: 
```
Translate to French:
Hello world → Bonjour le monde
Good morning → Bonjour
How are you → 
```

**Chain-of-Thought (CoT):** Prompt model to show reasoning steps.
Example: "Let's think step by step: ..."

**Self-Consistency:** Generate multiple reasoning paths, select most common answer. Improves accuracy on reasoning tasks.

### Advanced Techniques

**Instruction Following:**
Clear, specific instructions: "You are a helpful assistant. Answer the question concisely using only information from the context."

**Role Prompting:** "You are an expert Python programmer. Review this code for bugs."

**Format Specification:** "Respond in JSON format: {answer: string, confidence: float}"

**Negative Prompting:** "Do not include personal opinions or speculate."

**Prompt Decomposition:** Break complex task into subtasks. Chain results through multiple prompts.

**ReAct (Reason + Act):** Interleaves reasoning and action. "Thought: I need to search for X. Action: search(X). Observation: ..."

**Self-Critique:** Model generates response, then critiques and refines it.

### Prompt Optimization

**Manual Iteration:** Try variations, select best performing.

**Automatic Prompt Engineering:** Algorithms search prompt space. Gradient-based (AutoPrompt), Reinforcement Learning.

**Prompt Templates:** Reusable structures. Variables filled at runtime.

---

## LLM Inference Optimization

Inference for LLMs is expensive: Large models, long sequences, autoregressive generation.

### Key Challenges

**Memory Bandwidth Bound:** Loading weights from DRAM to compute units is bottleneck. Not compute-bound (GPU can compute faster than memory delivers).

**Sequential Generation:** Must generate one token at a time. Each token requires full model forward pass. 100 tokens = 100 forward passes.

**KV-Cache Growth:** Key and value tensors grow with sequence length. Memory consumption increases with longer contexts.

### Optimization Techniques

**KV-Cache:**
- Cache computed keys and values for past tokens
- Avoids recomputing attention for previous positions
- Memory: 2 * num_layers * batch_size * seq_len * d_model
- Essential for reasonable generation speed

**Quantization:**
- int8/int4 weights: 4-8x size reduction, faster memory transfer
- Activation quantization: Further speedup
- Minimal accuracy loss with careful calibration

**Model Pruning:**
- Remove less important parameters
- Structured pruning (attention heads, FFN neurons)
- Can achieve 30-50% sparsity with small accuracy drop

**Speculative Decoding:**
- Small draft model generates multiple tokens quickly
- Large model verifies in parallel
- Accepts correct tokens, rejects and regenerates wrong ones
- Speedup: 2-3x with similar quality

**Continuous Batching:**
- Traditional batching: Wait for entire batch to finish (longest sequence)
- Continuous: As sequences finish, add new ones to batch
- Higher throughput, better GPU utilization

**Paged Attention (vLLM):**
- Manages KV-cache like virtual memory paging
- Non-contiguous memory allocation
- Reduces memory fragmentation
- Higher batch size, better throughput

**Tensor Parallelism:**
- Split single model across GPUs
- Each GPU computes portion of each layer
- Enables models larger than single GPU memory
- Reduces per-GPU memory, increases bandwidth utilization

**Pipeline Parallelism:**
- Different layers on different GPUs
- Microbatching to keep pipeline full
- Balancing: Earlier layers faster, later layers slower

**Flash Attention:**
- Optimized attention algorithm
- Reduces memory I/O between GPU memory hierarchies
- Enables longer contexts with same memory
- 2-4x speedup for attention computation

**Quantized KV-Cache:**
- Stores cached KV in int8 instead of float16
- 2x memory reduction for cache
- Enables longer contexts or larger batch sizes
- Minimal quality degradation

### Serving Optimizations

**Model Compilation:**
- TensorRT, TorchScript, ONNX Runtime
- Fuses operations, optimizes compute graphs
- Hardware-specific optimizations

**Batching Strategies:**
- Dynamic batching: Accumulate requests up to timeout
- Padding: Pad sequences to same length for efficient batching
- Trade-off: Latency vs throughput

**Caching:**
- Semantic caching: Cache responses for similar prompts
- Prefix caching: Reuse computation for common prompt prefixes
- KV-cache reuse across requests (when applicable)

**Request Scheduling:**
- Priority queues: Critical requests first
- SLO-aware: Ensure latency SLAs met
- Fair sharing: Prevent starvation

### Latency Analysis

For generating N tokens:
- Model loading: One-time, seconds to minutes
- Prefill (processing prompt): Single forward pass, depends on prompt length
- Decode (generating tokens): N forward passes, each depends on model size and KV-cache size
- Time-to-first-token (TTFT): Prefill latency
- Inter-token latency: Time between tokens during decode

**Example (70B model, int8, A100 GPU):**
- TTFT: 1-2 seconds (1000 token prompt)
- Inter-token latency: 30-50ms
- Total for 100 tokens: ~5 seconds

- # Chapter VIII: Retrieval-Augmented Generation (RAG)

## RAG Architecture

RAG combines retrieval systems with generative LLMs to ground responses in external knowledge.

### Core Concept

**Problem with Pure LLMs:**
- Knowledge frozen at training time
- Cannot access real-time information
- Hallucinate facts not in training data
- Cannot cite sources

**RAG Solution:**
1. User query received
2. Retrieve relevant documents from knowledge base
3. Augment query with retrieved context
4. LLM generates response based on context
5. Response grounded in retrieved facts, often with citations

### Basic RAG Workflow

```
Query: "What was Q3 revenue for Acme Corp?"
    ↓
Embed query → vector: [0.23, -0.41, ...]
    ↓
Search vector database
    ↓
Retrieved docs: [Doc1, Doc2, Doc3] (most similar)
    ↓
Construct prompt:
"Answer based on context:
Context: [Doc1 content] [Doc2 content]
Question: What was Q3 revenue?
Answer:"
    ↓
LLM generates: "Based on the financial report, Q3 revenue was $45M..."
```

### RAG vs Fine-Tuning

**RAG:**
- Dynamic knowledge: Updated by adding/removing documents
- Source attribution: Can cite specific documents
- Lower compute: No model retraining
- Use: When knowledge changes frequently, citations needed, or data too large to fit in model

**Fine-Tuning:**
- Internalized knowledge: Learned patterns and behaviors
- No runtime retrieval overhead
- Better for: Learning style, format, domain reasoning patterns
- Use: When behavior/style adaptation needed

**Combination:** Fine-tune for domain reasoning, RAG for factual grounding.

---

## Vector Databases

Storage and retrieval of high-dimensional embeddings for similarity search.

### Core Operations

**Insert:** Store embedding vector with metadata and original content.

**Query:** Given query vector, find k most similar vectors.

**Similarity Metrics:**
- Cosine Similarity: cos(θ) = A·B / (||A|| ||B||), range [-1, 1], higher = more similar
- Euclidean Distance: L2 norm, lower = more similar
- Dot Product: A·B, higher = more similar (assumes normalized vectors)

### Indexing Algorithms

**Brute Force (Flat Index):**
- Computes similarity to every vector
- Exact results, O(n) complexity
- Works for small datasets (<100K vectors)

**Approximate Nearest Neighbors (ANN):**
- Trade exactness for speed
- Sub-linear query time
- Essential for large-scale (millions to billions of vectors)

**HNSW (Hierarchical Navigable Small World):**
- Builds multi-layer graph
- Search starts at top layer, navigates down
- Excellent recall-speed tradeoff
- Memory-intensive (stores graph)

**IVF (Inverted File Index):**
- Clusters vectors (k-means)
- Stores vectors in cluster buckets
- Search only relevant clusters
- Tunable: More clusters = faster but may miss results

**Product Quantization (PQ):**
- Compresses vectors for smaller memory footprint
- Splits vector into subvectors, quantizes each
- 8-32x compression
- Slight accuracy loss

**SPANN (Sparse Partitioned ANNS):**
- Learned partitioning of vector space
- Efficient for billion-scale datasets
- Used in large-scale production systems

### Vector Database Systems

**Pinecone:** Managed service, HNSW-based, auto-scaling, metadata filtering.

**Weaviate:** Open-source, HNSW, supports hybrid search (vector + keyword), GraphQL API.

**Milvus:** Open-source, multiple index types (IVF, HNSW, ANNOY), distributed architecture.

**Qdrant:** Open-source, Rust-based, HNSW, filtering with payloads.

**Faiss:** Facebook library, not database. Highly optimized for similarity search, GPU support, many index types.

**Chroma:** Open-source, lightweight, Python-first, easy integration with LangChain.

**pgvector:** PostgreSQL extension, SQL queries with vector similarity, good for existing Postgres users.

### Metadata Filtering

Combine vector similarity with metadata constraints.

Example: Find similar documents published after 2023 in category "AI".

**Pre-filtering:** Filter by metadata first, then vector search on subset. Accurate but may reduce candidate pool.

**Post-filtering:** Vector search first, then filter results by metadata. May miss relevant results filtered out.

**Hybrid:** Index supports combined vector+metadata queries. Best accuracy and performance.

---

## Embedding Models

Convert text to dense vectors capturing semantic meaning.

### Model Types

**Sentence Transformers:**
- Based on BERT/RoBERTa
- Fine-tuned with Siamese networks on sentence pairs
- Models: all-MiniLM-L6-v2 (384 dim, fast), all-mpnet-base-v2 (768 dim, accurate)

**OpenAI Embeddings:**
- text-embedding-3-small: 1536 dimensions, cost-effective
- text-embedding-3-large: 3072 dimensions, higher quality
- Proprietary, API-based

**Instructor:**
- Task-specific instructions prepended to text
- Single model, multiple tasks
- Example: "Represent the Science passage:" vs "Represent the Medical query:"

**E5 (Text Embeddings by Weakly-Supervised Contrastive Pre-training):**
- Multilingual
- Multiple sizes: small, base, large
- Strong performance on retrieval benchmarks

**BGE (BAAI General Embedding):**
- Strong Chinese and English performance
- Multiple sizes
- Instruction-aware

### Training Objectives

**Contrastive Learning:**
- Positive pairs (similar) pulled together
- Negative pairs (dissimilar) pushed apart
- Loss: Maximize similarity of positives, minimize similarity of negatives

**Triplet Loss:**
- Anchor, positive, negative
- Loss: d(anchor, positive) + margin < d(anchor, negative)
- Ensures positive closer than negative by margin

**Multiple Negatives Ranking Loss:**
- Batch of (query, positive document) pairs
- Treats other positives in batch as negatives
- Efficient: Leverages batch for hard negatives

**Distillation:**
- Student model learns from teacher embeddings
- Enables smaller, faster models with retained quality

### Embedding Dimensions

**Trade-offs:**
- Higher dimensions: More expressive, better quality, larger storage, slower retrieval
- Lower dimensions: Faster, less storage, potential quality loss

**Matryoshka Embeddings:**
- Single model produces embeddings of multiple dimensions
- Can truncate from 1024 to 512 or 256 dimensions
- Enables dimension-quality trade-off at inference

### Domain Adaptation

**Problem:** General embeddings may not work well for specialized domains (legal, medical, code).

**Solutions:**
- Fine-tune on domain data
- Continue pre-training on domain corpus
- Use domain-specific embedding models

**Domain-Specific Models:**
- CodeBERT: Code embeddings
- BioBERT: Biomedical text
- Legal-BERT: Legal documents

---

## Chunking Strategies

Splitting documents into chunks for embedding and retrieval.

### Why Chunking

**Embedding Window Limits:** Models have max input length (512-8192 tokens). Long documents exceed limit.

**Retrieval Precision:** Smaller chunks = more precise retrieval. Retrieving entire book vs relevant paragraph.

**Context Window:** LLM context windows limited. Need to fit multiple retrieved chunks.

### Chunking Methods

**Fixed-Size Chunking:**
- Split by character count or token count
- Simple, consistent sizes
- May split mid-sentence or mid-paragraph
- Chunks: 200-500 tokens typical

**Sentence-Based:**
- Split on sentence boundaries
- Preserves sentence integrity
- Variable chunk sizes
- Combine sentences until reaching target size

**Paragraph-Based:**
- Split on paragraph boundaries (\n\n)
- Preserves logical units
- Works well for structured documents
- May have very large or very small chunks

**Recursive Chunking:**
- Try splitting by paragraphs
- If chunk too large, split by sentences
- If still too large, split by fixed size
- Adapts to content structure

**Document Structure:**
- Use document structure (headings, sections)
- Each section becomes chunk
- Preserves semantic boundaries
- Requires structured documents (Markdown, HTML)

**Semantic Chunking:**
- Use embeddings to find semantic breaks
- Splits where topic shifts
- More expensive (requires embeddings)
- Better coherence

### Overlap

**Sliding Window:**
- Overlap between consecutive chunks
- Example: 500 tokens, 100 token overlap
- Prevents information loss at boundaries
- Increases storage (more chunks)

**Typical Overlap:** 10-20% of chunk size.

### Chunk Size Selection

**Factors:**
- Query length: Longer queries → larger chunks
- Precision needs: Higher precision → smaller chunks
- LLM context: Larger context → can use bigger chunks
- Domain: Technical docs → smaller (precise), narratives → larger

**Common Sizes:**
- Short: 128-256 tokens (high precision, Q&A)
- Medium: 256-512 tokens (balanced)
- Long: 512-1024 tokens (more context, summarization)

### Metadata Enrichment

Attach metadata to chunks:
- Document title, author, date
- Chunk position (e.g., chunk 3 of 10)
- Section/heading
- Source file path

Enables filtering and better context.

---

## Advanced RAG Patterns

### Hierarchical Retrieval

**Two-Stage:**
1. Retrieve broader chunks (paragraphs, sections)
2. Re-rank or refine to sentences
3. Send top sentences to LLM

**Document + Chunk:**
1. Embed entire document (summary embedding)
2. Also embed individual chunks
3. First retrieve relevant documents
4. Then retrieve chunks within those documents

**Benefits:** Reduces search space, faster, maintains document context.

### Query Transformation

**Query Expansion:**
- Generate multiple variations of query
- Example: "What is RAG?" → ["What is Retrieval Augmented Generation?", "How does RAG work?", "RAG architecture"]
- Retrieve for all variations, combine results
- Improves recall (finds more relevant docs)

**Query Decomposition:**
- Break complex query into sub-queries
- Example: "Compare revenue of Apple and Microsoft in 2023" → ["Apple revenue 2023", "Microsoft revenue 2023"]
- Retrieve for each, synthesize results

**Hypothetical Document Embeddings (HyDE):**
- LLM generates hypothetical answer to query
- Embed hypothetical answer
- Search with answer embedding (not query embedding)
- Finds documents similar to expected answer

### Reranking

**Problem:** Initial retrieval (cosine similarity) may not perfectly rank relevance.

**Solution:** Two-stage: Fast retrieval (top 100), then rerank with more expensive model (top 10).

**Cross-Encoder Reranking:**
- Model takes query + document as input
- Outputs relevance score
- More accurate than bi-encoder (embedding models) but slower
- Models: ms-marco-MiniLM, rerank-multilingual

**LLM-based Reranking:**
- Use LLM to score relevance
- Prompt: "Rate relevance of document to query on 1-10 scale"
- Very accurate but expensive

### Hybrid Search

Combines vector search with keyword (BM25) search.

**BM25:** Classic keyword matching. Scores based on term frequency and document frequency.

**Combination:**
- Run both vector and keyword search
- Combine scores: final_score = α * vector_score + (1-α) * bm25_score
- Typical α = 0.5 to 0.7

**Benefits:** Vector captures semantics, BM25 captures exact keywords. Complementary strengths.

### Contextual Compression

**Problem:** Retrieved chunks may contain irrelevant information.

**Solution:**
1. Retrieve chunks
2. LLM extracts only relevant parts for query
3. Compressed context to generation LLM
4. Reduces token usage, improves focus

### Multi-Query Retrieval

Generate multiple queries from original, retrieve for each, deduplicate and rank results. Improves coverage.

### Parent Document Retrieval

Embed and retrieve small chunks but return larger parent document to LLM. Balance: Small chunks (precise retrieval) + large context (full information).

---

## RAG Evaluation

### Retrieval Metrics

**Precision@k:** Of k retrieved docs, how many are relevant?

**Recall@k:** Of all relevant docs, how many are in top k?

**Mean Reciprocal Rank (MRR):** Average of reciprocal ranks of first relevant result. MRR = (1/N) Σ (1 / rank_first_relevant).

**NDCG@k (Normalized Discounted Cumulative Gain):** Considers relevance scores and position. Higher-ranked docs weighted more. Handles graded relevance (not just binary).

### Generation Metrics

**Faithfulness:** Generated response grounded in retrieved context. No hallucinations.
- Evaluation: Check if all claims in response supported by context
- Methods: NLI models, LLM-as-judge

**Relevance:** Response addresses the query.
- Evaluation: Human ratings, LLM-as-judge

**Groundedness:** Similar to faithfulness. Response uses information from context.

**Answer Correctness:** For Q&A, compare to ground truth answer.
- Exact match, F1 score, BLEU (text similarity)

**Citation Accuracy:** If system provides citations, verify correctness.
- Do citations point to actual supporting evidence?

### End-to-End Evaluation

**Context Relevance:** Are retrieved docs relevant to query?

**Context Utilization:** Does LLM use retrieved context effectively?

**Answer Quality:** Overall quality of final response.

### Evaluation Frameworks

**RAGAS (Retrieval Augmented Generation Assessment):**
- Automated metrics for RAG
- Faithfulness, answer relevance, context relevance
- Uses LLM to evaluate

**TruLens:** Observability and evaluation for RAG systems. Tracks retrieval, context, generation.

**Human Evaluation:**
- Gold standard
- Expensive, slow
- Used for benchmarking, spot-checking

### Common RAG Failures

**Retrieval Failures:**
- Poor embedding quality
- Query-document mismatch
- Too large/small chunks
- Insufficient context in chunks

**Generation Failures:**
- Ignoring context (LLM relies on parametric knowledge)
- Hallucinating despite context
- Over-relying on context (ignoring LLM's reasoning)
- Poor instruction following

**Fixes:**
- Better embeddings (domain-specific, fine-tuned)
- Improved chunking strategy
- Query transformation
- Reranking
- Prompt engineering (emphasize context usage)
- Fine-tune LLM for better context utilization

- # Chapter IX: Agentic AI

## AI Agent Fundamentals

AI Agent is a system that autonomously pursues goals through perception, reasoning, and action in an environment.

### Core Components

**Perception:** Observes environment (text inputs, API responses, tool outputs, user feedback).

**Reasoning:** Decides what to do next (planning, decision-making, problem decomposition).

**Action:** Executes actions (tool calls, API requests, code execution, text generation).

**Memory:** Maintains state (conversation history, intermediate results, learned facts).

**Environment:** External world agent interacts with (APIs, databases, web, filesystem, user).

### Agent vs Non-Agent LLM

**Non-Agent (Standard LLM):**
- Single-turn: Query → Response
- No tools or external actions
- Stateless (beyond conversation)

**Agent:**
- Multi-turn: Iteratively reasons and acts
- Uses tools to gather information or take actions
- Plans multi-step solutions
- Adapts based on feedback

---

## ReAct Pattern

ReAct (Reasoning + Acting) interleaves reasoning traces with actions.

### ReAct Loop

1. **Thought:** Agent reasons about current state, decides next action
2. **Action:** Executes action (tool call, query)
3. **Observation:** Receives result of action
4. Repeat until task complete or max steps reached

### Example

```
User: What's the weather in the capital of France?

Thought: I need to find the capital of France, then get weather for that city.
Action: search("capital of France")
Observation: The capital of France is Paris.

Thought: Now I know the capital is Paris. I need to get weather for Paris.
Action: get_weather("Paris")
Observation: Weather in Paris: 18°C, partly cloudy.

Thought: I have the weather information.
Final Answer: The weather in Paris (capital of France) is 18°C and partly cloudy.
```

### Prompting for ReAct

Provide structure in prompt:
```
You have access to these tools:
- search(query): Search the web
- calculate(expr): Evaluate mathematical expression

Use this format:
Thought: [your reasoning]
Action: [tool_name(arguments)]
Observation: [result will appear here]
... (repeat)
Thought: I now know the final answer
Final Answer: [answer to user]
```

### Benefits

**Interpretability:** Explicit reasoning traces show agent's thought process.

**Error Recovery:** Agent can observe failure and try alternative approach.

**Grounding:** Actions and observations keep agent grounded in facts.

---

## Tool Use & Function Calling

Agent's ability to use external tools expands capabilities beyond language generation.

### Function Calling

LLM decides when and how to call functions based on conversation.

**Workflow:**
1. Define available functions (name, description, parameters)
2. User query provided
3. LLM outputs function call (JSON): `{function: "get_weather", arguments: {city: "Paris"}}`
4. Application executes function, gets result
5. Result returned to LLM
6. LLM incorporates result into response

**Example Definition:**
```
{
  "name": "get_weather",
  "description": "Get current weather for a city",
  "parameters": {
    "city": {"type": "string", "description": "City name"}
  }
}
```

### Tool Categories

**Information Retrieval:**
- Web search (Google, Bing APIs)
- Database queries (SQL, GraphQL)
- Document retrieval (RAG systems)
- API calls (news, stocks, weather)

**Computation:**
- Calculator (evaluate expressions)
- Code interpreter (execute Python, JavaScript)
- Data analysis (pandas operations)

**Action Taking:**
- Send email, message
- Create calendar events
- File operations (read, write, delete)
- Control smart home devices

**Specialized Domain:**
- Medical databases
- Legal document search
- Scientific paper retrieval
- Financial data APIs

### Implementing Tool Use

**OpenAI Function Calling:**
- Native function calling in API
- Provide function definitions
- Model outputs structured function calls

**LangChain Tools:**
- Framework for defining and using tools
- Pre-built tools (search, math, SQL)
- Custom tool creation

**Semantic Kernel:**
- Microsoft's framework
- Skills as reusable functions
- Orchestration of multiple functions

**AutoGPT Style:**
- Agent autonomously selects and chains tools
- Pursues goals with minimal human intervention

### Challenges

**Hallucinated Tool Calls:** Agent invents tools that don't exist or incorrect parameters.

**Inefficient Tool Use:** Agent calls tools unnecessarily or in suboptimal order.

**Error Handling:** Tool failures must be communicated to agent for recovery.

**Security:** Agent must not execute dangerous operations. Sandboxing required.

---

## Planning & Reasoning

Agent's ability to decompose complex tasks into subtasks and execute them strategically.

### Planning Approaches

**Task Decomposition:**
- Break complex goal into smaller subgoals
- Solve subgoals sequentially or in parallel
- Combine results

**Example:**
```
Goal: Write a comprehensive report on climate change

Decomposition:
1. Research current climate data
2. Analyze historical trends
3. Review scientific consensus
4. Identify key impacts
5. Synthesize findings into report
```

**Hierarchical Planning:**
- High-level plan: Abstract steps
- Refine each step into detailed actions
- Tree structure: Plans expand into subplans

**Reactive Planning:**
- No explicit plan
- Decide next action based on current state
- Adapts dynamically to outcomes

**Chain-of-Thought Prompting:**
- Prompt LLM to think step-by-step
- Example: "Let's solve this step by step:"
- Improves reasoning on complex problems

**Tree of Thoughts:**
- Explores multiple reasoning paths (branches)
- Evaluates each path
- Selects best path or combines insights
- More comprehensive than linear chain-of-thought

**Self-Refine:**
- Agent generates solution
- Critiques own solution
- Refines iteratively
- Improves quality through self-feedback

### Reasoning Strategies

**Forward Chaining:**
- Start with known facts
- Apply rules to derive new facts
- Continue until goal reached

**Backward Chaining:**
- Start with goal
- Work backward to find supporting facts
- "To achieve X, I need Y. To achieve Y, I need Z..."

**Analogical Reasoning:**
- Identifies similar problems from memory
- Adapts previous solutions to current problem

**Abductive Reasoning:**
- Inference to best explanation
- Given observations, what's most likely cause?

---

## Memory Systems

Agent's ability to store and recall information across interactions.

### Memory Types

**Short-Term Memory (Working Memory):**
- Current conversation context
- Temporary state within episode
- Stored in LLM context window
- Limited by context length

**Long-Term Memory:**
- Persistent across conversations
- Facts learned over time
- Past interactions and outcomes
- Stored in external databases

**Episodic Memory:**
- Specific past experiences
- "I helped this user debug Python code last week"
- Enables personalization and continuity

**Semantic Memory:**
- General knowledge and facts
- "Python uses indentation for blocks"
- Not tied to specific episodes

**Procedural Memory:**
- Learned skills and procedures
- "How to debug code: check error message, isolate problem, test fix"

### Memory Architectures

**Buffer Memory:**
- Stores recent k messages
- Simple, fixed size
- Loses older information

**Summary Memory:**
- Periodically summarizes conversation
- Stores summary + recent messages
- Compresses long histories

**Vector Store Memory:**
- Embeds memories in vector database
- Retrieves relevant memories via similarity search
- Scalable to large memory banks

**Entity Memory:**
- Tracks entities (people, places, concepts)
- Updates entity properties over time
- Example: User preferences, learned facts

**Knowledge Graph Memory:**
- Stores memories as graph
- Nodes: Entities, Edges: Relationships
- Enables complex queries

### Memory Operations

**Store:** Add new information to memory. Decide what's worth remembering.

**Retrieve:** Fetch relevant memories. Similarity-based or query-based.

**Update:** Modify existing memories when new information arrives.

**Forget:** Remove outdated or irrelevant memories. Manage memory size.

### Implementing Memory

**Conversation History:**
- Store in LLM context (short-term)
- Summarize and store externally (long-term)

**Vector Database:**
- Embed memories
- Retrieve via similarity to current context

**SQL Database:**
- Structured memory (entities, attributes)
- Query by attributes

**Hybrid:**
- Combine vector (semantic) and structured (factual) memory

---

## Multi-Agent Systems

Multiple agents collaborate to solve complex tasks.

### Architectures

**Centralized:**
- Central controller coordinates agents
- Controller assigns tasks, aggregates results
- Simple coordination but single point of failure

**Decentralized:**
- Agents communicate peer-to-peer
- Negotiate and coordinate autonomously
- Robust but complex coordination

**Hierarchical:**
- Manager agent delegates to worker agents
- Workers report back to manager
- Scales better than flat structures

### Agent Roles

**Specialist Agents:**
- Each agent specialized for specific domain
- Example: Research agent, writing agent, coding agent
- Collaborate on multidisciplinary tasks

**Debate/Discussion:**
- Multiple agents with different perspectives
- Debate to reach consensus or generate ideas
- Improves solution quality through diverse viewpoints

**Reviewer/Critic:**
- One agent generates, another reviews
- Iterative refinement
- Example: Coder agent + reviewer agent

**Ensemble:**
- Multiple agents solve same problem independently
- Aggregate solutions (voting, averaging)
- Reduces individual agent errors

### Communication

**Shared Memory:**
- Agents read/write to shared workspace
- Simple but requires synchronization

**Message Passing:**
- Agents send messages to each other
- Explicit communication, clear interface

**Blackboard Architecture:**
- Shared knowledge base (blackboard)
- Agents contribute knowledge
- Opportunistic problem-solving

### Coordination

**Task Allocation:**
- Divide tasks among agents
- Based on capabilities, load, efficiency

**Conflict Resolution:**
- Agents may propose conflicting actions
- Resolution: Priority, voting, arbiter agent

**Synchronization:**
- Agents must coordinate timing
- Example: Data must be ready before analysis agent starts

### Examples

**AutoGPT:** Single agent with tool use, autonomous goal pursuit.

**BabyAGI:** Task creation and prioritization, iterative task execution.

**MetaGPT:** Multi-agent software company. Agents play roles (PM, architect, engineer, tester).

**CAMEL:** Communicative Agents for exploring Large-scale Language Model communication.

---

## Agent Evaluation

Measuring agent performance presents unique challenges compared to static models.

### Metrics

**Task Success Rate:**
- Percentage of tasks completed successfully
- Binary: Success or failure
- Threshold-based: Partial credit for near-success

**Efficiency:**
- Number of steps to complete task
- Computational cost (tokens used, API calls)
- Time to completion

**Generalization:**
- Performance on unseen tasks
- Transfer to new environments/tools

**Safety:**
- Harmful actions avoided
- Staying within boundaries
- Not executing dangerous operations

**Interpretability:**
- Clarity of reasoning traces
- Auditability of decisions

### Evaluation Environments

**Simulated Environments:**
- Controlled, repeatable
- Example: WebArena (web navigation), ALFWorld (interactive fiction)
- Allows large-scale testing

**Real-World Environments:**
- Actual APIs, databases, systems
- More realistic but harder to control
- Risk of unintended actions

**Benchmark Suites:**
- HumanEval (code generation)
- GAIA (general AI assistant tasks)
- AgentBench (multi-domain agent tasks)

### Challenges

**Non-Determinism:**
- LLMs and environments often non-deterministic
- Multiple runs needed for robust evaluation

**Long Horizons:**
- Tasks may require many steps
- Failure at any step cascades
- Credit assignment: Which step caused failure?

**Partial Observability:**
- Agent doesn't see full environment state
- Must infer from observations

**Evaluation Cost:**
- Each run may require many LLM calls
- Expensive at scale

### Safety Evaluation

**Red Teaming:**
- Adversarial testing
- Attempt to make agent fail or misbehave

**Sandboxing:**
- Run agent in isolated environment
- Limit potential damage

**Human-in-the-Loop:**
- Human approval for critical actions
- Safety guardrails

**Monitoring:**
- Log all agent actions
- Detect and halt dangerous behavior

### Iterative Improvement

**Error Analysis:**
- Categorize failure modes
- Prioritize most common issues

**Prompt Engineering:**
- Refine instructions based on failures
- Add examples, constraints

**Tool Improvement:**
- Better tool descriptions
- More robust error handling

**Memory Optimization:**
- Improve retrieval relevance
- Better memory management
