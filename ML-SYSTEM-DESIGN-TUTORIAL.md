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
