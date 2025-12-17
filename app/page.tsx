'use client'

import { motion, AnimatePresence } from 'framer-motion'
import { useState, useEffect } from 'react'
import TechStackSection from './components/TechStackSection'
import Navigation from './components/Navigation'

export default function Home() {
  const [showScrollTop, setShowScrollTop] = useState(false)

  useEffect(() => {
    const handleScroll = () => {
      setShowScrollTop(window.scrollY > 300)
    }
    window.addEventListener('scroll', handleScroll)
    return () => window.removeEventListener('scroll', handleScroll)
  }, [])

  const scrollToTop = () => {
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }

  const exportTechSkills = () => {
    // Format tech stacks data for export
    const formatTechStacks = (stacks: typeof techStacks) => {
      let output = 'KAVIN N RANGANATHAN - TECH SKILLS\n'
      output += '='.repeat(50) + '\n\n'
      output += `Generated on: ${new Date().toLocaleDateString('en-US', { 
        year: 'numeric', 
        month: 'long', 
        day: 'numeric' 
      })}\n\n`

      Object.values(stacks).forEach((section) => {
        output += `\n${section.title.toUpperCase()}\n`
        output += '-'.repeat(50) + '\n'

        if ('items' in section && section.items) {
          section.items.forEach((item: string) => {
            output += `  • ${item}\n`
          })
        }

        if ('categories' in section && section.categories) {
          section.categories.forEach((category: any) => {
            output += `\n  ${category.name}:\n`
            
            if (category.items && Array.isArray(category.items)) {
              category.items.forEach((item: string) => {
                output += `    - ${item}\n`
              })
            }

            if (category.subcategories && Array.isArray(category.subcategories)) {
              category.subcategories.forEach((subcat: any) => {
                output += `\n    ${subcat.name}:\n`
                if (subcat.items && Array.isArray(subcat.items)) {
                  subcat.items.forEach((item: string) => {
                    output += `      - ${item}\n`
                  })
                }
              })
            }
          })
        }
        output += '\n'
      })

      return output
    }

    const formattedData = formatTechStacks(techStacks)
    
    // Create and download file
    const blob = new Blob([formattedData], { type: 'text/plain' })
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = `Kavin_N_Ranganathan_Tech_Skills_${new Date().toISOString().split('T')[0]}.txt`
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
    URL.revokeObjectURL(url)
  }

  const techStacks = {
    programmingLanguages: {
      title: 'Programming Languages',
      categories: [
        {
          name: 'Backend',
          items: ['Python', 'Golang', 'PHP', 'Rust (Basics)']
        },
        {
          name: 'Frontend',
          items: ['JavaScript/TypeScript', 'HTML5', 'CSS3']
        },
        {
          name: 'Scripting',
          items: ['Bash', 'PowerShell']
        }
      ]
    },
    frontendDevelopment: {
      title: 'Frontend Development',
      categories: [
        {
          name: 'Frameworks',
          items: ['React.js', 'Next.js', 'Vue.js', 'Angular', 'Svelte']
        },
        {
          name: 'UI Libraries',
          items: ['Material-UI', 'Ant Design', 'Tailwind CSS', 'Bootstrap', 'Chakra UI']
        },
        {
          name: 'State Management',
          items: ['Redux', 'Zustand', 'Recoil', 'MobX']
        },
        {
          name: 'Build Tools',
          items: ['Webpack', 'Vite', 'Parcel', 'esbuild']
        }
      ]
    },
    backendDevelopment: {
      title: 'Backend Development',
      categories: [
        {
          name: 'Frameworks',
          items: ['FastAPI', 'Flask', 'Django', 'Express.js', 'NestJS', 'Laravel']
        },
        {
          name: 'API',
          items: ['REST', 'GraphQL', 'gRPC', 'WebSockets', 'tRPC']
        },
        {
          name: 'Testing',
          items: ['Pytest', 'Jest', 'Mocha', 'Selenium', 'Cypress', 'Playwright']
        }
      ]
    },
    versionControl: {
      title: 'Version Control & Collaboration',
      items: ['Github', 'Bitbucket', 'DVC', 'Git LFS']
    },
    databases: {
      title: 'Databases & Storage',
      categories: [
        {
          name: 'Relational',
          items: ['PostgreSQL', 'MySQL', 'Azure SQL Database', 'Cloud SQL']
        },
        {
          name: 'NoSQL',
          items: ['MongoDB', 'DynamoDB']
        },
        {
          name: 'In-Memory',
          items: ['Redis']
        },
        {
          name: 'Time-Series',
          items: ['InfluxDB']
        },
        {
          name: 'Graph',
          items: ['Neo4j', 'Amazon Neptune']
        },
        {
          name: 'Data Warehouses',
          items: ['BigQuery', 'Redshift', 'ClickHouse']
        },
        {
          name: 'Vector Databases',
          items: ['Pinecone', 'Weaviate', 'Milvus', 'Qdrant', 'ChromaDB', 'FAISS', 'Pgvector']
        }
      ]
    },
    visualization: {
      title: 'Visualization & BI Tools',
      items: ['Power BI', 'Looker', 'Grafana', 'Plotly', 'Apache Superset', 'Metabase', 'Redash', 'Apache Echarts', 'D3.js', 'Dash']
    },
    mlops: {
      title: 'MLOps',
      categories: [
        {
          name: 'Experiment Tracking',
          items: ['MLflow', 'Weights & Biases']
        },
        {
          name: 'Model Registry',
          items: ['MLflow Model Registry', 'AWS SageMaker Model Registry', 'Azure ML Model Registry']
        },
        {
          name: 'Feature Stores',
          items: ['Feast', 'AWS SageMaker Feature Store']
        },
        {
          name: 'Model Versioning',
          items: ['DVC', 'Git LFS']
        },
        {
          name: 'Model Deployment',
          items: ['TensorFlow Serving', 'TorchServe', 'NVIDIA Triton', 'BentoML', 'Seldon Core', 'KServe']
        },
        {
          name: 'Model Monitoring',
          items: ['Evidently AI']
        },
        {
          name: 'Data Quality',
          items: ['Great Expectations']
        },
        {
          name: 'Pipeline Orchestration',
          items: ['Apache Airflow', 'Kubeflow Pipelines']
        }
      ]
    },
    aiops: {
      title: 'AIOps',
      categories: [
        {
          name: 'Monitoring & Observability',
          items: ['Prometheus', 'Grafana', 'LangSmith']
        },
        {
          name: 'APM',
          items: ['Dynatrace']
        },
        {
          name: 'Log Management',
          items: ['ELK Stack (Elasticsearch, Logstash, Kibana)', 'Loki', 'Fluentd']
        },
        {
          name: 'Distributed Tracing',
          items: ['AWS X-Ray']
        },
        {
          name: 'Chaos Engineering',
          items: ['Chaos Monkey', 'Gremlin']
        },
        {
          name: 'Auto-Remediation',
          items: ['StackStorm', 'Rundeck', 'Ansible Tower']
        }
      ]
    },
    cicd: {
      title: 'CI/CD & DevOps',
      categories: [
        {
          name: 'CI/CD',
          items: ['Jenkins', 'GitHub Actions', 'Azure DevOps']
        },
        {
          name: 'Container & Orchestration',
          items: ['Docker', 'Kubernetes']
        }
      ]
    },
    openSource: {
      title: 'Open Source Tech Stacks',
      categories: [
        {
          name: 'Data Orchestration',
          items: ['Apache Airflow', 'Apache Nifi', 'Prefect', 'Dagster']
        },
        {
          name: 'Stream Processing',
          items: ['Apache Kafka', 'Apache Flink', 'Apache Pulsar', 'Apache Storm', 'Kafka Streams']
        },
        {
          name: 'Data Warehousing',
          items: ['Apache Druid', 'Apache Pinot', 'ClickHouse']
        },
        {
          name: 'Visualization',
          items: ['Apache Superset', 'Metabase', 'Redash']
        },
        {
          name: 'Data Processing',
          items: ['Apache Spark', 'Apache Beam', 'Dask', 'Ray']
        },
        {
          name: 'Message Queues',
          items: ['RabbitMQ', 'Apache ActiveMQ', 'NATS']
        }
      ]
    },
    agenticAI: {
      title: 'Agentic AI & LLM Frameworks',
      items: ['N8N', 'LangChain', 'LangGraph', 'CrewAI', 'Haystack', 'LlamaIndex', 'AutoGen', 'Semantic Kernel', 'DSPy', 'OpenAI Assistants API', 'Anthropic Claude API', 'Flowise', 'Dify', 'AgentGPT']
    },
    aiML: {
      title: 'AI/ML Frameworks & Libraries',
      categories: [
        {
          name: 'Deep Learning',
          items: ['PyTorch', 'TensorFlow', 'JAX', 'Keras', 'Hugging Face Transformers', 'ONNX']
        },
        {
          name: 'Classical ML',
          items: ['Scikit-learn', 'XGBoost', 'LightGBM', 'CatBoost', 'H2O.ai']
        },
        {
          name: 'NLP',
          items: ['spaCy', 'NLTK', 'Sentence Transformers', 'Tokenizers', 'Gensim']
        },
        {
          name: 'Computer Vision',
          items: ['OpenCV', 'YOLO', 'MMDetection', 'Detectron2', 'Albumentations']
        },
        {
          name: 'Reinforcement Learning',
          items: ['Stable Baselines3', 'Ray RLlib', 'OpenAI Gym']
        },
        {
          name: 'AutoML',
          items: ['AutoKeras', 'Auto-sklearn', 'TPOT', 'H2O AutoML', 'PyCaret']
        },
        {
          name: 'Model Optimization',
          items: ['ONNX Runtime', 'TensorRT', 'OpenVINO', 'Quantization', 'Pruning']
        }
      ]
    },
    cloudPlatforms: {
      title: 'Cloud Platforms',
      categories: [
        {
          name: 'AWS',
          subcategories: [
            { name: 'Compute', items: ['EC2', 'Lambda', 'ECS', 'EKS', 'Fargate', 'Batch'] },
            { name: 'Storage', items: ['S3', 'EBS', 'EFS', 'FSx', 'Glacier'] },
            { name: 'Database', items: ['RDS', 'DynamoDB', 'Aurora', 'DocumentDB', 'Neptune', 'Redshift'] },
            { name: 'AI/ML', items: ['SageMaker', 'Bedrock', 'Textract', 'Translate', 'Lex'] },
            { name: 'Data Engineering', items: ['Glue', 'Data Pipeline'] },
            { name: 'Networking', items: ['VPC', 'CloudFront', 'Route 53', 'API Gateway', 'ELB', 'Direct Connect'] },
            { name: 'Monitoring', items: ['CloudWatch'] },
            { name: 'Security', items: ['IAM', 'Secrets Manager'] }
          ]
        },
        {
          name: 'Azure',
          subcategories: [
            { name: 'Compute', items: ['Virtual Machines', 'Container Instances', 'AKS', 'App Services', 'Functions'] },
            { name: 'Storage', items: ['Blob Storage', 'Data Lake Storage', 'File Storage', 'Queue Storage'] },
            { name: 'Database', items: ['Azure SQL Database'] },
            { name: 'AI/ML', items: ['Azure ML', 'Cognitive Services', 'OpenAI Service', 'Bot Service', 'Form Recognizer'] },
            { name: 'Data Engineering', items: ['Data Factory', 'Databricks (Basics)'] },
            { name: 'Networking', items: ['Virtual Network', 'Application Gateway', 'Load Balancer', 'Front Door', 'VPN Gateway'] },
            { name: 'DevOps', items: ['Azure DevOps'] },
            { name: 'Monitoring', items: ['Azure Monitor', 'Application Insights', 'Log Analytics'] },
            { name: 'Identity', items: ['Entra ID', 'B2C', 'Managed Identities'] }
          ]
        },
        {
          name: 'GCP',
          subcategories: [
            { name: 'Compute', items: ['Compute Engine', 'App Engine', 'Cloud Run', 'GKE', 'Cloud Functions'] },
            { name: 'Storage', items: ['Cloud Storage'] },
            { name: 'Database', items: ['Cloud SQL'] },
            { name: 'AI/ML', items: ['Vertex AI', 'AutoML', 'AI Platform', 'Vision AI', 'Natural Language AI', 'Speech-to-Text'] },
            { name: 'Monitoring', items: ['Cloud Monitoring'] },
            { name: 'Security', items: ['Identity and Access Management', 'Secret Manager'] }
          ]
        }
      ]
    },
    apiIntegration: {
      title: 'API & Integration',
      categories: [
        {
          name: 'Protocols',
          items: ['REST APIs', 'GraphQL', 'gRPC', 'WebSockets', 'Server-Sent Events (SSE)']
        },
        {
          name: 'Frameworks',
          items: ['FastAPI', 'Flask', 'Express.js', 'NestJS', 'Spring Boot']
        },
        {
          name: 'API Gateway',
          items: ['Kong', 'Apigee', 'AWS API Gateway', 'Azure API Management', 'Google Cloud Endpoints']
        },
        {
          name: 'API Testing',
          items: ['Postman', 'Swagger/OpenAPI']
        }
      ]
    },
    security: {
      title: 'Security & Compliance',
      categories: [
        {
          name: 'Authentication',
          items: ['OAuth 2.0', 'JWT', 'SAML', 'OpenID Connect', 'Auth0', 'Okta']
        },
        {
          name: 'Secrets Management',
          items: ['HashiCorp Vault', 'AWS Secrets Manager', 'Azure Key Vault', 'GCP Secret Manager']
        },
        {
          name: 'Security Scanning',
          items: ['Snyk', 'SonarQube', 'Checkmarx', 'Veracode', 'Aqua Security', 'Trivy']
        },
        {
          name: 'Compliance',
          items: ['SOC 2', 'GDPR', 'HIPAA', 'PCI-DSS', 'ISO 27001']
        },
        {
          name: 'Network Security',
          items: ['SSL/TLS', 'mTLS', 'Zero Trust', 'VPN', 'Firewall Rules']
        },
        {
          name: 'OWASP',
          items: ['Top 10', 'Security Best Practices']
        }
      ]
    },
    architecture: {
      title: 'Architecture & Design Patterns',
      categories: [
        {
          name: 'Architecture Patterns',
          items: ['Microservices', 'Event-Driven', 'CQRS', 'Serverless', 'Hexagonal', 'Clean Architecture']
        },
        {
          name: 'AI Patterns',
          items: ['RAG (Retrieval Augmented Generation)', 'Fine-tuning', 'Prompt Engineering', 'Multi-agent Systems']
        },
        {
          name: 'Design Patterns',
          items: ['Singleton', 'Factory', 'Observer', 'Strategy', 'Repository', 'Circuit Breaker']
        },
        {
          name: 'API Design',
          items: ['RESTful principles', 'API versioning', 'Rate limiting', 'Pagination']
        },
        {
          name: 'Scalability',
          items: ['Load balancing', 'Caching strategies', 'CDN', 'Database sharding', 'Read replicas']
        }
      ]
    },
    dataEngineering: {
      title: 'Data Engineering & ETL',
      categories: [
        {
          name: 'ETL/ELT',
          items: ['Apache Airflow', 'Apache Nifi']
        },
        {
          name: 'Data Quality',
          items: ['Great Expectations']
        },
        {
          name: 'Data Catalog',
          items: ['Apache Atlas']
        }
      ]
    },
    performance: {
      title: 'Performance & Optimization',
      categories: [
        {
          name: 'Profiling',
          items: ['cProfile', 'Py-Spy', 'memory_profiler', 'Chrome DevTools']
        },
        {
          name: 'Caching',
          items: ['Redis', 'Memcached', 'Varnish', 'CDN caching']
        },
        {
          name: 'Load Testing',
          items: ['JMeter', 'Locust', 'k6', 'Gatling', 'Apache Bench']
        },
        {
          name: 'Model Optimization',
          items: ['Quantization', 'Pruning', 'Knowledge Distillation', 'ONNX', 'TensorRT']
        }
      ]
    },
    realTime: {
      title: 'Real-Time & Streaming',
      categories: [
        {
          name: 'Streaming Platforms',
          items: ['Apache Kafka', 'Apache Flink', 'Apache Storm', 'Kafka Streams', 'Spark Streaming']
        },
        {
          name: 'Cloud Streaming',
          items: ['AWS Kinesis', 'Azure Event Hubs', 'GCP Pub/Sub']
        },
        {
          name: 'Real-Time Communication',
          items: ['WebSockets', 'Server-Sent Events', 'Socket.io']
        }
      ]
    },
    documentation: {
      title: 'Documentation & Collaboration',
      categories: [
        {
          name: 'Documentation',
          items: ['Confluence', 'Notion', 'Swagger/OpenAPI', 'MkDocs']
        },
        {
          name: 'Project Management',
          items: ['Jira', 'Asana', 'Linear', 'Trello', 'Monday.com']
        },
        {
          name: 'Communication',
          items: ['Slack', 'Microsoft Teams', 'Discord']
        }
      ]
    },
    specializedAI: {
      title: 'Specialized AI/ML Areas',
      categories: [
        {
          name: 'LLM Fine-tuning',
          items: ['LoRA', 'QLoRA', 'PEFT', 'Full Fine-tuning']
        },
        {
          name: 'Prompt Engineering',
          items: ['Zero-shot', 'Few-shot', 'Chain-of-Thought', 'ReAct']
        },
        {
          name: 'Prompt Version-Control',
          items: ['Promptlayer']
        },
        {
          name: 'Model Compression',
          items: ['Quantization (INT8, INT4)', 'Distillation', 'Pruning']
        },
        {
          name: 'Distributed Training',
          items: ['DeepSpeed', 'Megatron-LM', 'Horovod', 'Ray Train']
        },
        {
          name: 'Edge AI',
          items: ['TensorFlow Lite', 'Core ML', 'ONNX Runtime Mobile']
        },
        {
          name: 'MLOps Best Practices',
          items: ['A/B Testing', 'Canary Deployment', 'Shadow Mode', 'Blue-Green Deployment']
        }
      ]
    },
    mathematics: {
      title: 'Mathematics & Statistical Foundations for AI/ML',
      categories: [
        {
          name: 'Linear Algebra',
          items: ['Vectors, Matrices, Tensors', 'Matrix Operations (Multiplication, Transpose, Inverse)', 'Eigenvalues & Eigenvectors', 'Singular Value Decomposition (SVD)', 'Principal Component Analysis (PCA)', 'Matrix Factorization', 'Vector Spaces & Linear Transformations']
        },
        {
          name: 'Calculus & Optimization',
          items: ['Differential Calculus (Derivatives, Partial Derivatives)', 'Gradient Descent & Variants (SGD, Adam, RMSprop, AdaGrad)', 'Backpropagation', 'Chain Rule', 'Convex Optimization', 'Lagrange Multipliers', 'Newton\'s Method', 'Gradient Clipping', 'Learning Rate Scheduling']
        },
        {
          name: 'Probability & Statistics',
          items: ['Probability Distributions (Normal, Binomial, Poisson, Exponential, Bernoulli)', 'Conditional Probability & Bayes Theorem', 'Maximum Likelihood Estimation (MLE)', 'Maximum A Posteriori (MAP)', 'Expectation-Maximization (EM) Algorithm', 'Hypothesis Testing (t-test, chi-square, ANOVA)', 'p-values, Confidence Intervals', 'Statistical Significance', 'Central Limit Theorem', 'Law of Large Numbers']
        },
        {
          name: 'Regression Analysis',
          subcategories: [
            { name: 'Linear Regression', items: ['Simple', 'Multiple', 'Polynomial'] },
            { name: 'Regularization', items: ['Ridge (L2)', 'Lasso (L1)', 'Elastic Net'] },
            { name: 'Logistic Regression', items: ['Binary', 'Multinomial', 'Ordinal'] },
            { name: 'Non-linear Regression', items: ['Polynomial', 'Spline', 'Local Regression (LOESS)'] },
            { name: 'Advanced Regression', items: ['Generalized Linear Models (GLM)', 'Poisson Regression', 'Negative Binomial Regression', 'Quantile Regression', 'Robust Regression', 'Bayesian Regression'] },
            { name: 'Time Series Regression', items: ['ARIMA', 'SARIMA', 'Prophet'] },
            { name: 'Evaluation Metrics', items: ['R²', 'Adjusted R²', 'MSE', 'RMSE', 'MAE', 'MAPE'] }
          ]
        },
        {
          name: 'Classification',
          subcategories: [
            { name: 'Binary Classification', items: ['Logistic Regression', 'SVM', 'Perceptron'] },
            { name: 'Multi-class Classification', items: ['One-vs-Rest', 'One-vs-One', 'Softmax'] },
            { name: 'Probabilistic Classifiers', items: ['Naive Bayes', 'Gaussian Naive Bayes'] },
            { name: 'Tree-based', items: ['Decision Trees', 'Random Forest', 'Extra Trees'] },
            { name: 'Boosting', items: ['AdaBoost', 'Gradient Boosting', 'XGBoost', 'LightGBM', 'CatBoost'] },
            { name: 'Neural Networks', items: ['MLPs', 'CNNs for classification'] },
            { name: 'Ensemble Methods', items: ['Bagging', 'Stacking', 'Voting'] },
            { name: 'Evaluation Metrics', items: ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC Curve', 'AUC-ROC', 'Precision-Recall Curve', 'Confusion Matrix', 'Cohen\'s Kappa', 'Matthews Correlation Coefficient (MCC)', 'Log Loss (Cross-Entropy)'] }
          ]
        },
        {
          name: 'Clustering & Unsupervised Learning',
          subcategories: [
            { name: 'Partitioning', items: ['K-Means', 'K-Medoids', 'K-Modes'] },
            { name: 'Hierarchical', items: ['Agglomerative', 'Divisive', 'Dendrogram'] },
            { name: 'Density-based', items: ['DBSCAN', 'OPTICS', 'HDBSCAN'] },
            { name: 'Distribution-based', items: ['Gaussian Mixture Models (GMM)', 'Expectation-Maximization'] },
            { name: 'Dimensionality Reduction', items: ['PCA (Principal Component Analysis)', 't-SNE (t-Distributed Stochastic Neighbor Embedding)', 'UMAP (Uniform Manifold Approximation and Projection)', 'Autoencoders', 'Variational Autoencoders (VAE)', 'LDA (Linear Discriminant Analysis)', 'ICA (Independent Component Analysis)'] },
            { name: 'Evaluation Metrics', items: ['Silhouette Score', 'Davies-Bouldin Index', 'Calinski-Harabasz Index', 'Elbow Method'] }
          ]
        },
        {
          name: 'Time Series Analysis',
          subcategories: [
            { name: 'Components', items: ['Trend', 'Seasonality', 'Cyclic', 'Irregular'] },
            { name: 'Models', items: ['ARIMA', 'SARIMA', 'ARIMAX', 'Exponential Smoothing (Holt-Winters)', 'Prophet', 'NeuralProphet', 'LSTM', 'GRU for Time Series', 'Temporal Convolutional Networks (TCN)', 'Transformer-based (Temporal Fusion Transformer)'] },
            { name: 'Stationarity', items: ['ADF Test', 'KPSS Test', 'Differencing'] },
            { name: 'Autocorrelation', items: ['ACF', 'PACF'] },
            { name: 'Forecasting Evaluation', items: ['MAE', 'RMSE', 'MAPE', 'SMAPE', 'MASE'] }
          ]
        },
        {
          name: 'Recommendation Systems',
          items: ['Collaborative Filtering: User-based, Item-based', 'Matrix Factorization: SVD, NMF, ALS', 'Content-Based Filtering', 'Hybrid Methods', 'Deep Learning: Neural Collaborative Filtering, AutoRec', 'Evaluation: Precision@K, Recall@K, NDCG, MAP, MRR']
        },
        {
          name: 'Natural Language Processing (NLP) Mathematics',
          subcategories: [
            { name: 'Text Representation', items: ['Bag of Words (BoW)', 'TF-IDF', 'Word Embeddings (Word2Vec, GloVe, FastText)', 'Contextual Embeddings (BERT, GPT, ELMo)'] },
            { name: 'Sequence Models', items: ['Hidden Markov Models (HMM)', 'RNN', 'LSTM', 'GRU', 'Transformer Architecture', 'Attention Mechanisms (Self-Attention, Multi-Head Attention)'] },
            { name: 'Language Modeling', items: ['N-grams', 'Perplexity'] },
            { name: 'Topic Modeling', items: ['LDA (Latent Dirichlet Allocation)', 'NMF'] }
          ]
        },
        {
          name: 'Computer Vision Mathematics',
          items: ['Convolution Operations: Kernels, Filters, Stride, Padding', 'Pooling: Max Pooling, Average Pooling', 'Image Transformations: Affine, Perspective, Rotation', 'Object Detection: IoU (Intersection over Union), Non-Max Suppression (NMS)', 'Architectures: CNN, ResNet, VGG, Inception, YOLO, R-CNN family', 'Evaluation: mAP (mean Average Precision), Dice Coefficient, IoU']
        },
        {
          name: 'Deep Learning Fundamentals',
          subcategories: [
            { name: 'Activation Functions', items: ['ReLU', 'Leaky ReLU', 'Sigmoid', 'Tanh', 'Softmax', 'GELU', 'Swish'] },
            { name: 'Loss Functions', items: ['MSE', 'MAE', 'Huber Loss', 'Cross-Entropy (Binary, Categorical)', 'Hinge Loss', 'Triplet Loss', 'Contrastive Loss', 'Focal Loss', 'Dice Loss'] },
            { name: 'Normalization', items: ['Batch Norm', 'Layer Norm', 'Instance Norm', 'Group Norm'] },
            { name: 'Regularization', items: ['Dropout', 'DropConnect', 'L1/L2 Regularization', 'Early Stopping'] },
            { name: 'Weight Initialization', items: ['Xavier/Glorot', 'He', 'LeCun'] },
            { name: 'Architectures', items: ['Feedforward Neural Networks (FNN)', 'Convolutional Neural Networks (CNN)', 'Recurrent Neural Networks (RNN, LSTM, GRU)', 'Transformers', 'Vision Transformers (ViT)', 'Generative Adversarial Networks (GANs)', 'Variational Autoencoders (VAE)'] }
          ]
        },
        {
          name: 'Feature Engineering & Selection',
          subcategories: [
            { name: 'Feature Scaling', items: ['Standardization', 'Normalization', 'Min-Max Scaling'] },
            { name: 'Encoding', items: ['One-Hot', 'Label Encoding', 'Target Encoding', 'Frequency Encoding'] },
            { name: 'Feature Selection', items: ['Filter Methods (Correlation, Chi-square, ANOVA)', 'Wrapper Methods (Forward/Backward Selection, RFE)', 'Embedded Methods (Lasso, Tree-based importance)'] },
            { name: 'Feature Extraction', items: ['PCA', 'LDA', 'Autoencoders'] },
            { name: 'Interaction Features', items: ['Polynomial Features', 'Cross Products'] }
          ]
        },
        {
          name: 'Model Evaluation & Validation',
          items: ['Cross-Validation: K-Fold, Stratified K-Fold, Leave-One-Out, Time Series Split', 'Bias-Variance Tradeoff', 'Overfitting & Underfitting', 'Learning Curves', 'Validation Curves', 'Statistical Tests: t-test, Wilcoxon, McNemar\'s test', 'Fairness Metrics: Demographic Parity, Equal Opportunity, Disparate Impact']
        },
        {
          name: 'Experiment Design',
          items: ['A/B Testing', 'Multivariate Testing', 'Statistical Power Analysis', 'Sample Size Calculation', 'Randomization', 'Stratification', 'ANOVA (Analysis of Variance)', 'Multiple Hypothesis Testing (Bonferroni, FDR)']
        }
      ]
    }
  }

  return (
    <main className="min-h-screen relative">
      <Navigation />
      
      {/* Hero Section */}
      <motion.section
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.8 }}
        className="relative overflow-hidden bg-black text-white pt-20 md:pt-24"
      >
        {/* Animated Background Elements */}
        <div className="absolute inset-0 overflow-hidden">
          <div className="absolute -top-40 -right-40 w-80 h-80 bg-red-600 rounded-full mix-blend-screen filter blur-3xl opacity-30 animate-blob"></div>
          <div className="absolute -bottom-40 -left-40 w-80 h-80 bg-yellow-500 rounded-full mix-blend-screen filter blur-3xl opacity-30 animate-blob animation-delay-2000"></div>
          <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-80 h-80 bg-red-500 rounded-full mix-blend-screen filter blur-3xl opacity-20 animate-blob animation-delay-4000"></div>
        </div>
        
        <div className="absolute inset-0 bg-gradient-to-b from-black/50 to-black"></div>
        
        <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-32 md:py-40 lg:py-48">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2, duration: 0.8 }}
            className="text-center"
          >
            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.4, duration: 0.6 }}
              className="mb-6"
            >
              <h1 className="text-5xl sm:text-6xl md:text-7xl lg:text-8xl font-extrabold mb-4 bg-clip-text text-transparent bg-gradient-to-r from-red-500 via-red-400 to-yellow-500 leading-tight">
                Kavin N Ranganathan
              </h1>
            </motion.div>
            
            <motion.p
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.6, duration: 0.6 }}
              className="text-xl sm:text-2xl md:text-3xl mb-6 text-gray-200 font-semibold"
            >
              AI/ML - Solution Architect | Serial Entrepreneurship — In Motion
            </motion.p>
            
            <motion.p
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.8, duration: 0.6 }}
              className="text-lg md:text-xl text-gray-300 max-w-3xl mx-auto leading-relaxed"
            >
              Building ventures and architecting AI solutions that generate competitive advantage. Technical founder with a track record in accelerating businesses and closing strategic deals.
            </motion.p>
            
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 1, duration: 0.6 }}
              className="mt-10"
            >
              <motion.a
                href="#tech-stacks"
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                className="inline-block px-8 py-4 bg-gradient-to-r from-red-600 to-red-500 backdrop-blur-md text-white rounded-full font-semibold shadow-xl shadow-red-500/30 hover:shadow-2xl hover:shadow-red-500/50 transition-all duration-300 border border-red-500/30 hover:border-red-400/50"
              >
                Explore Tech Stacks
              </motion.a>
            </motion.div>
          </motion.div>
        </div>
        
        {/* Decorative Wave */}
        <div className="absolute bottom-0 left-0 right-0">
          <svg className="w-full h-20 md:h-24" viewBox="0 0 1440 120" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M0 120L60 105C120 90 240 60 360 45C480 30 600 30 720 37.5C840 45 960 60 1080 67.5C1200 75 1320 75 1380 75L1440 75V120H1380C1320 120 1200 120 1080 120C960 120 840 120 720 120C600 120 480 120 360 120C240 120 120 120 60 120H0Z" fill="url(#gradient)" />
            <defs>
              <linearGradient id="gradient" x1="0" y1="0" x2="1440" y2="0" gradientUnits="userSpaceOnUse">
                <stop stopColor="#000000" />
                <stop offset="0.5" stopColor="#0a0a0a" />
                <stop offset="1" stopColor="#000000" />
              </linearGradient>
            </defs>
          </svg>
        </div>
      </motion.section>

      {/* My Story Section */}
      <section id="my-story" className="relative bg-black py-20 md:py-32 overflow-hidden">
        {/* Background Effects */}
        <div className="absolute inset-0 overflow-hidden">
          <div className="absolute top-20 right-10 w-72 h-72 bg-red-600 rounded-full mix-blend-screen filter blur-3xl opacity-10"></div>
          <div className="absolute bottom-20 left-10 w-72 h-72 bg-yellow-500 rounded-full mix-blend-screen filter blur-3xl opacity-10"></div>
        </div>

        <div className="relative max-w-5xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: "-100px" }}
            transition={{ duration: 0.8 }}
            className="text-center mb-16"
          >
            <motion.h2
              initial={{ opacity: 0, scale: 0.9 }}
              whileInView={{ opacity: 1, scale: 1 }}
              viewport={{ once: true }}
              transition={{ duration: 0.6 }}
              className="text-4xl sm:text-5xl md:text-6xl font-extrabold mb-4 text-transparent bg-clip-text bg-gradient-to-r from-red-500 via-red-600 to-yellow-500"
            >
              My Story
            </motion.h2>
            <motion.p
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: 0.2, duration: 0.6 }}
              className="text-2xl md:text-3xl font-bold text-gray-200 mb-2"
            >
              Break the Tradition. Bring the Truth.
            </motion.p>
          </motion.div>

          <div className="space-y-8 md:space-y-12">
            {/* Opening */}
            <motion.div
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: "-50px" }}
              transition={{ duration: 0.6 }}
              className="bg-gray-900/50 backdrop-blur-lg rounded-2xl p-6 md:p-8 border border-gray-800/60 shadow-xl shadow-red-500/10"
            >
              <p className="text-lg md:text-xl text-gray-300 leading-relaxed">
                I am driven by three questions that shape how I think, learn, and build:
              </p>
              <div className="mt-4 flex flex-wrap gap-4">
                <span className="px-4 py-2 bg-red-600/20 border border-red-500/30 rounded-lg text-red-400 font-semibold text-lg">
                  Why?
                </span>
                <span className="px-4 py-2 bg-yellow-500/20 border border-yellow-500/30 rounded-lg text-yellow-400 font-semibold text-lg">
                  When?
                </span>
                <span className="px-4 py-2 bg-red-600/20 border border-red-500/30 rounded-lg text-red-400 font-semibold text-lg">
                  How?
                </span>
              </div>
            </motion.div>

            {/* Journey */}
            <motion.div
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: "-50px" }}
              transition={{ delay: 0.1, duration: 0.6 }}
              className="bg-gray-900/50 backdrop-blur-lg rounded-2xl p-6 md:p-8 border border-gray-800/60 shadow-xl shadow-red-500/10"
            >
              <p className="text-base md:text-lg text-gray-300 leading-relaxed mb-4">
                My journey has been anything but conventional.
              </p>
              <p className="text-base md:text-lg text-gray-300 leading-relaxed mb-4">
                I've faced over <span className="text-red-400 font-bold">700 rejections</span> and failed countless interviews. I don't hold a traditional technical degree—yet I am a technologist by practice, passion, and mindset.
              </p>
            </motion.div>

            {/* Academic Background */}
            <motion.div
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: "-50px" }}
              transition={{ delay: 0.2, duration: 0.6 }}
              className="bg-gray-900/50 backdrop-blur-lg rounded-2xl p-6 md:p-8 border border-gray-800/60 shadow-xl shadow-red-500/10"
            >
              <h3 className="text-xl md:text-2xl font-bold text-gray-100 mb-4 flex items-center">
                <span className="w-1 h-8 bg-gradient-to-b from-red-500 to-yellow-500 rounded-full mr-3"></span>
                My Academic Foundation
              </h3>
              <div className="space-y-3 text-gray-300">
                <p className="text-base md:text-lg">
                  <span className="text-red-400 font-semibold">B.Com</span> – Business Process & Services
                </p>
                <p className="text-base md:text-lg">
                  <span className="text-red-400 font-semibold">MBA</span> – Human Resources & Marketing
                </p>
              </div>
              <p className="text-base md:text-lg text-gray-300 leading-relaxed mt-4">
                Rather than limiting me, this background gave me a deep understanding of how businesses truly operate. I've worked hands-on across multiple verticals—HR, operations, sales, strategy, and go-to-market execution. This perspective allows me to approach technology not merely as code, but as a business enabler.
              </p>
            </motion.div>

            {/* Career Journey */}
            <motion.div
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: "-50px" }}
              transition={{ delay: 0.3, duration: 0.6 }}
              className="bg-gray-900/50 backdrop-blur-lg rounded-2xl p-6 md:p-8 border border-gray-800/60 shadow-xl shadow-red-500/10"
            >
              <h3 className="text-xl md:text-2xl font-bold text-gray-100 mb-4 flex items-center">
                <span className="w-1 h-8 bg-gradient-to-b from-red-500 to-yellow-500 rounded-full mr-3"></span>
                My Career Evolution
              </h3>
              <p className="text-base md:text-lg text-gray-300 leading-relaxed mb-6">
                I've worked with <span className="text-red-400 font-bold">eight different organizations</span>, deliberately taking on diverse roles to understand systems from the ground up.
              </p>
              
              <div className="space-y-4">
                <div className="pl-4 border-l-2 border-red-500/50">
                  <p className="text-base md:text-lg text-gray-300 mb-2">
                    I began my career as an <span className="text-yellow-400 font-semibold">HR Intern</span>, where I learned the fundamentals of talent acquisition and HR operations—sourcing, screening, interviewing, onboarding, and employee record management.
                  </p>
                </div>
                <div className="pl-4 border-l-2 border-yellow-500/50">
                  <p className="text-base md:text-lg text-gray-300 mb-2">
                    Through campus placements, I was selected by multiple companies and transitioned into a <span className="text-yellow-400 font-semibold">Market Research Analyst</span> role, gaining exposure to pre-sales, business development, digital marketing, virtual assistance, and document management.
                  </p>
                </div>
                <div className="pl-4 border-l-2 border-red-500/50">
                  <p className="text-base md:text-lg text-gray-300 mb-2">
                    Despite steady progress, I felt disconnected from my purpose. Alongside my job, I began pursuing a <span className="text-red-400 font-semibold">Professional Diploma in AI & ML</span>—a decision that completely reshaped my direction.
                  </p>
                </div>
                <div className="pl-4 border-l-2 border-yellow-500/50">
                  <p className="text-base md:text-lg text-gray-300 mb-2">
                    Eventually, I left my job to fully commit to learning, building, and failing forward.
                  </p>
                </div>
              </div>
            </motion.div>

            {/* Rapid Evolution */}
            <motion.div
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: "-50px" }}
              transition={{ delay: 0.4, duration: 0.6 }}
              className="bg-gray-900/50 backdrop-blur-lg rounded-2xl p-6 md:p-8 border border-gray-800/60 shadow-xl shadow-red-500/10"
            >
              <h3 className="text-xl md:text-2xl font-bold text-gray-100 mb-4 flex items-center">
                <span className="w-1 h-8 bg-gradient-to-b from-red-500 to-yellow-500 rounded-full mr-3"></span>
                What Followed Was a Rapid Evolution
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mt-4">
                {[
                  'AI/ML Intern (twice)',
                  'Business Analyst',
                  'AI/ML Trainer & Developer',
                  'Lead GenAI Developer',
                  'AI Solution Architect'
                ].map((role, index) => (
                  <motion.div
                    key={index}
                    initial={{ opacity: 0, x: -20 }}
                    whileInView={{ opacity: 1, x: 0 }}
                    viewport={{ once: true }}
                    transition={{ delay: index * 0.1, duration: 0.4 }}
                    className="px-4 py-3 bg-gray-800/50 border border-gray-700/50 rounded-lg hover:border-red-500/50 transition-colors"
                  >
                    <p className="text-gray-200 font-medium flex items-center">
                      <span className="w-2 h-2 bg-red-500 rounded-full mr-2"></span>
                      {role}
                    </p>
                  </motion.div>
                ))}
              </div>
              <p className="text-base md:text-lg text-gray-300 leading-relaxed mt-6">
                Each role reinforced a core belief: <span className="text-red-400 font-semibold">technology alone is not enough. It must be usable, scalable, and commercially relevant.</span>
              </p>
            </motion.div>

            {/* Co-Founder */}
            <motion.div
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: "-50px" }}
              transition={{ delay: 0.5, duration: 0.6 }}
              className="bg-gradient-to-br from-red-600/20 to-yellow-500/20 backdrop-blur-lg rounded-2xl p-6 md:p-8 border border-red-500/30 shadow-xl shadow-red-500/20"
            >
              <p className="text-base md:text-lg text-gray-200 leading-relaxed">
                Also a <span className="text-yellow-400 font-bold">Co-Founder of Grevya Technologies</span>, where I focus on building future-ready solutions at the intersection of technology, business, and strategy.
              </p>
            </motion.div>

            {/* Closing Statement */}
            <motion.div
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: "-50px" }}
              transition={{ delay: 0.6, duration: 0.6 }}
              className="bg-gray-900/50 backdrop-blur-lg rounded-2xl p-6 md:p-8 border-2 border-red-500/50 shadow-xl shadow-red-500/20"
            >
              <div className="text-center space-y-4">
                <p className="text-xl md:text-2xl font-bold text-gray-100">
                  I don't follow traditional paths.
                </p>
                <p className="text-xl md:text-2xl font-bold text-red-400">
                  I question them.
                </p>
                <p className="text-xl md:text-2xl font-bold text-yellow-400">
                  I redesign them.
                </p>
                <p className="text-xl md:text-2xl font-bold text-gray-100">
                  And I build something better—grounded in truth, resilience, and execution.
                </p>
              </div>
            </motion.div>
          </div>
        </div>
      </section>

      {/* Tech Stacks Section */}
      <section id="tech-stacks" className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20 md:py-28 scroll-mt-20">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-100px" }}
          transition={{ duration: 0.8 }}
          className="text-center mb-16 md:mb-20"
        >
          <motion.h2
            initial={{ opacity: 0, scale: 0.9 }}
            whileInView={{ opacity: 1, scale: 1 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6 }}
            className="text-4xl sm:text-5xl md:text-6xl font-extrabold mb-6 text-transparent bg-clip-text bg-gradient-to-r from-red-500 via-red-600 to-yellow-500"
          >
            Tech Stacks
          </motion.h2>
          <motion.p
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: 0.2, duration: 0.6 }}
            className="text-gray-400 text-lg md:text-xl max-w-3xl mx-auto leading-relaxed mb-6"
          >
            A comprehensive overview of technologies and tools I work with
          </motion.p>
          
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: 0.3, duration: 0.6 }}
            className="flex justify-center"
          >
            <motion.button
              onClick={exportTechSkills}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              className="inline-flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-red-600 to-red-500 text-white rounded-full font-semibold shadow-lg shadow-red-500/30 hover:shadow-xl hover:shadow-red-500/50 transition-all duration-300 border border-red-400/50 hover:border-red-300/70"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
              Export Tech Skills
            </motion.button>
          </motion.div>
        </motion.div>

        <div className="space-y-16 md:space-y-24">
          <TechStackSection data={techStacks.programmingLanguages} />
          <TechStackSection data={techStacks.frontendDevelopment} />
          <TechStackSection data={techStacks.backendDevelopment} />
          <TechStackSection data={techStacks.versionControl} />
          <TechStackSection data={techStacks.databases} />
          <TechStackSection data={techStacks.visualization} />
          <TechStackSection data={techStacks.mlops} />
          <TechStackSection data={techStacks.aiops} />
          <TechStackSection data={techStacks.cicd} />
          <TechStackSection data={techStacks.openSource} />
          <TechStackSection data={techStacks.agenticAI} />
          <TechStackSection data={techStacks.aiML} />
          <TechStackSection data={techStacks.cloudPlatforms} />
          <TechStackSection data={techStacks.apiIntegration} />
          <TechStackSection data={techStacks.security} />
          <TechStackSection data={techStacks.architecture} />
          <TechStackSection data={techStacks.dataEngineering} />
          <TechStackSection data={techStacks.performance} />
          <TechStackSection data={techStacks.realTime} />
          <TechStackSection data={techStacks.documentation} />
          <TechStackSection data={techStacks.specializedAI} />
          <TechStackSection data={techStacks.mathematics} />
        </div>
      </section>

      {/* Contact Section */}
      <section id="contact" className="relative bg-black py-20 md:py-32 overflow-hidden">
        {/* Background Effects */}
        <div className="absolute inset-0 overflow-hidden">
          <div className="absolute top-10 left-20 w-96 h-96 bg-red-600 rounded-full mix-blend-screen filter blur-3xl opacity-10"></div>
          <div className="absolute bottom-10 right-20 w-96 h-96 bg-yellow-500 rounded-full mix-blend-screen filter blur-3xl opacity-10"></div>
        </div>

        <div className="relative max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: "-100px" }}
            transition={{ duration: 0.8 }}
            className="text-center mb-12"
          >
            <motion.h2
              initial={{ opacity: 0, scale: 0.9 }}
              whileInView={{ opacity: 1, scale: 1 }}
              viewport={{ once: true }}
              transition={{ duration: 0.6 }}
              className="text-4xl sm:text-5xl md:text-6xl font-extrabold mb-4 text-transparent bg-clip-text bg-gradient-to-r from-red-500 via-red-600 to-yellow-500"
            >
              Contact
            </motion.h2>
            <motion.p
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: 0.2, duration: 0.6 }}
              className="text-lg md:text-xl text-gray-400 max-w-2xl mx-auto"
            >
              Let's connect and explore opportunities together
            </motion.p>
          </motion.div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 md:gap-8">
            {/* Phone */}
            <motion.a
              href="tel:+919597375091"
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: "-50px" }}
              transition={{ delay: 0.2, duration: 0.6 }}
              whileHover={{ scale: 1.05, y: -5 }}
              className="group bg-gray-900/50 backdrop-blur-lg rounded-2xl p-8 border border-gray-800/60 hover:border-red-500/50 shadow-xl shadow-red-500/10 hover:shadow-2xl hover:shadow-red-500/20 transition-all duration-500 relative overflow-hidden"
            >
              <div className="absolute inset-0 bg-gradient-to-br from-red-500/0 to-yellow-500/0 group-hover:from-red-500/10 group-hover:to-yellow-500/10 transition-all duration-500 pointer-events-none"></div>
              <div className="relative z-10">
                <div className="flex items-center justify-center w-16 h-16 mb-4 bg-red-600/20 rounded-full border border-red-500/30 group-hover:bg-red-600/30 group-hover:border-red-500/50 transition-all duration-300">
                  <svg className="w-8 h-8 text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 5a2 2 0 012-2h3.28a1 1 0 01.948.684l1.498 4.493a1 1 0 01-.502 1.21l-2.257 1.13a11.042 11.042 0 005.516 5.516l1.13-2.257a1 1 0 011.21-.502l4.493 1.498a1 1 0 01.684.949V19a2 2 0 01-2 2h-1C9.716 21 3 14.284 3 6V5z" />
                  </svg>
                </div>
                <h3 className="text-lg font-semibold text-gray-300 mb-2 group-hover:text-red-400 transition-colors">
                  Phone
                </h3>
                <p className="text-xl md:text-2xl font-bold text-red-400 group-hover:text-red-300 transition-colors">
                  +91 9597375091
                </p>
              </div>
            </motion.a>

            {/* Email */}
            <motion.a
              href="mailto:kavin.nr19@gmail.com"
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: "-50px" }}
              transition={{ delay: 0.3, duration: 0.6 }}
              whileHover={{ scale: 1.05, y: -5 }}
              className="group bg-gray-900/50 backdrop-blur-lg rounded-2xl p-8 border border-gray-800/60 hover:border-yellow-500/50 shadow-xl shadow-yellow-500/10 hover:shadow-2xl hover:shadow-yellow-500/20 transition-all duration-500 relative overflow-hidden"
            >
              <div className="absolute inset-0 bg-gradient-to-br from-yellow-500/0 to-red-500/0 group-hover:from-yellow-500/10 group-hover:to-red-500/10 transition-all duration-500 pointer-events-none"></div>
              <div className="relative z-10">
                <div className="flex items-center justify-center w-16 h-16 mb-4 bg-yellow-500/20 rounded-full border border-yellow-500/30 group-hover:bg-yellow-500/30 group-hover:border-yellow-500/50 transition-all duration-300">
                  <svg className="w-8 h-8 text-yellow-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 8l7.89 5.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                  </svg>
                </div>
                <h3 className="text-lg font-semibold text-gray-300 mb-2 group-hover:text-yellow-400 transition-colors">
                  Email
                </h3>
                <p className="text-lg md:text-xl font-bold text-yellow-400 group-hover:text-yellow-300 transition-colors break-all">
                  kavin.nr19@gmail.com
                </p>
              </div>
            </motion.a>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="relative bg-black text-white py-12 md:py-16 mt-20 border-t border-gray-800">
        <div className="absolute inset-0 bg-gradient-to-r from-red-600/5 to-yellow-500/5"></div>
        <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center">
            <motion.p
              initial={{ opacity: 0 }}
              whileInView={{ opacity: 1 }}
              viewport={{ once: true }}
              transition={{ duration: 0.6 }}
              className="text-gray-400 text-sm md:text-base"
            >
              © {new Date().getFullYear()} Kavin N Ranganathan. All rights reserved.
            </motion.p>
          </div>
        </div>
      </footer>

      {/* Scroll to Top Button */}
      <AnimatePresence>
        {showScrollTop && (
          <motion.button
            initial={{ opacity: 0, scale: 0, y: 20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0, y: 20 }}
            transition={{ duration: 0.3, type: "spring" }}
            onClick={scrollToTop}
            whileHover={{ scale: 1.1, y: -5 }}
            whileTap={{ scale: 0.9 }}
            className="fixed bottom-8 right-8 z-50 w-14 h-14 md:w-16 md:h-16 bg-gradient-to-br from-red-600 to-red-500 text-white rounded-full shadow-lg shadow-red-500/30 hover:shadow-xl hover:shadow-red-500/50 border-2 border-red-400/50 hover:border-red-300/70 transition-all duration-300 flex items-center justify-center group"
            aria-label="Scroll to top"
          >
            <svg
              className="w-6 h-6 md:w-7 md:h-7 transform group-hover:-translate-y-1 transition-transform duration-300"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2.5}
                d="M5 10l7-7m0 0l7 7m-7-7v18"
              />
            </svg>
          </motion.button>
        )}
      </AnimatePresence>
    </main>
  )
}

