MLOps Data Drift Simulation
Overview

This project demonstrates a practical MLOps workflow focused on data versioning, drift detection, experiment tracking, and model lifecycle management.

A binary image classifier (Square vs Circle) is trained and evaluated under changing data conditions. A simulated production drift (inverted images and noise) causes performance degradation. The system is then updated with a new dataset version to restore robustness.

Objectives

Detect and measure data drift

Version datasets using DVC

Track experiments and metrics with MLflow

Register and manage model versions

Enforce model input/output schema with MLflow Signature

Ensure full reproducibility between code, data, and models

MLOps Stack

Git – Code versioning

DVC – Data versioning

MinIO – S3-compatible storage backend

MLflow – Experiment tracking and Model Registry

Docker – Service orchestration

Scikit-learn – Model training

Key Concepts Demonstrated

Data drift simulation and monitoring

Dataset evolution (V1 → V2 → V3)

Model Registry lifecycle management

Reproducible training pipeline

Strict schema validation with Model Signature
