# Simulation de Data Drift en MLOps

## Présentation

Ce projet met en œuvre un workflow MLOps reproductible autour de la versioning des données, la détection de drift, le suivi d’expériences et la gouvernance des modèles.

Un classifieur d’images binaire (Square vs Circle) est entraîné puis évalué sous différentes conditions de données. Une dérive simulée en production (images inversées et bruit) dégrade les performances. Le système est ensuite renforcé via une nouvelle version de dataset afin d’améliorer la robustesse.

## Objectifs

- Détecter et mesurer une dérive de données (data drift)
- Versionner les datasets avec DVC
- Suivre les expériences et métriques avec MLflow
- Enregistrer et gérer les versions de modèles via le Model Registry
- Enforcer un schéma d’entrée/sortie (Model Signature) pour sécuriser le déploiement
- Garantir la reproductibilité entre code, données et artefacts

## Stack MLOps

- Git : versioning du code
- DVC : versioning des données
- MinIO : stockage S3-compatible (backend DVC / artefacts)
- MLflow : tracking des expériences et Model Registry
- Docker : orchestration des services
- Scikit-learn : entraînement du modèle

## Points clés

1. Simulation et monitoring de data drift
2. Évolution contrôlée du dataset (V1 → V2 → V3)
3. Gouvernance des modèles (versions, staging) via MLflow Registry
4. Pipeline d’entraînement traçable et auditable
5. Validation stricte des entrées/sorties via Model Signature

## Author

Nassim GASTLI
MSc 2 Data Management & Artificial Intelligence, ECE Paris
MLOps & Machine Learning Engineering
