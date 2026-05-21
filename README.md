# Metadata Quality Assessment of Latvian Open Data Portal

This repository contains the Python scripts, processed datasets, and analysis results used in the bachelor thesis:

**“Latvijas Atvērto datu portāla datu kopu metadatu kvalitātes novērtēšana”**

## Running the analysis

### 1. Download metadata from CKAN API

```bash
python scripts/01_download_ckan.py
```

### 2. Run MQA analysis

```bash
python scripts/03_mqa_scoring.py
```
