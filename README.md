# Aadhaar Data Analysis Project

## Project Overview
This project aims to analyze Aadhaar enrolment and update data to simpler proactive planning for Aadhaar Seva Kendras. It predicts biometric update surges to avoid service denial and reduce operational strain.

## Setup Instructions

### Prerequisites
- Conda or Mamba installed
- Python 3.10+

### Installation
1. Clone the repository
2. Create the environment:
    ```bash
    conda env create -f environment.yml
    conda activate aadhaar_hackathon
    ```
    OR using pip:
    ```bash
    pip install -r requirements.txt
    ```

## Directory Structure
- `data/raw`: Place CSV files here (`Aadhaar_Enrolment_dataset.csv`, `Aadhaar_Demographic_Update_dataset.csv`, `Aadhaar_Biometric_Update_dataset.csv`)
- `data/processed`: For parquet files
- `notebooks`: Jupyter notebooks for analysis and forecasting
- `src`: Source code for ETL, features, and modeling
- `app`: Streamlit dashboard

## Deliverables
- `environment.yml`: Reproducible Python environment
- `notebooks/00_schema_analysis.ipynb`: Data exploration notebook
- `data_dictionary.md`: Documentation of data schemas
