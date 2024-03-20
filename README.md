# LLama-2-For-Lib-Rec

## Overview

This project is part of a master's thesis that involves generating software library recommendations for Java applications. The project uses a machine learning model to process project data and generate recommendations in Maven format.

## Project Structure

```plaintext
project-root
│
├── output                  # Generated recommendations will be stored here
│   └── {date}
│       └── recommendations_{time}.txt
│
├── data                    # You need to provide your own dataset here
│   └── processed           # Processed data will be saved here by data_process.py
│       └── parsed_data.csv
│
├── preprocessing           # Scripts for data preprocessing
│   ├── __pycache__
│   ├── __init__.py
│   └── data_process.py
│
├── prompt                  # Scripts for generating the recommendation prompt
│   ├── __pycache__
│   ├── __init__.py
│   └── prompt.py
│
├── notebooks               # Jupyter notebooks (if any)
│
├── .env                    # Environment variables and API keys
├── .gitignore              # Specifies intentionally untracked files to ignore
├── main.py                 # Main script to run the project
├── requirements.txt        # Dependencies for the project
└── README.md               # Documentation for the project
```

## Prerequisites

- Python 3.9 or higher
- Pip (Python package installer)
- Virtual environment (recommended)

## Setup

Before running the project, you must create a `data` directory with a `processed` subdirectory where `parsed_data.csv` will be stored. You can obtain this data file by running the `data_process.py` script after populating the data directory with your `project_dependencies_readmes.csv`.

## Environment Variables

You must provide a ``.env` file in the project root with the following variable:
```bash 
HF_TOKEN=your_huggingface_api_token 
```
Make sure to replace `your_huggingface_api_token` with your actual HuggingFace API token.

## Installing Dependencies

To install the required Python packages, navigate to the project root and run the following commands:

- Windows:
```bash
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```
-macOS:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Running the Scripts

To run the preprocessing and prompt generation scripts, follow these steps:
- Ensure the data directory contains your project_dependencies_readmes.csv.
- Execute `data_process.py` to clean the data and generate `parsed_data.csv`.
- Run `prompt.py` to generate recommendations.
    - Windows:
    ```bash
    python preprocessing\data_process.py
    python prompt\prompt.py
    ```
    or simpy 
    ```bash
    python main.py
    ```
    - macOS:
    ```bash
    python3 preprocessing/data_process.py
    python3 prompt/prompt.py
    ```
     or simpy 
    ```bash
    python3 main.py
    ```


## Acknowledgments

Will be soon......
