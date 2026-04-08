# CodeSense

A Deep Learning-powered CLI tool that reviews your Python code in real-time — right inside your terminal.

## What it does

- Bug risk score — fine-tuned CodeBERT model (94.8% accuracy)
- Code quality score  
- Complexity score
- Warnings for high-risk code

## Setup

git clone https://github.com/Jit-das01/codesense
cd codesense
python3 -m venv venv && source venv/bin/activate
pip install transformers torch rich click
python codesense.py yourfile.py

## Model

Fine-tuned microsoft/codebert-base on 3,000 Python functions from CodeSearchNet.
Training accuracy: 94.8%

## Tech Stack

Python · HuggingFace Transformers · CodeBERT · Rich · Click · PyTorch

Built by Jeet Das — github.com/Jit-das01
