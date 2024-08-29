#!/bin/sh
pip install torch transformers datasets wandb tqdm scikit-learn sentencepiece
pip install fastapi uvicorn
pip install "dvc[s3]"
git clone https://github.com/dmytro-spodarets/devops-qa-dataset.git
cd devops-qa-dataset
dvc pull
cd ..
