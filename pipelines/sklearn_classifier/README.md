# Modelo de Classificação de flores do tipo Iris
Essa pipeline utiliza modelos de classificação do sklearn para desenvolver um modelo para classificação baseado no [Iris dataset](https://archive.ics.uci.edu/ml/datasets/iris).

## Steps

### Prepare

### Featurize

### Train

### Evaluate

## Executando a pipeline local

## Estrutura do projeto
```console
$ tree
.
├── cm_classes.csv
├── data                      # <-- Directory with raw and intermediate data
│   ├── features              # <-- Extracted feature dataset
│   │   ├── test.pkl
│   │   └── train.pkl
│   └── prepared              # <-- Raw dataset splitted on train/test
│       ├── test.csv
│       └── train.csv
├── dvc.lock
├── dvc.yaml                  # <-- DVC pipeline file
├── model.pkl                 # <-- Trained model file
├── params.yaml               # <-- Parameters file
├── README.md
├── scores.json               # <-- Model final metrics
└── src
    ├── evaluate.py
    ├── featurize.py
    ├── prepare.py
    ├── requirements.txt          # <-- Pipeline requirements
    └── train.py

```