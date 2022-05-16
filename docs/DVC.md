# DVC

DVC é um sistema de controle de versão de código aberto para projetos de Machine Learning. 
O DVC não é apenas um Git para arquivos grandes, mas o sistema realmente atende a todas as minhas necessidades de experimento e gerenciamento de pipeline. 

Controle de versão para projetos de ML, precisamos considerar não apenas o código, mas também os dados e modelos. 
DVC é uma ferramenta fácil de usar que funciona no topo do Git. 
O DVC utiliza um arquivo dvc para ajudá-lo a controlar a versão de seus artefatos de ML e o Git é responsável pelo controle de versão do código e desse arquivo dvc.

Essa documentação mostra detalhes sobre:
- Como versionar dados em projetos de ML
- Como definir o armazenamento remoto que mantém seus artefatos de ML e recuperar artefatos de ML do armazenamento remoto para seu projeto local
- Como acompanhar os arquivos de dados quando você fez alterações ou adicionou um novo conjunto de dados
- Como alternar entre diferentes versões
- Como construir pipelines reprodutíveis de ML com DVC

[//]: # (- Como manter seu modelo em produção sempre atualizado.)

Para mais informações, consulte a [documentação oficial](https://dvc.org/doc).

Antes de iniciar o controle de versão, você precisará do DVC instalado em seu sistema. 
Verifique o [documento oficial do DVC sobre como instalar o DVC](https://dvc.org/doc/install). 

Então vamos supor que temos um exemplo simples de repositório git seguindo a estrutura abaixo:

```
exemplo-dvc-pipeline/ 
| 
|-- data/ 
    |-- training.csv 
|--metrics/ 
|-- model/ 
|-- src/ 
    |-- train.py
```

- data : o nosso dataset será armazenado nesta pasta. Neste exemplo, usamos o arquivo CSV como nosso arquivo de conjunto de treinamento
- model : o modelo treinado será armazenado nesta pasta
- métricas : outros artefatos de ML podem ser armazenados nesta pasta
- src : scripts python serão armazenados nesta pasta (por exemplo, train.py, validate.py)

O fluxo de trabalho das seguintes seções é mostrado na Figura 1 abaixo:

[//]: # (TODO: Figura 1)

**Controle de versão de artefatos de ML**

O DVC usa o chamado arquivo *.dvc que contém um hash md5 exclusivo para vincular o conjunto de dados ao projeto. 
O DVC armazena a cópia desse arquivo de dados no cache DVC, usando as duas primeiras letras do hash como nome da pasta. 
Em seguida, use o comando Git (por exemplo, git commit) para gravar este estágio.

Usando dvc add para iniciar o rastreamento de arquivos e gerar o arquivo .dvc 
```
dvc add data/training.csv
```

Usando git commit para a versão do arquivo .dvc
```
git add data/training.csv.dvc data/.gitignore 
git commit -m "Adicionar dados de treinamento"
```

Após esta etapa, existem alguns novos arquivos gerados na pasta do seu projeto, mostrados abaixo (somente as pastas necessárias, arquivos nesta etapa):

```
exemplo-dvc-pipeline/ 
| 
|-- data/ 
   |-- training.csv 
   |-- training.csv.dvc --> arquivo dvc para training.csv 
|-- .dvc/ 
   |-- cache/ 
      |-- a3/ --> uma cópia de training.csv 
         |-- 04afb96060aad90176268345e10355 
   |-- gitignore 
   |-- config --> mantém informações sobre armazenamento remoto (veja a próxima seção)
```

**Armazenar artefatos de ML com versão em armazenamento remoto**

O DVC suporta vários armazenamentos em nuvem que nos permitem armazenar nosso conjunto de dados ou modelos em armazenamento em nuvem remoto (por exemplo, S3, google drive). 
Para fazer isso, primeiro precisamos configurar o armazenamento remoto usando DVC e, em seguida, enviar os artefatos de ML para esse armazenamento remoto. 
Ao fazer isso, não precisamos manter o grande conjunto de dados em nosso repositório Git, o arquivo *.dvc leve e legível por humanos que contém o link para nosso conjunto de dados real será mantido no repositório Git.
Para mais informações, consulte a [documentação oficial](https://dvc.org/doc/command-reference/remote/add).

Configurar armazenamento remoto usando dvc (neste exemplo, usamos S3).
```
dvc remote add -d [storage_name] s3://[bucket]/[dvc_storage]
```

Carregar conjunto de dados para armazenamento remoto S3. 
```
dvc push
```

Depois de executar os comandos acima, um controle remoto será adicionado à configuração do DVC, que se parece com:
```
['remote "storage_name"'] 
url = s3://[bucket]/[dvc_storage] 
[core] 
remote = storage_name
```

**Recuperando artefatos de ML**

Tendo dados rastreados por DVC armazenados remotamente, eles podem ser baixados para o repositório Git quando necessário. 
Como o arquivo DVC armazenado no repositório Git contém o hash para identificar exclusivamente os dados e as informações de armazenamento remoto são armazenadas na configuração do DVC, ele sabe onde encontrar os dados no armazenamento remoto e baixar os dados para o projeto local.

Usando dvc pull para baixar dados
```commandline
dvc pull
```

**Fazendo alterações no conjunto de dados**

Ao fazer alterações no conjunto de dados localmente, o comando DVC add nos permite rastrear a versão mais recente do conjunto de dados. 
Ele atualizará o hash md5 dentro do arquivo DVC e, em seguida, a nova versão do conjunto de dados será vinculada ao projeto. 
Basicamente, ele segue as mesmas etapas dos artefatos de Versionamento de ML.

Usando dvc add para rastrear a nova versão do dataset
```commandline
dvc add data/training.csv
```

Empurre a nova versão para armazenamento remoto
```commandline
dvc push
```

Grava o estágio para a nova versão do dataset 
```commandline
git commit -m "conjunto de treinamento atualizado"
```

**Alternar entre as versões**
Quando queremos reverter para uma determinada versão do conjunto de dados, o git checkout nos ajudará a fazer o checkout de um commit ou uma revisão do arquivo DVC e, em seguida, usar o dvc checkout para sincronizar os dados.

Usando git checkout para finalizar um commit que você quer
```
git checkout <...>
```

Usando dvc checkout para sincronizar dados
```commandline
dvc checkout
```

## DVC PipeLines
**O DVC oferece suporte à criação de pipelines de ML**

O DVC também permite que você construa seus pipelines de ML, ele fornece um arquivo dvc.yaml que descreve cada etapa de um pipeline no projeto de ML e pode ser gerado manualmente ou pelo comando DVC. 
Outro arquivo dvc.lock é semelhante ao arquivo .dvc que mencionamos anteriormente em Controle de versão de artefatos de ML, que contém informações de hash md5 relacionadas a artefatos de ML. 
Isso nos permite executar e reproduzir facilmente qualquer estágio em seu pipeline de ML (por exemplo, estágio de treinamento, estágio de validação).

Primeiro, você precisará do DVC instalado, com versão de dados e armazenamento remoto adicionado. 
Em seguida, o fluxo de trabalho de exemplo de estabelecimento do pipeline de ML a seguir é descrito abaixo.

[//]: # (TODO: Figura 2. pipelines de ML)

Execute [`dvc run`](!https://dvc.org/doc/command-reference/run)  para criar estágios. Os estágios são as etapas exclusivas de um pipeline e podem ser rastreados via git. 
Os estágios também conectam o código à sua entrada e saída de dados (semelhante ao Snakemake). Os estágios com todas as dependências e parâmetros correspondentes são gravados em um arquivo de pipeline especial chamado dvc.yaml.
Esses são os principais parâmetros:
```
-n <stage> : especifica um nome para o estágio gerado por este comando
-p [<path>:]<params_list> : especifica um conjunto de dependências de parâmetro das quais o estágio depende
-d <path> : especifica um arquivo ou diretório do qual o estágio depende
-o <path> : especifica um arquivo ou diretório que é o resultado da execução do comando
-m <path> : especifica um arquivo de métricas produzido por este estágio. Esta opção se comporta como -o , mas registra o arquivo em um campo de métricas dentro do estágio dvc.yaml
--plots <path>:  especifica um arquivo de métricas de plotagem produzido por este estágio. Esta opção se comporta como -o , mas registra o arquivo em um campo de plotagens dentro do estágio dvc.yaml

```
Especifique o nome do estágio (-n), entrada (-d), saída (-o) para criar o estágio de treinamento:


```
dvc run -n train \
  -d src/train.py \
  -d data/training.csv \ 
  -o model/model.pkl \ 
  python src/train.py
```

Faça o mesmo para o estágio de validação
```
dvc run -n validate \ 
  -d src/validate.py \
  -d data/validation.csv \ 
  -M metric/accuracy.json \ 
  --plots-no-cachemetrics/matrix.png \ 
  python src/ valid.py
```
-M grava métricas em um destino de saída especificado


Depois de executados com sucesso os comandos acima, um arquivo dvc.yaml é gerado e contém esses dois estágios. 
Assim, o arquivo dvc.lock é apresentado e contém vários hashes e cada hash identifica exclusivamente um arquivo.

**dvc.yaml**
```
stages:
  train:
    cmd: python src/train.py 
    deps:
    - data/training.csv
    - src/train.py
    outs:
    - model/model.pkl
  validate:
    cmd: python src/validate.py 
    deps:
    - data/validation.csv
    - src/validate.py
    metrics:
    - metrics/accuracy.json:
        cache: false
    plots:
    - metrics/confusion_matrix.png:
        cache: false
```

**dvc.lock**
```
train:
  cmd: python src/train.py
  deps:
  - path: data/training.csv
    md5: a304afb96060aad90176268345e10355
    size: 123407
  - path: src/train.py
    md5: 094631814e0233a8c189b5f5dd214f61
    size: 4445
  outs:
  - path: model/model.pkl
    md5: 1b320cd680be56cbbf71eb649ff29d25
    size: 704750
validate:
  cmd: python src/validate.py
  deps:
  - path: data/validation.csv
    md5: e57e5c9144f8eb84563b692d41c80f46
    size: 50112
  - path: src/validate.py
    md5: cd59a289500b05233548fc48fa0f13dd
    size: 4876
  outs:
  - path: metrics/confusion_matrix.png
    md5: 8149ea4e39ea587930d9d16c75d80d8f
    size: 15926
  - path: metrics/accuracy.json
    md5: 0be41efa33b74e9cc33f1e3cf12d4f69
    size: 126
```

Agora você tem 2 estágios em seu pipeline de ML. 
Você pode usar os comandos `git add` e `git commit` para permitir que a versão do git controle este estado. 
No futuro, se você mudou alguma coisa em seu projeto (por exemplo, dados), você pode primeiro seguir os artefatos de Versioning ML antes de permitir que o dvc rastreie a nova versão dos dados e, em seguida, você pode facilmente executar novamente o estágio de treinamento ou validação pelo comando DVC, como `dvc repro` para repetir essa etapa. 
Se você deseja reproduzir um estado em um ponto no tempo, pode alternar entre versões antes para sincronizar seu projeto local com esse estado específico e, em seguida, alterar o que quiser e, finalmente, usar `dvc repro STAGE_NAME` para executar novamente rapidamente determinados estágios em seu próprio pipeline de ML.

------

## Criando uma pipeline na prática

Vamos construir um modelo para classificar o conjunto de dados iris. 

O script será dividido nas famosas etapas de Machine Learning:
1. Coletae de dados
2. Gerar os recursos
3. Treinar o modelo
4. Avaliar o modelo

Para cada uma das etapas, vamos precisar:
- Escreva um script python
- Salve os parâmetros que cada script usa em um arquivo `params.yaml`
- Especifique os arquivos dos quais o script depende
- Especifique os arquivos que o script gera

A estrutura do projeto será:
```
tutorial_dvc/ 
|-- data/ -> Diretório onde salvaremos nossos datasets
|-- metrics/  -> Diretório onde salvaremos o resultado das metricas
|-- model/  -> Diretório onde salvaremos nosso modelo
|-- src/  -> Diretório onde salvaremos os scripts de cada etapa
```

### Iniciando o projeto
Vamos chamar o nosso projeto de `tutorial_dvc`.

O Primeiro passo é inicializar o git e dvc no diretório raiz do repositório:
```
cd tutorial_dvc
git init
dvc init
```

Na sequencia, precisamos adicionar o armazenamento remoto a ser usado (exemplo s3):
```
dvc remote add -d remote s3://bucket_name/project_name
dvc remote modify remote region us-east-1
```

Agora vamos implementar cada etapa.

### 1. Coleta de dados
Estamos usando o método load_iris() do Scikit para carregar dados na memória, mas vamos salvá-lo em um arquivo para poder usá-lo como uma dependência para a próxima etapa do pipeline. 
Então, nesta etapa, vamos reunir os dados e salvá-los em um arquivo csv (salvaremos em `data/prepared`).
Vamos definir o nome para este estágio como `prepare`.

O DVC usa um arquivo chamado params.yam como arquivo de parâmetros padrão, então vamos criar um e definir os paramêtros lá:

**arquivo params.yaml**
```
prepare:
    test_size: 0.33
    random_state: 42
```
`prepare` é o nome da etapa, `test_size` e `random_state`  são os nomes dos parâmetros que vamos definir para essa etapa.

Aqui está o arquivo final `prepare.py`, que salvaremos dentro da pasta `src`:
```
import os
import yaml
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

pipeline_path = os.path.abspath(os.path.join(os.path.join(__file__, os.pardir), os.pardir))

# read params
params = yaml.safe_load(open(os.path.join(pipeline_path, 'params.yaml')))['prepare']

# create folder to save file
data_path = os.path.join(pipeline_path, 'data', 'prepared')
os.makedirs(data_path, exist_ok=True)

# fetch data
data = load_iris()

# split
X_train, X_test, y_train, y_test = train_test_split(
    data.data,
    [data.target_names[i] for i in data.target],
    test_size=params['test_size'],
    random_state=params['random_state']
)

# save
df_train = pd.DataFrame(np.concatenate((X_train,  np.expand_dims(y_train, 1)), axis=1), columns=data.feature_names + ['target'])
df_test = pd.DataFrame(np.concatenate((X_test,  np.expand_dims(y_test, 1)), axis=1), columns=data.feature_names + ['target'])

df_train.to_csv(os.path.join(data_path, 'train.csv'))
df_test.to_csv(os.path.join(data_path, 'test.csv'))
```

Esta etapa dependemos apenas do arquivo de código do script `prepare.py`, e irá gerar os CSVs na pasta `data/prepared`.

#### dvc run — Construindo estágios usando DVC

Agora podemos contruir nosso __step__ utilizando o DVC. O DVC salva os __steps__ do pipeline em um arquivo dvc.yaml (legível por humanos) e um dvc.lock(isso é apenas para uso do DVC). 
Para criar um estágio de pipeline, usamos o comando `dvc run`. 

Depois de definir essas opções, adicionamos um argumento de comando onde especificamos como realmente executar esta etapa do pipeline. 
Nesta etapa o comando será `python3 src/prepare.py`.
```
dvc run -n prepare -p prepare.test_size,prepare.random_state -d src/all_pipeline.py -o data/prepared python3 src/all_pipeline.py
```

A estrutura do projeto ficará assim:
```
tutorial_dvc/ 
|-- data/
    |-- prepare/
        |-- train.csv
        |-- test.csv
|-- metrics/
|-- model/
|-- src 
    |-- prepare.py 
|-- params.yaml   
|-- dvc.lock
|-- dvc.yaml
```

E estes são os conteúdos do arquivo dvc.yaml gerado automaticamente pelo DVC:
```
stages:
  prepare:
    cmd: python3 src/all_pipeline.py
    deps:
    - src/all_pipeline.py
    params:
    - prepare.random_state
    - prepare.test_size
    outs:
    - data/prepared
```

#### dvc dag  — Visualize o pipeline usando DVC
O comando exibe os estágios de um pipeline. 
Temos apenas um estágio até agora, mas vamos ver:

```
dvc dag 
 +---------+      
 | prepare |      
 +---------+      
      *           
      *   
```

#### dvc repro  — Reproduzindo os pipelines usando DVC
O comando reproduz pipelines completos ou parciais executando comandos definidos em seus estágios. 

Vamos testar então (ainda não fiz nenhuma alteração):

```
dvc reprodução
O estágio 'preparar' não foi alterado, ignorando                                          
Dados e pipelines atualizados.
```

E se eu mudar algum parametro no `params.yaml`e depois execute o comando novamente?

```
dvc reprodução
Executando o estágio 'prepare' com o comando:                                            
        python3 src/prepare.py 
Atualizando o arquivo de bloqueio 'dvc.lock'
Para rastrear as alterações com o git, execute:
git add dvc.lock

```
Ele executa o estágio novamente porque adicionamos o parâmetro alterado como parâmetro para este estágio. 
O DVC então viu que alteramos esse parâmetro e executamos o estágio novamente.
O arquivo dvc.yaml ainda é o mesmo, mas se você verificar o dvc.lock, verá que os parâmetros mudaram lá.

### 2. Gerar os recursos
Nesta etapa, salvaremos a features em arquivos pickle dentro da pasta `data/features`. 
Vamos definir o nome para este estágio como `featurize`.

Aqui está o arquivo final `featurize.py`, que salvaremos dentro da pasta `src`:
```
import sys
import os
import pandas as pd
import pickle

# read command line params
if len(sys.argv) != 3:
    sys.stderr.write('Arguments error. Usage:\n')
    sys.stderr.write(
        '\tpython featurize.py data-dir-path features-dir-path\n'
    )
    sys.exit(1)

data_path = sys.argv[1]
features_path = sys.argv[2]

os.makedirs(features_path, exist_ok=True)

# read the data from file
df_train = pd.read_csv(os.path.join(data_path, 'train.csv'))
df_test = pd.read_csv(os.path.join(data_path, 'test.csv'))


def encode_targets(df_1, df_2):
    targets_train = list(df_1["target"].unique())
    targets_test = list(df_2["target"].unique())

    unique_targets = list(sorted(set(targets_train + targets_test)))
    class_to_id = {
        l: i for i, l in enumerate(unique_targets)
    }

    df_1["target"] = [class_to_id[l] for l in df_1["target"]]
    df_2["target"] = [class_to_id[l] for l in df_2["target"]]


def save_pkl(df, filename):
    output_file = os.path.join(features_path, filename)
    output = df.values

    with open(output_file, 'wb') as f:
        pickle.dump(output, f)


# we need to encode both train and test target
encode_targets(df_train, df_test)

# save data to pickle (appending the labels column)
save_pkl(df_train, 'train.pkl')
save_pkl(df_test, 'test.pkl')
```

Vamos depender do arquivo de script e da pasta data/prepared. Adicionamos os parametros no arquivo `params.yaml` e executamos o `dvc run`:
```
dvc run -n featurize -d src/featurize.py -d data/prepared -o data/features python3 src/featurize.py data/prepared data/features
```

### 3. Treinar o modelo
Nesta etapa, vamos finalmente treinar o modelo e salvá-lo em um arquivo pickle dentro da pasta `models. 
Vamos definir o nome para este estágio como `train`.

Aqui está o arquivo final `train.py`, que salvaremos dentro da pasta `src`:
```
import sys
import os
import yaml
import importlib
import pickle

# read the command line params
if len(sys.argv) != 3:
    sys.stderr.write('Arguments error. Usage:\n')
    sys.stderr.write(
        '\tpython3 train.py features-dir-path model-filename\n'
    )
    sys.exit(1)

features_path = sys.argv[1]
model_filename = sys.argv[2]

pipeline_path = os.path.abspath(os.path.join(os.path.join(__file__, os.pardir), os.pardir))

# read pipeline params
params = yaml.safe_load(open(os.path.join(pipeline_path, 'params.yaml')))['train']


# load the train features
features_train_pkl = os.path.join(features_path, 'train.pkl')
with open(features_train_pkl, 'rb') as f:
    train_data = pickle.load(f)

X = train_data[:, :-1]
y = train_data[:, -1]

# train the model
model = getattr(
    importlib.import_module(params['import_module']),
    params['name']
)(**params['params'])
model.fit(X, y)

# save the model
with open(model_filename, 'wb') as f:
    pickle.dump(model, f)
```

Vamos depender do arquivo de script e da pasta data/features. Adicionamos os parametros no arquivo `params.yaml` e executamos o `dvc run`:
```
dvc run -n train -p train.import_module,train.name,train.params.n_estimators,train.params.n_jobs,train.params.random_state -d src/train.py -d data/features -o models/model.pkl python3 src/train.py data/features models/model.pkl
```

### 4. Avaliar o modelo


Nesta etapa, vamos avaliar o modelo e salvar as métricas em `metrics`. 
Vamos definir o nome para este estágio como `evaluate`.

Aqui está o arquivo final `evaluate.py`, que salvaremos dentro da pasta `src`:
```
import sys
import os

import pandas as pd
from sklearn.metrics import (
    get_scorer,
    SCORERS,
)
import pickle
import json
import yaml
import numpy as np

# read command line parameters
if len(sys.argv) != 5:
    sys.stderr.write('Arguments error. Usage:\n')
    sys.stderr.write(
        '\tpython3 evaluate.py model-filename features-dir-path scores-filename\
                plots-filename\n'
    )
    sys.exit(1)

model_filename = sys.argv[1]
features_path = sys.argv[2]
test_features_file = os.path.join(os.path.join(features_path, 'test.pkl'))
scores_file = sys.argv[3]
plots_file = sys.argv[4]

pipeline_path = os.path.abspath(os.path.join(os.path.join(__file__, os.pardir), os.pardir))

# read pipeline params
params = yaml.safe_load(open(os.path.join(pipeline_path, 'params.yaml')))['evaluate']

# load features
with open(test_features_file, 'rb') as f:
    test_features = pickle.load(f)

X_test = test_features[:, :-1]
y_test = test_features[:, -1]

# load model
with open(model_filename, 'rb') as f:
    model = pickle.load(f)

# make predictions
predictions_by_class = model.predict_proba(X_test)
predictions = np.argmax(predictions_by_class, axis=1)

# generate scores
metrics = {}
params['metrics'] = [f for f in params['metrics'] if f in list(SCORERS.keys())]
for m in params['metrics']:
    m_result = get_scorer(m)(model, X_test, y_test)
    metrics[m] = float(m_result)

# save scores
with open(scores_file, 'w') as f:
    json.dump(metrics, f)

pd.DataFrame({'predicted': list(predictions), 'real': list(y_test)}).to_csv(plots_file, index=False)
```

Vamos depender do arquivo de script e das pastas data/features e model. Adicionamos os parametros no arquivo `params.yaml` e executamos o `dvc run`:
```
dvc run -n evaluate -p evaluate.metrics -d src/evaluate.py -d models/model.pkl -d data/features --metrics-no-cache metrics/scores.json python3 src/evaluate.py models/model.pkl data/features metrics/scores.json metrics/cm_classes.csv
```

### Estrutura final do projeto

A estrutura final do projeto ficará assim:
```
tutorial_dvc/ 
|-- data/
    |-- prepare/
        |-- train.csv
        |-- test.csv
    |-- fetures/
        |-- train.pkl
        |-- test.pkl
|-- metrics/
    |-- cm_classes.csv
    |-- scores.json
|-- model/
    |-- model.pkl
|-- src 
    |-- prepare.py 
    |-- featurize.py 
    |-- train.py 
    |-- evaluate.py 
|-- params.yaml   
|-- dvc.lock
|-- dvc.yaml
```

**dvc.yaml**
```
stages:
  prepare:
    cmd: python3 src/prepare.py
    deps:
    - src/prepare.py
    params:
    - prepare.random_state
    - prepare.test_size
    outs:
    - data/prepared
  featurize:
    cmd: python3 src/featurize.py data/prepared data/features
    deps:
    - data/prepared
    - src/featurize.py
    outs:
    - data/features
  train:
    cmd: python3 src/train.py data/features models/model.pkl
    deps:
    - data/features
    - src/train.py
    params:
    - train.import_module
    - train.name
    - train.params.n_estimators
    - train.params.n_jobs
    - train.params.random_state
    outs:
    - models/model.pkl
  evaluate:
    cmd: python3 src/evaluate.py models/model.pkl data/features scores.json cm_classes.csv
    deps:
    - data/features
    - models/model.pkl
    - src/evaluate.py
    params:
    - evaluate.metrics
    metrics:
    - scores.json:
        cache: false
```

**params.yaml**
```
prepare:
    test_size: 0.33
    random_state: 42
train:
  import_module: sklearn.ensemble
  name: RandomForestClassifier
  params:
    n_estimators: 10
    n_jobs: -1
    random_state: 42
evaluate:
  metrics:
    - accuracy
    - balanced_accuracy
    - adjusted_rand_score
```

------

## Referências:
### DVC
- [MLOps: Data versioning with DVC — Part Ⅰ](https://yizhenzhao.medium.com/mlops-data-versioning-with-dvc-part-%E2%85%B0-8b3221df8592)
- [The ultimate guide to building maintainable Machine Learning pipelines using DVC](https://towardsdatascience.com/the-ultimate-guide-to-building-maintainable-machine-learning-pipelines-using-dvc-a976907b2a1b)
- [Tracking ML Experiments With Data Version Control](https://www.analyticsvidhya.com/blog/2021/06/mlops-tracking-ml-experiments-with-data-version-control/)
- [Creating Data Science Pipelines using DVC](https://blog.koverhoop.com/creating-datascience-pipelines-using-dvc-ea7d934fafac)
- [Versioning Datasets with Git & DVC](https://www.analyticsvidhya.com/blog/2021/06/mlops-versioning-datasets-with-git-dvc/)
- [mlops-guide.github.io](https://github.com/mlops-guide/mlops-guide.github.io/blob/main/docs/)
- [End to End Machine Learning Pipeline With MLOps Tools (MLFlow+DVC+Flask+Heroku+EvidentlyAI+Github Actions)](https://medium.com/@shanakachathuranga/end-to-end-machine-learning-pipeline-with-mlops-tools-mlflow-dvc-flask-heroku-evidentlyai-github-c38b5233778c)
- [https://madewithml.com/#mlops](https://madewithml.com/#mlops)
- [https://github.com/GokuMohandas/MLOps](https://github.com/GokuMohandas/MLOps)
- [https://github.com/shanakaChathu/churn_model](https://github.com/shanakaChathu/churn_model)
- [https://github.com/hzdr/mlops_comparison](https://github.com/hzdr/mlops_comparison)

### DVC + CML
- [CML](https://github.com/iterative/cml)
- [CML with DVC use case](https://github.com/iterative/cml_dvc_case)
- [DVC CI/CD MLOps Pipeline](https://github.com/mlops-guide/dvc-gitactions)
