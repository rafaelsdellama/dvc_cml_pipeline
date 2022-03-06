# DVC and CML Pipelines

## Instalação

```console
$ git clone 
$ cd dvc_cml_pipeline
```

Crie uma [virtualenv](https://virtualenv.pypa.io/en/stable/) e instale o requirements da pipeline que deseja executar:

```console
$ virtualenv -p python3 venv
$ source venv/bin/activate
$ cd pipelines/PIPELINE
$ pip install -r src/requirements.txt
```

# Referências:
- [The ultimate guide to building maintainable Machine Learning pipelines using DVC](https://towardsdatascience.com/the-ultimate-guide-to-building-maintainable-machine-learning-pipelines-using-dvc-a976907b2a1b)
- [Creating Data Science Pipelines using DVC](https://blog.koverhoop.com/creating-datascience-pipelines-using-dvc-ea7d934fafac)
- [CML](https://github.com/iterative/cml#using-cml-with-dvc)