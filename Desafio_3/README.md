# Funções de probabilidade

Neste desafio vamos praticar nossos conhecimentos em probabilidade e estatística,
conhecimentos fundamentais para qualquer cientista de dados.

## Objetivo

O objetivo deste desafio é explorar as principais funções sobre distribuições de probabilidade
como PDF, CDF e quantis e as relações entre duas das principais distribuições: a normal e a binomial.

Para isso, utilizaremos dados artificiais e reais. Como dados reais, exploraremos o _data set_
[Pulsar Star](https://archive.ics.uci.edu/ml/datasets/HTRU2) disponibilizado pelo Dr. Robert Lyon da Universidade de Manchester.

Esse _data set_ consiste de 8 variáveis a respeito de 17898 observações de estrelas. Essas
estrelas foram consideradas "candidatas" a serem estrelas do tipo pulsar, que têm forte 
importância para os astrofísicos. Uma nona coluna do _data set_ especifica se a estrela é
realmente um pulsar (caso positivo, 1) ou não (caso negativo, 0).

## Tópicos

Neste desafios nós vamos explorar:

* Probabilidade
* Estatística
* NumPy
* SciPy
* StatsModels

## Requisitos

Você precisará de Python 3 e pip. É altamente recomendado utilizar ambientes virtuais
com o virtualenv e o arquivo `requirements.txt` para instalar os pacotes dependências
do desafio:

```bash
$ pip3 install virtualenv
$ virtualenv venv -p python3
$ source venv/bin/activate
$ pip install -r requirements.txt
```

Windows

```bash
> pip3 install virtualenv
> virtualenv ..\venv -p python3
> ..\venv\Scripts\activate
> pip install -r requirements.txt
```

Quando finalizado, você pode desativar o ambiente virtual do virtualenv com:

```bash
$ deactivate
```
