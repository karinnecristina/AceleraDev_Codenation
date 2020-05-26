# Feature engineering

Neste desafio vamos praticar _feature engineering_, a arte de processar
variáveis do _data set_ a fim de torná-las mais adequadas aos algoritmos
de ML e produzir melhores resultados.

## Objetivo

O objetivo deste desafio é adquirir conhecimento e prática nas ferramentas
mais usuais de engenharia de variáveis. Com o domínio apropriado das
técnicas básicas, como _one-hot encoding_, normalização e padroniação,
o analista está mais bem preparado para conduzir uma etapa de preprocessamento
dos dados que traga bons resultados da aplicação dos algoritmos de ML.

Para isso, vamos contar com o _data set_ [Countries of the world](https://www.kaggle.com/fernandol/countries-of-the-world)
que contém 20 variáveis, como população, área costeira e tamanho dos setores de produção, de 227 países.

## Tópicos

Neste desafios nós vamos explorar:

* Feature engineering
* Processamento de texto

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
