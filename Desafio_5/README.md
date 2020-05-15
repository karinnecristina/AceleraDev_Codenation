# Redução de dimensionalidade e seleção de

Neste desafio vamos praticar redução de dimensionalidade com PCA e seleção
de variáveis com RFE.

## Objetivo

O objetivo deste desafio é explorar sobre como funciona o PCA e como podemos
obter _data sets_ de dimensões mais baixas através dele.

Para isso, vamos contar com o _data set_ [FIFA 2019](https://www.kaggle.com/karangadiya/fifa19)
que contém originalmente 89 variáveis com diversos atributos de mais de 18 mil jogadores
do _game_ FIFA 2019.

## Tópicos

Neste desafios nós vamos explorar:

* Redução de dimensionalidade
* PCA
* Seleção de variáveis
* RFE

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
