# vis-clothes
Aplicação web que recebe a foto de uma roupa e retorna sua classificação de cor, tipo e estampa. O dataset utilizado é o [Vibrent Clothes Rental Dataset](https://www.kaggle.com/datasets/kaborg15/vibrent-clothes-rental-dataset?resource=download&select=outfits.csv).


## Estrutura de pastas
- *csvs_datasets*
    - Devido ao desbalanceamento da quantidade de roupas em categorias de estampa, foram testadas 3 diferentes abordagens de amostragem. Essa pasta contém arquivos csv considerando, para estampa, 2 categorias (solid e non-solid), 3 categorias (solid, non-solid e geometric) e 6 categorias (solid, floral, pattern, stripes, checkers e animal print).
- *raw_data*
    - Arquivos *outfits* e *picture_triplets* vêm do próprio *dataset*.
    - Arquivo *parsed_dataset* contém a maior amostra de imagens do dataset, filtrando as categorias de estampa e categorias a serem usadas.
- *app*
    - O arquivo *main* contém a configuração da API feita com fastAPI. O método post recebe um zip com imagens a serem analisadas e retorna os resultados da classificação.
    - O arquivo *clothes_models* carrega os modelos de classificação de cor, estampa e categoria e gera o resultado com as 3 informações para as imagens recebidas na função *apply_cloth_model*.
    - O arquivo *segmentation* tem funções utilizadas no pré-processamento das imagens recebidas para classificação, esse pré-processamento inclui alterar o tamanho da imagem para o utilizado nos modelos e retirar o fundo das imagens para gerar melhores resultados.
- *models*
    - Modelos de cor, estampa e categoria treinados. Os modelos finais utilizados foram *jit_model_color_66*, *jit_model_detail_6* e *model_categories*.

## Modelos

### Classificação de cores
- Treinamento e teste feitos no arquivo *train_color_model.ipynb*
- Acurácia: 67%

## Classificação de estampas
- Treinamento e teste feitos no arquivo **
- Acurácia: 

### Classificação de categorias
- Treinamento e teste feitos no aquivo *categories_model.ipynb*.
- Acurácia: 63%

## Acessando e utilizando a página

### Rodando o backend

- Na raiz desse repositório há o arquivo *requirements.txt*, contendo todas as dependências, para instalar rode o comando:

```
pip install -r requirements.txt
```
- Para executar a API, rode o comando:

```
fastapi run app/main.py
```
- O backend estará disponível em http://localhost:8000

## Rodando o Frontend

- O repositório do frontend da aplicação está disponível [aqui](https://github.com/ellencoutinho/vis-clothes-front). 

- As instruções para rodar a aplicação está no README do repositório.

- Seguindo as instruções a página estará disponível em http://localhost:5173.

## Observações

- O zip submetido na página deve ser um **zip com imagens** (um **zip com uma pasta de imagens** por exemplo não será aceito)
