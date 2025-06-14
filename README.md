# vis-clothes
Aplicação web que recebe a foto de uma roupa e retorna sua classificação de cor, tipo e estampa. O dataset utilizado é o [Vibrent Clothes Rental Dataset](https://www.kaggle.com/datasets/kaborg15/vibrent-clothes-rental-dataset?resource=download&select=outfits.csv).


## Estrutura de pastas
- Raiz do repositório 
    - Arquivos com o treinamento e teste dos modelos de cor e categorias (*categories_model* e *train_color_model*)
    - Arquivos utilizados para retirar o fundo das imagens (*image_segmentation* e *segmentation*)
    - Arquivo de dependências *requirements.txt*.
- *csvs_datasets*
    - Devido ao desbalanceamento da quantidade de roupas em categorias de estampa, foram testadas 3 diferentes abordagens de amostragem. Essa pasta contém arquivos csv considerando, para estampa, 2 categorias (solid e non-solid), 3 categorias (solid, non-solid e geometric) e 6 categorias (solid, floral, pattern, stripes, checkers e animal print).
- *raw_data*
    - Arquivos *outfits* e *picture_triplets* vêm do próprio *dataset*.
    - Arquivo *parsed_dataset* tem as imagens do dataset original, mas alguma colunas foram tratadas para retirar listas de valores.
    - Arquivo *cleaned_dataset* tem apenas as colunas e valores que são utilizadas nos modelos, imagens que não tinham valores correspondentes nas categorias analisadas pelos modelos foram retiradas. A categoria *Solid* de estampa foi criada analisando as imagens manualmente.
- *app*
    - O arquivo *main* contém a configuração da API feita com fastAPI. O método post recebe um zip com imagens a serem analisadas e retorna os resultados da classificação.
    - O arquivo *clothes_models* carrega os modelos de classificação de cor, estampa e categoria e gera o resultado com as 3 informações para as imagens recebidas na função *apply_cloth_model*.
    - O arquivo *segmentation* tem funções utilizadas no pré-processamento das imagens recebidas para classificação, esse pré-processamento inclui alterar o tamanho da imagem para o utilizado nos modelos e retirar o fundo das imagens para gerar melhores resultados.
- *models*
    - Modelos de cor, estampa e categoria treinados. Os modelos finais utilizados foram *jit_model_color_66*, *jit_model_detail_6* e *model_categories*.

## Modelos

Todos os modelos foram treinados após retirar o fundo das imagens para melhorar os resultados. Além disso, a medida de acurácia foi feita com um conjunto de imagens balanceados por categoria em todos os modelos.

### Classificação de cores
- Treinamento e teste feitos no arquivo *train_color_model.ipynb*.
- Acurácia: 67%

Para treino e teste do modelo foi utilizado o *cleaned_dataset* que contém 37680 imagens. Após balancear as categorias, com valores variando entre 1000 e 1500 por categoria, restaram 18632 imagens. Esse valor foi dividido na proporção 80% para treinamento e 20% para teste.

### Classificação de estampas
- Treinamento e teste feitos no arquivo *model_torch.ipynb* na pasta *detail_model*.
- Acurácia: 79%

Para treinar o modelo, foi utilizado o CSV para 6 categorias de estampa (*6_details_categories*) que contém 6845 imagens, porém, os três arquivos da pasta *csvs_datasets* foram utilizados no notebook para comparação das métricas para cada modelo.

### Classificação de categorias
- Treinamento e teste feitos no aquivo *categories_model.ipynb*.
- Acurácia: 63%

Para treino e teste do modelo foi utilizado o *cleaned_dataset*. O dataset foi dividido em uma proporção de 80% para treinamento e 20% para teste. O conjunto de teste teve as categorias balanceadas para cerca de 400 cada.

### Trocando um modelo no backend

É possível trocar o modelo utilizado pela aplicação alterando o arquivo *clothes_models* que se encontra na pasta *app*.

Com o modelo salvo na pasta *models*, altere o nome do arquivo passado na função *torch.jit.load* da variável correspondente (*color_model*, *detail_model* ou *category_model*).

As variáveis *IMG_SIZE*, *COLOR_LABELS*, *CATEGORY_LABELS*, *DETAILS_LABELS*, *color_transform*, *details_transform* e *category_transform*, estão alinhadas com as categorias e formato de imagem utilizadas no treinamento dos modelos. Pode ser necessário atualizar algumas dessas variáveis ao trocar um ou mais modelos utilizados.

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

### Rodando o Frontend

- O repositório do frontend da aplicação está disponível [aqui](https://github.com/ellencoutinho/vis-clothes-front). 

- Após clonar o repositório sigas as instruções do README.

- Seguindo as instruções a página estará disponível em http://localhost:5173.

## Observações

- O zip submetido na página deve ser um **zip com imagens** (um **zip com uma pasta de imagens** por exemplo não será aceito)
