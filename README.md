# vis-clothes
Aplicação web que recebe a foto de uma roupa e retorna sua classificação de cor, tipo e estampa. O dataset utilizado é o [Vibrent Clothes Rental Dataset](https://www.kaggle.com/datasets/kaborg15/vibrent-clothes-rental-dataset?resource=download&select=outfits.csv).


## Estrutura de pastas
- *csvs_datasets*
    - Devido ao desbalanceamento da quantidade de roupas em categorias de estampa, foram testadas 3 diferentes abordagens de amostragem. Essa pasta contém arquivos csv considerando, para estampa, 2 categorias (solid e non-solid), 3 categorias (solid, non-solid e geometric) e 6 categorias (solid, floral, pattern, stripes, checkers e animal print).
- *raw_data*
    - Arquivo *initial_filtered_clothes* contém a maior amostra de imagens do dataset, filtrando as categorias de estampa a serem usadas.
    - Arquivos *outfits* e *picture_triplets* vêm do próprio *dataset*.