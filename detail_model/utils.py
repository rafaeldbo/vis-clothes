import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
import pandas as pd
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import os
import shutil
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import LabelBinarizer
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

device = torch.device("cpu")

class CNNModel(nn.Module):
    """Classe destinada a criação do modelo de CNN a ser treinado e testado.
    """
    def __init__(self, num_categories):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3)
        self.fc1 = nn.Linear(32 * 14 * 14, 128)  # ajustado para entrada 64x64
        self.fc2 = nn.Linear(128, num_categories)

    def forward(self, x):
        x = F.relu(self.conv1(x))      # Conv1 + ReLU
        x = self.pool(x)               # MaxPool1
        x = F.relu(self.conv2(x))      # Conv2 + ReLU
        x = self.pool(x)               # MaxPool2
        x = x.view(-1, 32 * 14 * 14)   # Flatten
        x = F.relu(self.fc1(x))        # Dense1 + ReLU
        x = self.fc2(x)                # Dense2 (Logits)
        return x

def dataframe_create(csv_files_path, csv_filename, src_image_path) -> pd.DataFrame:
    """Retorna um `DataFrame` (`pandas`) contendo a relação do nome 
    dos arquivos que existem em um diretório com a classificação deles
    de acordo com o que é especificado no arquivo CSV de entrada

    Args:
        `csv_files_path` (str): Caminho onde os arquivos CSV criados estão localizados (csv_files, por exemplo).
        `csv_filename` (str): Nome do arquivo CSV contendo as definições das imagens de acordo com as categorias a serem analisadas.
        `src_image_path` (str): Caminho de onde as imagens serão retiradas.

    Raises:
        `Exception`: Se o arquivo CSV não for encontrado.

    Returns:
        `pd.DataFrame`: DataFrame contendo o nome das imagens e a respectiva classe à qual ela está associada.
    """
    
    if os.path.isfile(os.path.join(csv_files_path, csv_filename)):
        df = pd.read_csv(os.path.join(csv_files_path, csv_filename))[["file_name", "Details"]]
    else:
        raise Exception(f"File: '{os.path.join(csv_files_path, csv_filename)}' not found.")
    
    lista = []

    for _, row in df.iterrows():
        if os.path.isfile(f"./{src_image_path}/{row['file_name']}"):
            lista.append(row)

    return pd.DataFrame(lista)

def dataset_categories_create(df: pd.DataFrame, dataset_path):
    """Cria as categorias (subpastas) dado um DataFrame que contém as colunas `file_name` e `Details`.

    Args:
        `df` (pd.DataFrame): DataFrame que contém as imagens e categorias
        `dataset_path` (str): Caminho no qual as subpastas serão criadas
    """
    
    categories = [category.lower().replace("-", "_") for category in list(df.Details.value_counts().index)]

    for category in categories:
        os.makedirs(os.path.join(dataset_path, category), exist_ok=True)

def dataset_create(df: pd.DataFrame, dataset_path: str, src_image_path: str):
    """Salva as imagens em `dataset_path` de acordo com sua respectiva categoria, que é dada
    no DataFrame `df`. As imagens devem vir de um diretório de origem (`src_image_path`)

    Args:
        `df` (pd.DataFrame): DataFrame que contém as imagens e categorias.
        `dataset_path` (str): Caminho da pasta onde as imagens serão inseridas.
        `src_image_path` (str): Caminho da pasta das imagens que serão copiadas (origem).
    """
    for _, row in df.iterrows():
        try:
            if not os.path.isfile(f"{dataset_path}/{row['Details'].lower().replace('-', '_')}/{row['file_name']}"):
                shutil.copy(f"./{src_image_path}/{row['file_name']}", f"{dataset_path}/{row['Details'].lower().replace('-', '_')}")
        except:
            pass

def train_model(num_categories: int, transform: Compose, epochs: int = 10, learning_rate: float = 0.001) -> tuple[DataLoader, DataLoader]:
    """Realiza o treinamento do modelo `CNNModel` e salva os pesos do treinamento em um arquivo *.pth* (por exemplo, model_weights_2, para 2 categorias). Retorna dois DataLoaders, um de treino e outro de teste, respectivamente.

    Args:
        num_categories (int): Número de categorias.
        transform (Compose): Objeto `Compose` (argumento para `ImageFolder`)
        epochs (int, optional): Quantidade de vezes que o modelo será treinado. O valor padrão é 10.
        learning_rate (float, optional): Argumento utilizado para o otimizador Adam. O valor padrão é 0.001.

    Returns:
        tuple[DataLoader,DataLoader]: DataLoaders que podem ser utilizados posteriormente.
    """
    flag_train = True

    model = CNNModel(num_categories)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    dataset = datasets.ImageFolder(root=f"dataset_{num_categories}", transform=transform)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_data, test_data = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    if os.path.exists(f"model_weights_{num_categories}.pth"):
        answer = input("Um arquivo com os pesos desse modelo já existe. Deseja treinar novamente? (y/n)")

        if answer.lower() == "y":
            flag_train = True

        elif answer.lower() == "n":
            flag_train = False

        while answer.lower() not in ["y", "n"]: 
            answer = input("Resposta inválida, digite (y) para sim, (n) para não. Deseja treinar novamente? (y/n)")

    if flag_train:
        for epoch in range(epochs):
            model.train()

            for images, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        
            print(f'Epoch {epoch + 1}: Loss = {loss.item()}')

        torch.save(model.state_dict(), f"model_weights_{num_categories}.pth") # Salva os pesos
        torch.save(model, f"model_complete_{num_categories}.pth") # Salva o modelo em si

    return train_loader, test_loader

def test_model(model: CNNModel, model_weights: str, test_loader: DataLoader):
    """Testa o modelo com um determinado `test_loader` e salva a precisão em um dicionário.

    Args:
        `model` (CNNModel): Modelo a ser testado.
        `model_weights` (str): Arquivo contendo os pesos do modelo treinado.
        `test_loader` (DataLoader): DataLoader com as imagens e categorias atribuídas à elas.

    Returns:
        `result` (dict): Dicionário com precisão do modelo.
    """

    model.load_state_dict(torch.load(model_weights))

    result = {}

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    result[model_weights] = 100 * correct / total

    return result

def test_model_full_metrics(model: CNNModel, model_weights_path: str, test_loader: DataLoader):
    """
    Testa o modelo com um determinado `test_loader` e calcula diversas métricas de classificação.

    Args:
        `model` (CNNModel): Instância do modelo a ser testado (deve ter a arquitetura do modelo treinado).
        `model_weights_path` (str): Caminho para o arquivo .pth contendo os pesos do modelo treinado.
        `test_loader` (DataLoader): DataLoader com as imagens e categorias atribuídas a elas.
        `class_names` (list, optional): Lista de nomes das classes para melhor visualização da matriz de confusão.
                                        Se não for fornecida, a matriz usará índices numéricos.

    Returns:
        `results` (dict): Dicionário contendo várias métricas de desempenho do modelo.
    """
    model.load_state_dict(torch.load(model_weights_path, map_location="cpu"))

    model.to(device)

    model.eval()

    all_predictions = []
    all_true_labels = []
    all_probabilities = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            
            probabilities = F.softmax(outputs, dim=1) 
            
            _, predicted = torch.max(outputs, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_true_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

    all_predictions = np.array(all_predictions)
    all_true_labels = np.array(all_true_labels)
    all_probabilities = np.array(all_probabilities)

    accuracy = accuracy_score(all_true_labels, all_predictions)
    
    precision = precision_score(all_true_labels, all_predictions, average='macro', zero_division=0)
    recall = recall_score(all_true_labels, all_predictions, average='macro', zero_division=0)
    f1 = f1_score(all_true_labels, all_predictions, average='macro', zero_division=0)
    
    conf_matrix = confusion_matrix(all_true_labels, all_predictions)

    results = {
        "model_weights": model_weights_path,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": conf_matrix
    }

    if len(np.unique(all_true_labels)) == 2:
        results["roc_auc"] = roc_auc_score(all_true_labels, all_probabilities[:, 1])

    elif len(np.unique(all_true_labels)) > 2:
        lb = LabelBinarizer()
        lb.fit(all_true_labels)
        all_true_labels_one_hot = lb.transform(all_true_labels)

        try:
            results["roc_auc_ovr"] = roc_auc_score(all_true_labels_one_hot, all_probabilities, multi_class='ovr', average='macro')

        except ValueError as e:
            print(f"Não foi possível calcular AUC-ROC para multiclasse (one-vs-rest): {e}")
            print("Isso pode acontecer se houver apenas uma classe presente em um batch ou no conjunto de teste.")

    return results