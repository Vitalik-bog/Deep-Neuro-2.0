# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 20:24:56 2021

@author: AM4
"""
import pandas as pd
import numpy as np
from neural import Perceptron

# Загрузка данных
df = pd.read_csv('data.csv')

# Перемешивание данных
df = df.iloc[np.random.permutation(len(df))]

# Подготовка обучающих данных
y = df.iloc[0:100, 4].values
y = np.where(y == "Iris-setosa", 1, -1)
X = df.iloc[0:100, [0, 2]].values

# Параметры сети
inputSize = X.shape[1]  # количество входных сигналов

# ВАЖНО: Здесь задаем 3 скрытых слоя!
hiddenSizes = [10, 5, 2]  # Явно указываем 3 слоя с разным количеством нейронов
# Или можно так (автоматически создаст 3 слоя):
# hiddenSizes = 10  # создаст слои: [10, 5, 2]

outputSize = 1  # количество выходных сигналов

# Создание сети
print(f"Создание сети с архитектурой: Входной слой ({inputSize}) -> ", end="")
for i, h in enumerate(hiddenSizes):
    print(f"Скрытый слой {i+1} ({h}) -> ", end="")
print(f"Выходной слой ({outputSize})")

NN = Perceptron(inputSize, hiddenSizes, outputSize)

# Обучение
print("\nНачало обучения...")
NN.train(X, y, n_iter=5, eta=0.01)

# Тестирование на всех данных
print("\nТестирование...")
y_all = df.iloc[:, 4].values
y_all = np.where(y_all == "Iris-setosa", 1, -1)
X_all = df.iloc[:, [0, 2]].values

# Предсказание
out, hidden_predict = NN.predict(X_all)

# Подсчет ошибок
errors = np.sum(out.flatten() != y_all)
print(f"Количество ошибок: {errors} из {len(y_all)}")
print(f"Точность: {(1 - errors/len(y_all)) * 100:.2f}%")