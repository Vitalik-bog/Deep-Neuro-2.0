import numpy as np

class MLP:
    
    def __init__(self, inputSize, outputSize, learning_rate=0.1, hiddenSizes=5):
        # инициализируем нейронную сеть 
        # веса инициализируем случайными числами, но теперь будем хранить их списком
        self.weights = [
            np.random.uniform(-2, 2, size=(inputSize, hiddenSizes)),  # веса скрытого слоя
            np.random.uniform(-2, 2, size=(hiddenSizes, outputSize))  # веса выходного слоя
        ]
        self.learning_rate = learning_rate
        self.layers = None

    # сигмоида
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # нам понадобится производная от сигмоиды при вычислении градиента
    def derivative_sigmoid(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))
     
    # прямой проход 
    def feed_forward(self, x):
        input_ = x.reshape(1, -1) if x.ndim == 1 else x  # обеспечиваем правильную размерность
        hidden_ = self.sigmoid(np.dot(input_, self.weights[0])) # выход скрытого слоя
        output_ = self.sigmoid(np.dot(hidden_, self.weights[1])) # выход сети
        
        self.layers = [input_, hidden_, output_]
        return self.layers[-1]
    
    # backprop для одного примера
    def backward(self, target):
        # считаем производную ошибки сети
        err = (target - self.layers[-1])
    
        # прогоняем производную ошибки обратно ко входу, считая градиенты и корректируя веса
        for i in range(len(self.layers)-1, 0, -1):
            # ошибка слоя * производную функции активации
            err_delta = err * self.derivative_sigmoid(self.layers[i])       
            
            # пробрасываем ошибку на предыдущий слой
            err = np.dot(err_delta, self.weights[i - 1].T)
            
            # ошибка слоя * производную функции активации * на входные сигналы слоя
            dw = np.dot(self.layers[i - 1].T, err_delta)
            
            # обновляем веса слоя
            self.weights[i - 1] += self.learning_rate * dw
    
    # функция обучения с использованием стохастического градиентного спуска
    def train(self, X, y, n_iter=50, shuffle=True):
        n_samples = X.shape[0]
        
        for epoch in range(n_iter):
            # перемешиваем данные в начале каждой эпохи (опционально)
            if shuffle:
                indices = np.random.permutation(n_samples)
                X_shuffled = X[indices]
                y_shuffled = y[indices]
            else:
                X_shuffled = X
                y_shuffled = y
            
            # проходим по каждому примеру последовательно
            for i in range(n_samples):
                # берем один обучающий пример
                xi = X_shuffled[i]
                target = y_shuffled[i]
                
                # прямой проход
                self.feed_forward(xi)
                
                # обратный проход и обновление весов
                self.backward(target)
            
            # выводим ошибку каждые 10 эпох
            if epoch % 10 == 0:
                predictions = self.predict(X)
                error = np.mean(np.square(y - predictions))
                print(f"Эпоха: {epoch} || Средняя ошибка: {error}")
    
    # функция предсказания для нескольких примеров
    def predict(self, X):
        predictions = []
        for i in range(X.shape[0]):
            output = self.feed_forward(X[i])
            predictions.append(output.flatten())
        return np.array(predictions).reshape(-1, 1)