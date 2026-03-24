import numpy as np

class Perceptron:
    def __init__(self, inputSize, hiddenSizes, outputSize):
        if isinstance(hiddenSizes, int):
            hiddenSizes = [hiddenSizes, max(1, hiddenSizes//2), max(1, hiddenSizes//4)]
        
        # Первый скрытый слой
        self.Win1 = np.zeros((1+inputSize, hiddenSizes[0]))
        self.Win1[0,:] = np.random.randint(0, 3, size=(hiddenSizes[0]))
        self.Win1[1:,:] = np.random.randint(-1, 2, size=(inputSize, hiddenSizes[0]))
        
        # Второй скрытый слой
        self.Win2 = np.zeros((1+hiddenSizes[0], hiddenSizes[1]))
        self.Win2[0,:] = np.random.randint(0, 3, size=(hiddenSizes[1]))
        self.Win2[1:,:] = np.random.randint(-1, 2, size=(hiddenSizes[0], hiddenSizes[1]))
        
        # Третий скрытый слой
        self.Win3 = np.zeros((1+hiddenSizes[1], hiddenSizes[2]))
        self.Win3[0,:] = np.random.randint(0, 3, size=(hiddenSizes[2]))
        self.Win3[1:,:] = np.random.randint(-1, 2, size=(hiddenSizes[1], hiddenSizes[2]))
        
        # Выходной слой
        self.Wout = np.random.randint(0, 2, size=(1+hiddenSizes[2], outputSize)).astype(np.float64)
    
    def predict(self, Xp):
        if Xp.ndim == 1:
            Xp = Xp.reshape(1, -1)
        
        # Первый скрытый слой
        hidden1 = np.dot(Xp, self.Win1[1:,:]) + self.Win1[0,:]
        hidden1 = np.where(hidden1 >= 0.0, 1, -1).astype(np.float64)
        
        # Второй скрытый слой
        hidden2 = np.dot(hidden1, self.Win2[1:,:]) + self.Win2[0,:]
        hidden2 = np.where(hidden2 >= 0.0, 1, -1).astype(np.float64)
        
        # Третий скрытый слой
        hidden3 = np.dot(hidden2, self.Win3[1:,:]) + self.Win3[0,:]
        hidden3 = np.where(hidden3 >= 0.0, 1, -1).astype(np.float64)
        
        # Выходной слой
        out = np.dot(hidden3, self.Wout[1:,:]) + self.Wout[0,:]
        out = np.where(out >= 0.0, 1, -1).astype(np.float64)
        
        return out, [hidden1, hidden2, hidden3]
    
    def train(self, X, y, n_iter=50, eta=0.01):
        best_accuracy = 0
        best_Wout = None
        
        for epoch in range(n_iter):
            correct = 0
            
            for xi, target in zip(X, y):
                xi = xi.reshape(1, -1)
                
                # Прямое распространение
                hidden1 = np.dot(xi, self.Win1[1:,:]) + self.Win1[0,:]
                hidden1 = np.where(hidden1 >= 0.0, 1, -1).astype(np.float64)
                
                hidden2 = np.dot(hidden1, self.Win2[1:,:]) + self.Win2[0,:]
                hidden2 = np.where(hidden2 >= 0.0, 1, -1).astype(np.float64)
                
                hidden3 = np.dot(hidden2, self.Win3[1:,:]) + self.Win3[0,:]
                hidden3 = np.where(hidden3 >= 0.0, 1, -1).astype(np.float64)
                
                out = np.dot(hidden3, self.Wout[1:,:]) + self.Wout[0,:]
                prediction = np.where(out >= 0.0, 1, -1).astype(np.float64)
                
                if prediction == target:
                    correct += 1
                
                # Обновляем ТОЛЬКО выходной слой
                # ИСПРАВЛЕНО: берем значение ошибки из массива
                error = (target - prediction).item()  # .item() преобразует в скаляр
                
                if error != 0:
                    # Обновление весов выходного слоя
                    self.Wout[1:,:] = self.Wout[1:,:] + eta * error * hidden3.T
                    # Обновление bias
                    self.Wout[0,:] = self.Wout[0,:] + eta * error
            
            accuracy = correct/len(X) * 100
            
            # Сохраняем лучшие веса
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_Wout = self.Wout.copy()
            
            if epoch % 10 == 0:
                print(f"Эпоха {epoch}, Точность на обучении: {accuracy:.2f}%")
        
        # Восстанавливаем лучшие веса
        self.Wout = best_Wout
        print(f"Лучшая точность на обучении: {best_accuracy:.2f}%")
        
        return self