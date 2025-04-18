import numpy as np
import csv
import time
from tqdm import tqdm

start_time = time.time()

class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes_1, hidden_nodes_2,
                 hidden_nodes_3, hidden_nodes_4, hidden_nodes_5,
                 hidden_nodes_6, hidden_nodes_7,
                 output_nodes, learning_rate):
        
        self.inodes = input_nodes
        self.hnodes1 = hidden_nodes_1
        self.hnodes2 = hidden_nodes_2
        self.hnodes3 = hidden_nodes_3
        self.hnodes4 = hidden_nodes_4
        self.hnodes5 = hidden_nodes_5
        self.hnodes6 = hidden_nodes_6
        self.hnodes7 = hidden_nodes_7
        self.onodes = output_nodes
        self.lr = learning_rate

        # Используем He-инициализацию для слоёв с Leaky ReLU
        self.wih1 = np.random.normal(0.0, np.sqrt(2.0 / self.inodes),
                                     (self.hnodes1, self.inodes))
        self.bias1 = np.zeros((self.hnodes1,))  # bias первого скрытого слоя

        self.wh1h2 = np.random.normal(0.0, np.sqrt(2.0 / self.hnodes1),
                                      (self.hnodes2, self.hnodes1))
        self.bias2 = np.zeros((self.hnodes2,))
        
        # Аналогично для последующих слоёв...
        self.wh2h3 = np.random.normal(0.0, np.sqrt(2.0 / self.hnodes2),
                                      (self.hnodes3, self.hnodes2))
        self.bias3 = np.zeros((self.hnodes3,))
        
        self.wh3h4 = np.random.normal(0.0, np.sqrt(2.0 / self.hnodes3),
                                      (self.hnodes4, self.hnodes3))
        self.bias4 = np.zeros((self.hnodes4,))
        
        self.wh4h5 = np.random.normal(0.0, np.sqrt(2.0 / self.hnodes4),
                                      (self.hnodes5, self.hnodes4))
        self.bias5 = np.zeros((self.hnodes5,))
        
        self.wh5h6 = np.random.normal(0.0, np.sqrt(2.0 / self.hnodes5),
                                      (self.hnodes6, self.hnodes5))
        self.bias6 = np.zeros((self.hnodes6,))
        
        self.wh6h7 = np.random.normal(0.0, np.sqrt(2.0 / self.hnodes6),
                                      (self.hnodes7, self.hnodes6))
        self.bias7 = np.zeros((self.hnodes7,))
        
        self.who   = np.random.normal(0.0, np.sqrt(2.0 / self.hnodes7),
                                      (self.onodes, self.hnodes7))
        self.bias_out = np.zeros((self.onodes,))
        
        # Leaky ReLU для скрытых слоёв
        self.activation_function = lambda x: np.where(x >= 0, x, 0.01 * x)
        # Для выходного слоя – линейная функция (identity) для задач регрессии
        self.output_activation = lambda x: x

    def train(self, inputs_batch, targets_batch):
        inputs = np.array(inputs_batch, dtype=np.float32)
        targets = np.array(targets_batch, dtype=np.float32).reshape(-1, 1)
        
        # --- Прямое распространение ---
        hidden1_inputs = np.dot(inputs, self.wih1.T) + self.bias1
        hidden1_outputs = self.activation_function(hidden1_inputs)
        
        hidden2_inputs = np.dot(hidden1_outputs, self.wh1h2.T) + self.bias2
        hidden2_outputs = self.activation_function(hidden2_inputs)
        
        hidden3_inputs = np.dot(hidden2_outputs, self.wh2h3.T) + self.bias3
        hidden3_outputs = self.activation_function(hidden3_inputs)
        
        hidden4_inputs = np.dot(hidden3_outputs, self.wh3h4.T) + self.bias4
        hidden4_outputs = self.activation_function(hidden4_inputs)
        
        hidden5_inputs = np.dot(hidden4_outputs, self.wh4h5.T) + self.bias5
        hidden5_outputs = self.activation_function(hidden5_inputs)
        
        hidden6_inputs = np.dot(hidden5_outputs, self.wh5h6.T) + self.bias6
        hidden6_outputs = self.activation_function(hidden6_inputs)
        
        hidden7_inputs = np.dot(hidden6_outputs, self.wh6h7.T) + self.bias7
        hidden7_outputs = self.activation_function(hidden7_inputs)
        
        final_inputs = np.dot(hidden7_outputs, self.who.T) + self.bias_out
        # Выходной слой – линейная функция
        final_outputs = self.output_activation(final_inputs)
        
        # --- Обратное распространение ---
        # Для регрессии часто используется MSE (среднеквадратичная ошибка)
        error = targets - final_outputs  # ошибка на выходе
        
        # Производная линейной функции равна 1
        grad_output = error  # если использовать MSE, можно напрямую брать ошибку
        
        # Ошибка и градиенты для слоя между hidden7 и output
        hidden7_errors = np.dot(grad_output, self.who)
        threshold = 0.5         
        # Градиент для слоев вычисляем с учётом производной Leaky ReLU
        grad_hidden7 = hidden7_errors * np.where(hidden7_outputs >= 0, 1.0, 0.01)
        grad_hidden7 = np.clip(grad_hidden7, -threshold, threshold)
        hidden6_errors = np.dot(grad_hidden7, self.wh6h7)
        grad_hidden6 = hidden6_errors * np.where(hidden6_outputs >= 0, 1.0, 0.01)
        grad_hidden6 = np.clip(grad_hidden6, -threshold, threshold)
        hidden5_errors = np.dot(grad_hidden6, self.wh5h6)
        grad_hidden5 = hidden5_errors * np.where(hidden5_outputs >= 0, 1.0, 0.01)
        grad_hidden5 = np.clip(grad_hidden5, -threshold, threshold)
        hidden4_errors = np.dot(grad_hidden5, self.wh4h5)
        grad_hidden4 = hidden4_errors * np.where(hidden4_outputs >= 0, 1.0, 0.01)
        grad_hidden4 = np.clip(grad_hidden4, -threshold, threshold)
        hidden3_errors = np.dot(grad_hidden4, self.wh3h4)
        grad_hidden3 = hidden3_errors * np.where(hidden3_outputs >= 0, 1.0, 0.01)
        grad_hidden3 = np.clip(grad_hidden3, -threshold, threshold)
        hidden2_errors = np.dot(grad_hidden3, self.wh2h3)
        grad_hidden2 = hidden2_errors * np.where(hidden2_outputs >= 0, 1.0, 0.01)
        grad_hidden2 = np.clip(grad_hidden2, -threshold, threshold)
        hidden1_errors = np.dot(grad_hidden2, self.wh1h2)
        grad_hidden1 = hidden1_errors * np.where(hidden1_outputs >= 0, 1.0, 0.01)
        grad_hidden1 = np.clip(grad_hidden1, -threshold, threshold)
        
        # --- Обновление весов и смещений ---
        # Выходной слой
        self.who   += self.lr * np.dot(grad_output.T, hidden7_outputs)
        self.bias_out += self.lr * np.sum(grad_output, axis=0)
        
        self.wh6h7 += self.lr * np.dot(grad_hidden7.T, hidden6_outputs)
        self.bias7 += self.lr * np.sum(grad_hidden7, axis=0)
        
        self.wh5h6 += self.lr * np.dot(grad_hidden6.T, hidden5_outputs)
        self.bias6 += self.lr * np.sum(grad_hidden6, axis=0)
        
        self.wh4h5 += self.lr * np.dot(grad_hidden5.T, hidden4_outputs)
        self.bias5 += self.lr * np.sum(grad_hidden5, axis=0)
        
        self.wh3h4 += self.lr * np.dot(grad_hidden4.T, hidden3_outputs)
        self.bias4 += self.lr * np.sum(grad_hidden4, axis=0)
        
        self.wh2h3 += self.lr * np.dot(grad_hidden3.T, hidden2_outputs)
        self.bias3 += self.lr * np.sum(grad_hidden3, axis=0)
        
        self.wh1h2 += self.lr * np.dot(grad_hidden2.T, hidden1_outputs)
        self.bias2 += self.lr * np.sum(grad_hidden2, axis=0)
        
        self.wih1  += self.lr * np.dot(grad_hidden1.T, inputs)
        self.bias1 += self.lr * np.sum(grad_hidden1, axis=0)

    def query(self, inputs_batch):
        inputs = np.array(inputs_batch, dtype=np.float32)
        
        hidden1_inputs = np.dot(inputs, self.wih1.T) + self.bias1
        hidden1_outputs = self.activation_function(hidden1_inputs)
        
        hidden2_inputs = np.dot(hidden1_outputs, self.wh1h2.T) + self.bias2
        hidden2_outputs = self.activation_function(hidden2_inputs)
        
        hidden3_inputs = np.dot(hidden2_outputs, self.wh2h3.T) + self.bias3
        hidden3_outputs = self.activation_function(hidden3_inputs)
        
        hidden4_inputs = np.dot(hidden3_outputs, self.wh3h4.T) + self.bias4
        hidden4_outputs = self.activation_function(hidden4_inputs)
        
        hidden5_inputs = np.dot(hidden4_outputs, self.wh4h5.T) + self.bias5
        hidden5_outputs = self.activation_function(hidden5_inputs)
        
        hidden6_inputs = np.dot(hidden5_outputs, self.wh5h6.T) + self.bias6
        hidden6_outputs = self.activation_function(hidden6_inputs)
        
        hidden7_inputs = np.dot(hidden6_outputs, self.wh6h7.T) + self.bias7
        hidden7_outputs = self.activation_function(hidden7_inputs)
        
        final_inputs = np.dot(hidden7_outputs, self.who.T) + self.bias_out
        final_outputs = self.output_activation(final_inputs)
        return final_outputs

# Остальная часть кода (загрузка данных, обучение, тестирование) остается без принципиальных изменений.

# Параметры сети
input_nodes = 7
hidden_nodes_1 = 512
hidden_nodes_2 = 512
hidden_nodes_3 = 512
hidden_nodes_4 = 512
hidden_nodes_5 = 256
hidden_nodes_6 = 256
hidden_nodes_7 = 256
output_nodes = 1
learning_rate = 0.001

epochs = 500
batch_size = 32

# Загрузка train-данных
inputs_1 = []
with open('input_train.csv', 'r', newline='') as file:
    reader = csv.reader(file)
    for row in reader:
        inputs_1.append([float(x) for x in row])

inputs_1 = np.array(inputs_1)

max_values = np.max(inputs_1, axis=0)

inputs_1_normalized = inputs_1 / max_values

X_train = inputs_1_normalized[:, :7]
y_train = inputs_1_normalized[:, 7:]

n_samples = X_train.shape[0]

# Инициализация сети
n = NeuralNetwork(input_nodes, hidden_nodes_1, hidden_nodes_2,
                  hidden_nodes_3, hidden_nodes_4, hidden_nodes_5,
                  hidden_nodes_6, hidden_nodes_7,
                  output_nodes, learning_rate)

# Обучение
initial_lr = n.lr
decay_rate = 0.99
for epoch in tqdm(range(epochs)):

    # Снижение lr
    n.lr = initial_lr * (decay_rate ** epoch)

    permutation = np.random.permutation(n_samples)
    X_train = X_train[permutation]
    y_train = y_train[permutation]

    total_error = 0.0

    for start_ind in range(0, n_samples, batch_size):
        end_ind = start_ind + batch_size
        if end_ind > n_samples:
            end_ind = n_samples

        X_batch = X_train[start_ind:end_ind]
        y_batch = y_train[start_ind:end_ind]

        # Шаг обучения
        n.train(X_batch, y_batch)

        # Подсчёт ошибки в данном батче
        outputs_batch = n.query(X_batch)
        nonzero = (y_batch != 0)
        error_batch = np.abs(y_batch[nonzero] - outputs_batch[nonzero]) / y_batch[nonzero] * 100

 
        # Чтобы сложить с total_error (float), приведём cupy-скаляр к float
        batch_error_val = np.sum(error_batch).item()  # .item() даст обычный float
        total_error += batch_error_val

    mean_error = total_error / n_samples
    if (epoch + 1) % 10 == 0:
        print(f"Эпоха {epoch + 1}, средняя ошибка: {mean_error:.2f}%")

print("Обучение завершено")

# Тестовые данные
input_test_list = []
with open('input_test.csv', 'r', newline='') as file:
    reader = csv.reader(file)
    for row in reader:
        input_test_list.append([float(x) for x in row])

input_test_list = np.array(input_test_list)
input_test_normalized = input_test_list / max_values

X_test = input_test_normalized[:, :7]
y_test = input_test_normalized[:, 7:]

# Предсказания
predictions_normalized = n.query(X_test)
predictions = predictions_normalized * max_values[7]  # возвр. масштаб

true_values = input_test_list[:, 7]

procent_of_fails_list = []
number_of_tests = len(true_values)
for i in range(number_of_tests):
    ans = true_values[i]
    pred = predictions[i, 0]

    print(f"\nТест {i + 1}:")
    print(f"Ответ сети: {pred:.2f}")
    print(f"Правильный ответ: {ans:.2f}")

    if ans != 0:
        # pred и ans могут быть 0d cupy-скалярами,
        # поэтому берем item() или float()
        error_val = float(abs(pred - ans) / ans * 100)
    else:
        error_val = 0

    procent_of_fails_list.append(error_val)

# Вычисление средней процентной ошибки вручную
sum_error = 0.0
for val in procent_of_fails_list:
    sum_error += val

mean_error = sum_error / len(procent_of_fails_list)

# Вывод финальной информации
print(f"\nКоэффициент обучения (финальный): {n.lr}")
print(f"Количество эпох: {epochs}")
print(f"Средняя процентная ошибка (тест): {mean_error:.2f}%")

end_time = time.time()
elapsed_time = end_time - start_time
print("Время выполнения программы (мин):", elapsed_time / 60.0)

