import numpy as np
import csv
import time
from tqdm import tqdm

start_time = time.time()

class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes_1, hidden_nodes_2,
                 hidden_nodes_3, hidden_nodes_4, hidden_nodes_5,
                 output_nodes, learning_rate):
        
        self.inodes = input_nodes
        self.hnodes1 = hidden_nodes_1
        self.hnodes2 = hidden_nodes_2
        self.hnodes3 = hidden_nodes_3
        self.hnodes4 = hidden_nodes_4
        self.hnodes5 = hidden_nodes_5
        self.onodes = output_nodes
        
        # Веса (форма такая же, как в вашем исходном коде)
        # (hnodes1, inodes) => (512, 7)
        self.wih1 = np.random.normal(0.0, pow(self.inodes, -0.5),
                                     (self.hnodes1, self.inodes))
        self.wh1h2 = np.random.normal(0.0, pow(self.hnodes1, -0.5),
                                      (self.hnodes2, self.hnodes1))
        self.wh2h3 = np.random.normal(0.0, pow(self.hnodes2, -0.5),
                                      (self.hnodes3, self.hnodes2))
        self.wh3h4 = np.random.normal(0.0, pow(self.hnodes3, -0.5),
                                      (self.hnodes4, self.hnodes3))
        self.wh4h5 = np.random.normal(0.0, pow(self.hnodes4, -0.5),
                                      (self.hnodes5, self.hnodes4))
        self.who   = np.random.normal(0.0, pow(self.hnodes5, -0.5),
                                      (self.onodes, self.hnodes5))
        
        self.lr = learning_rate
        
        # Leaky ReLU
        self.activation_function = lambda x: np.where(x >= 0, x, 0.01 * x)


    def train(self, inputs_batch, targets_batch):
        """
        Обучение на целой партии (batch).
        - inputs_batch: shape (batch_size, inodes)       [например, (32, 7)]
        - targets_batch: shape (batch_size, 1)           [например, (32, 1)]
        
        Вся логика прямого и обратного прохода выполняется векторизованно.
        """
        # ---------------- Приводим типы и формы ----------------
        # (batch_size, inodes)
        inputs = np.array(inputs_batch, dtype=np.float32)
        # (batch_size, 1)
        targets = np.array(targets_batch, dtype=np.float32).reshape(-1, 1)

        batch_size = inputs.shape[0]

        # =============== ПРЯМОЙ ПРОХОД ===============

        # hidden1_inputs = wih1 dot inputs (но с учетом batch)
        # self.wih1 shape = (512, 7), inputs shape = (batch_size, 7)
        # Чтобы результат был (batch_size, 512), делаем:
        hidden1_inputs = np.dot(inputs, self.wih1.T)  # => (batch_size, 512)
        hidden1_outputs = self.activation_function(hidden1_inputs)

        hidden2_inputs = np.dot(hidden1_outputs, self.wh1h2.T)  # => (batch_size, 256)
        hidden2_outputs = self.activation_function(hidden2_inputs)

        hidden3_inputs = np.dot(hidden2_outputs, self.wh2h3.T)  # => (batch_size, 128)
        hidden3_outputs = self.activation_function(hidden3_inputs)

        hidden4_inputs = np.dot(hidden3_outputs, self.wh3h4.T)  # => (batch_size, 64)
        hidden4_outputs = self.activation_function(hidden4_inputs)

        hidden5_inputs = np.dot(hidden4_outputs, self.wh4h5.T)  # => (batch_size, 32)
        hidden5_outputs = self.activation_function(hidden5_inputs)

        # final_inputs = hidden5_outputs dot who.T
        # who shape = (1, 32), so who.T = (32, 1)
        final_inputs = np.dot(hidden5_outputs, self.who.T)  # => (batch_size, 1)
        final_outputs = self.activation_function(final_inputs)  # => (batch_size, 1)

        # =============== ОБРАТНЫЙ ПРОХОД ===============
        output_errors = targets - final_outputs  # (batch_size, 1)

        # Градиент Leaky ReLU на выходе
        output_grad = np.where(final_outputs >= 0, 1.0, 0.01)  # (batch_size, 1)

        # Перемножаем покомпонентно (batch_size,1)
        grad_output = output_errors * output_grad

        # ------ Ошибки слоя hidden5 ------
        # hidden5_errors = grad_output dot who
        # grad_output shape: (batch_size,1), who shape: (1,32)
        hidden5_errors = np.dot(grad_output, self.who)  # => (batch_size, 32)

        hidden5_grad = np.where(hidden5_outputs >= 0, 1.0, 0.01)  # (batch_size, 32)
        grad_hidden5 = hidden5_errors * hidden5_grad  # (batch_size, 32)

        # ------ Ошибки hidden4 ------
        hidden4_errors = np.dot(grad_hidden5, self.wh4h5)  # => (batch_size, 64)
        hidden4_grad = np.where(hidden4_outputs >= 0, 1.0, 0.01)
        grad_hidden4 = hidden4_errors * hidden4_grad  # => (batch_size, 64)

        # ------ Ошибки hidden3 ------
        hidden3_errors = np.dot(grad_hidden4, self.wh3h4)  # => (batch_size, 128)
        hidden3_grad = np.where(hidden3_outputs >= 0, 1.0, 0.01)
        grad_hidden3 = hidden3_errors * hidden3_grad  # => (batch_size, 128)

        # ------ Ошибки hidden2 ------
        hidden2_errors = np.dot(grad_hidden3, self.wh2h3)  # => (batch_size, 256)
        hidden2_grad = np.where(hidden2_outputs >= 0, 1.0, 0.01)
        grad_hidden2 = hidden2_errors * hidden2_grad  # => (batch_size, 256)

        # ------ Ошибки hidden1 ------
        hidden1_errors = np.dot(grad_hidden2, self.wh1h2)  # => (batch_size, 512)
        hidden1_grad = np.where(hidden1_outputs >= 0, 1.0, 0.01)
        grad_hidden1 = hidden1_errors * hidden1_grad  # => (batch_size, 512)

        # =============== ОБНОВЛЕНИЕ ВЕСОВ ===============

        # who shape = (1, 32)
        # Чтобы корректно получить dWho = (1,32), делаем:
        #   grad_output: (batch_size, 1),  hidden5_outputs: (batch_size, 32)
        #   dWho = grad_output^T dot hidden5_outputs  => (1,32)
        self.who += self.lr * np.dot(grad_output.T, hidden5_outputs)

        # wh4h5 shape = (32,64)
        #   grad_hidden5: (batch_size,32), hidden4_outputs: (batch_size,64)
        #   dWh4h5 = grad_hidden5^T dot hidden4_outputs => (32,64)
        self.wh4h5 += self.lr * np.dot(grad_hidden5.T, hidden4_outputs)

        # wh3h4 shape = (64,128)
        #   grad_hidden4: (batch_size,64), hidden3_outputs: (batch_size,128)
        #   dWh3h4 = grad_hidden4^T dot hidden3_outputs => (64,128)
        self.wh3h4 += self.lr * np.dot(grad_hidden4.T, hidden3_outputs)

        # wh2h3 shape = (128,256)
        self.wh2h3 += self.lr * np.dot(grad_hidden3.T, hidden2_outputs)

        # wh1h2 shape = (256,512)
        self.wh1h2 += self.lr * np.dot(grad_hidden2.T, hidden1_outputs)

        # wih1 shape = (512,7)
        #   grad_hidden1: (batch_size,512), inputs: (batch_size,7)
        #   dWih1 = grad_hidden1^T dot inputs => (512,7)
        self.wih1 += self.lr * np.dot(grad_hidden1.T, inputs)


    def query(self, inputs_batch):
        """
        Прямой проход для батча.
        inputs_batch: (batch_size, 7)
        Возвращает final_outputs: (batch_size, 1)
        """
        inputs = np.array(inputs_batch, dtype=np.float32)

        hidden1_inputs = np.dot(inputs, self.wih1.T)
        hidden1_outputs = self.activation_function(hidden1_inputs)

        hidden2_inputs = np.dot(hidden1_outputs, self.wh1h2.T)
        hidden2_outputs = self.activation_function(hidden2_inputs)

        hidden3_inputs = np.dot(hidden2_outputs, self.wh2h3.T)
        hidden3_outputs = self.activation_function(hidden3_inputs)

        hidden4_inputs = np.dot(hidden3_outputs, self.wh3h4.T)
        hidden4_outputs = self.activation_function(hidden4_inputs)

        hidden5_inputs = np.dot(hidden4_outputs, self.wh4h5.T)
        hidden5_outputs = self.activation_function(hidden5_inputs)

        final_inputs = np.dot(hidden5_outputs, self.who.T)
        final_outputs = self.activation_function(final_inputs)
        return final_outputs


# --------------- Параметры ---------------
input_nodes = 7
hidden_nodes_1 = 1024
hidden_nodes_2 = 512
hidden_nodes_3 = 256
hidden_nodes_4 = 128
hidden_nodes_5 = 64
output_nodes = 1
learning_rate = 0.01

epochs = 500
batch_size = 8  # размер мини-батча

# --------------- Загрузка тренировочных данных ---------------
inputs_1 = []
with open('input_train.csv', 'r', newline='') as file:
    reader = csv.reader(file)
    for row in reader:
        inputs_1.append([float(x) for x in row])

inputs_1 = np.array(inputs_1)
max_values = np.max(inputs_1, axis=0)

# Нормировка
inputs_1_normalized = inputs_1 / max_values

# Разделим на X и y (предполагаем, что последние значения - это цель)
X_train = inputs_1_normalized[:, :7]  # (n_samples, 7)
y_train = inputs_1_normalized[:, 7:]  # (n_samples, 1)

n_samples = X_train.shape[0]

# --------------- Создаём нейронную сеть ---------------
n = NeuralNetwork(input_nodes, hidden_nodes_1, hidden_nodes_2,
                  hidden_nodes_3, hidden_nodes_4, hidden_nodes_5,
                  output_nodes, learning_rate)

# --------------- Обучение ---------------
for epoch in tqdm(range(epochs)):

    # Каждые 200 эпох уменьшаем lr в 2 раза
    if (epoch % 200 == 0) and (epoch != 0):
        n.lr /= 1.5

    # Перемешиваем индексы
    permutation = np.random.permutation(n_samples)
    X_train = X_train[permutation]
    y_train = y_train[permutation]

    total_error = 0.0

    # Идём по мини-батчам
    for start_idx in range(0, n_samples, batch_size):
        end_idx = start_idx + batch_size
        if end_idx > n_samples:
            end_idx = n_samples
        
        X_batch = X_train[start_idx:end_idx]  # (batch_size, 7)
        y_batch = y_train[start_idx:end_idx]  # (batch_size, 1)

        # Шаг обучения на батче
        n.train(X_batch, y_batch)

        # Для подсчёта ошибки (на этих же данных)
        outputs_batch = n.query(X_batch)  # (batch_size,1)
        # Ошибку считаем в процентах: |target - out| / target * 100
        # (осторожно: если y_batch где-то 0, обрабатывать отдельно)
        # Здесь упрощённый вариант:
        mask_nonzero = (y_batch != 0)
        error_batch = np.abs(y_batch[mask_nonzero] - outputs_batch[mask_nonzero]) / y_batch[mask_nonzero] * 100
        total_error += np.sum(error_batch)

    # Средняя ошибка за эпоху
    mean_error = total_error / n_samples
    if (epoch + 1) % 100 == 0:
        print(f"Эпоха {epoch+1}, средняя ошибка: {mean_error:.2f}%")

print("Обучение завершено")

# --------------- Тестирование ---------------
input_test_list = []
with open('input_test.csv', 'r', newline='') as file:
    reader = csv.reader(file)
    for row in reader:
        input_test_list.append([float(x) for x in row])

input_test_list = np.array(input_test_list)
input_test_normalized = input_test_list / max_values

X_test = input_test_normalized[:, :7]  # (n_test, 7)
y_test = input_test_normalized[:, 7:]  # (n_test, 1) - нормированные цели

predictions_normalized = n.query(X_test)  # (n_test, 1)
predictions = predictions_normalized * max_values[7]  # денормируем
true_values = input_test_list[:, 7]  # реальные Y

procent_of_fails_list = []
number_of_tests = len(true_values)
for i in range(number_of_tests):
    ans = true_values[i]
    pred = predictions[i, 0]
    
    print(f"\nТест {i + 1}:")
    print(f"Ответ сети: {pred:.2f}")
    print(f"Правильный ответ: {ans:.2f}")
    
    error = abs(pred - ans) / ans * 100 if ans != 0 else 0
    procent_of_fails_list.append(error)

mean_error = np.mean(procent_of_fails_list)
print(f"\nКоэффициент обучения (финальный): {n.lr}")
print(f"Количество эпох: {epochs}")
print(f"Средняя процентная ошибка (тест): {mean_error:.2f}%")

end_time = time.time()
elapsed_time = end_time - start_time
print("Время выполнения программы (мин):", elapsed_time / 60.0)