import numpy as np
import csv
import time

start_time = time.time()
class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes_1, hidden_nodes_2, hidden_nodes_3, output_nodes, learning_rate):
        self.inodes = input_nodes
        self.hnodes1 = hidden_nodes_1
        self.hnodes2 = hidden_nodes_2
        self.hnodes3 = hidden_nodes_3 
        self.onodes = output_nodes
        
   
        self.wih1 = np.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes1, self.inodes))
        self.wh1h2 = np.random.normal(0.0, pow(self.hnodes1, -0.5), (self.hnodes2, self.hnodes1))
        self.wh2h3 = np.random.normal(0.0, pow(self.hnodes2, -0.5), (self.hnodes3, self.hnodes2))  
        self.who = np.random.normal(0.0, pow(self.hnodes3, -0.5), (self.onodes, self.hnodes3))  
        
        self.lr = learning_rate
        self.activation_function = lambda x: np.where(x >= 0, x, 0.01 * x) 
        
    def train(self, inputs_list, targets_list):
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T
        
        hidden1_inputs = np.dot(self.wih1, inputs)
        hidden1_outputs = self.activation_function(hidden1_inputs)
        
        hidden2_inputs = np.dot(self.wh1h2, hidden1_outputs)
        hidden2_outputs = self.activation_function(hidden2_inputs)
        
      
        hidden3_inputs = np.dot(self.wh2h3, hidden2_outputs)
        hidden3_outputs = self.activation_function(hidden3_inputs)
        
        final_inputs = np.dot(self.who, hidden3_outputs)  
        final_outputs = self.activation_function(final_inputs)
        
   
        output_errors = targets - final_outputs
        hidden3_errors = np.dot(self.who.T, output_errors)  
        hidden2_errors = np.dot(self.wh2h3.T, hidden3_errors) 
        hidden1_errors = np.dot(self.wh1h2.T, hidden2_errors)
        
        output_grad = np.where(final_outputs >= 0, 1, 0.01)
        hidden3_grad = np.where(hidden3_outputs >= 0, 1, 0.01)  
        hidden2_grad = np.where(hidden2_outputs >= 0, 1, 0.01)
        hidden1_grad = np.where(hidden1_outputs >= 0, 1, 0.01)
        

        self.who += self.lr * np.dot((output_errors * output_grad), hidden3_outputs.T) 
        self.wh2h3 += self.lr * np.dot((hidden3_errors * hidden3_grad), hidden2_outputs.T)  
        self.wh1h2 += self.lr * np.dot((hidden2_errors * hidden2_grad), hidden1_outputs.T)
        self.wih1 += self.lr * np.dot((hidden1_errors * hidden1_grad), inputs.T)
    
    def query(self, inputs_list):
        inputs = np.array(inputs_list, ndmin=2).T
        
        hidden1_inputs = np.dot(self.wih1, inputs)
        hidden1_outputs = self.activation_function(hidden1_inputs)
        
        hidden2_inputs = np.dot(self.wh1h2, hidden1_outputs)
        hidden2_outputs = self.activation_function(hidden2_inputs)
        
   
        hidden3_inputs = np.dot(self.wh2h3, hidden2_outputs)
        hidden3_outputs = self.activation_function(hidden3_inputs)
        
        final_inputs = np.dot(self.who, hidden3_outputs)
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs


input_nodes = 4
hidden_nodes_1 = 256
hidden_nodes_2 = 128
hidden_nodes_3 = 64
output_nodes = 1
learning_rate = 0.01
epochs = 500
number_of_trains = 20000


inputs_1 = []
with open('input_train.csv', 'r', newline='') as file:
    reader = csv.reader(file)
    for row in reader:
        inputs_1.append([float(x) for x in row])


inputs_1 = np.array(inputs_1)
max_values = np.max(inputs_1, axis=0)
inputs_1_normalized = inputs_1 / max_values  


n = NeuralNetwork(input_nodes, hidden_nodes_1, hidden_nodes_2, hidden_nodes_3, output_nodes, learning_rate)


for epoch in range(epochs):
    np.random.shuffle(inputs_1_normalized)
    total_error = 0
    for i in range(number_of_trains):
        input_list = inputs_1_normalized[i, :4]
        target_list = [inputs_1_normalized[i, 4]]
        n.train(input_list, target_list)
        output = n.query(input_list)[0, 0]
        total_error += abs(target_list[0] - output) / target_list[0] * 100
    if (epoch + 1) % 100 == 0:
        print(f"Эпоха {epoch + 1}, Средняя ошибка: {total_error / number_of_trains:.2f}%")


print("Обучение завершено")


input_test_list = []
with open('input_test.csv', 'r', newline='') as file:
    reader = csv.reader(file)
    for row in reader:
        input_test_list.append([float(x) for x in row])

input_test_list = np.array(input_test_list)
input_test_normalized = input_test_list / max_values 

procent_of_fails_list = []
number_of_tests = len(input_test_list)
for i in range(number_of_tests):
    v_0, sin_alpha, H, g = input_test_normalized[i, :4]
    ans_normalized = input_test_normalized[i, 4]
    ans = input_test_list[i, 4] 
    
    predicted_normalized = n.query([v_0, sin_alpha, H, g])[0, 0]
    predicted = predicted_normalized * max_values[4] 
    
    print(f"\nТест {i + 1}:")
    print(f"Ответ сети: {predicted:.2f}")
    print(f"Правильный ответ: {ans:.2f}")
    
    error = abs(predicted - ans) / ans * 100
    procent_of_fails_list.append(error)


mean_error = sum(procent_of_fails_list) / len(procent_of_fails_list)
print(f"\nКоэффициент обучения: {n.lr}")
print(f"Количество эпох: {epochs}")
print(f"Средняя процентная ошибка: {mean_error:.2f}%")
end_time = time.time()
elased_time = end_time - start_time
print("время выполнения программы", elased_time/60.0)
'''Эпоха 100, Средняя ошибка: 2.73%
Эпоха 200, Средняя ошибка: 2.25%
Эпоха 300, Средняя ошибка: 2.05%
Эпоха 400, Средняя ошибка: 2.00%
Эпоха 500, Средняя ошибка: 1.93%
Обучение завершено

Тест 1:
Ответ сети: 12.69
Правильный ответ: 10.69

Тест 2:
Ответ сети: 19.04
Правильный ответ: 17.55

Тест 3:
Ответ сети: 268.96
Правильный ответ: 267.19

Тест 4:
Ответ сети: 189.66
Правильный ответ: 187.92

Тест 5:
Ответ сети: 207.07
Правильный ответ: 206.19

Тест 6:
Ответ сети: 326.24
Правильный ответ: 326.38

Тест 7:
Ответ сети: 254.25
Правильный ответ: 253.93

Тест 8:
Ответ сети: 75.77
Правильный ответ: 74.26

Тест 9:
Ответ сети: 68.98
Правильный ответ: 69.02

Тест 10:
Ответ сети: 11.92
Правильный ответ: 12.70

Тест 11:
Ответ сети: 29.03
Правильный ответ: 29.72

Тест 12:
Ответ сети: 184.55
Правильный ответ: 183.42

Тест 13:
Ответ сети: 198.04
Правильный ответ: 197.68

Тест 14:
Ответ сети: 206.54
Правильный ответ: 208.05

Тест 15:
Ответ сети: 270.75
Правильный ответ: 267.90

Тест 16:
Ответ сети: 260.39
Правильный ответ: 260.67

Тест 17:
Ответ сети: 147.67
Правильный ответ: 147.18

Тест 18:
Ответ сети: 44.19
Правильный ответ: 42.69

Тест 19:
Ответ сети: 77.70
Правильный ответ: 76.65

Тест 20:
Ответ сети: 126.88
Правильный ответ: 126.53

Тест 21:
Ответ сети: 273.13
Правильный ответ: 271.73

Тест 22:
Ответ сети: 141.75
Правильный ответ: 141.79

Тест 23:
Ответ сети: 120.57
Правильный ответ: 120.01

Тест 24:
Ответ сети: 8.86
Правильный ответ: 11.91

Тест 25:
Ответ сети: 333.24
Правильный ответ: 331.56

Тест 26:
Ответ сети: 188.47
Правильный ответ: 189.35

Тест 27:
Ответ сети: 238.37
Правильный ответ: 235.61

Тест 28:
Ответ сети: 347.00
Правильный ответ: 346.02

Тест 29:
Ответ сети: 288.85
Правильный ответ: 291.10

Тест 30:
Ответ сети: 16.07
Правильный ответ: 15.43

Тест 31:
Ответ сети: 97.91
Правильный ответ: 94.21

Тест 32:
Ответ сети: 20.92
Правильный ответ: 19.87

Тест 33:
Ответ сети: 114.55
Правильный ответ: 114.05

Тест 34:
Ответ сети: 411.16
Правильный ответ: 410.89

Тест 35:
Ответ сети: 481.46
Правильный ответ: 478.76

Тест 36:
Ответ сети: 309.08
Правильный ответ: 308.68

Тест 37:
Ответ сети: 470.07
Правильный ответ: 466.35

Тест 38:
Ответ сети: 64.88
Правильный ответ: 65.91

Тест 39:
Ответ сети: 206.93
Правильный ответ: 207.01

Тест 40:
Ответ сети: 67.33
Правильный ответ: 68.08

Тест 41:
Ответ сети: 23.81
Правильный ответ: 21.42

Тест 42:
Ответ сети: 235.27
Правильный ответ: 233.47

Тест 43:
Ответ сети: 581.73
Правильный ответ: 580.27

Тест 44:
Ответ сети: 30.46
Правильный ответ: 27.96

Тест 45:
Ответ сети: 6.15
Правильный ответ: 6.20

Тест 46:
Ответ сети: 177.32
Правильный ответ: 177.42

Тест 47:
Ответ сети: 253.21
Правильный ответ: 253.86

Тест 48:
Ответ сети: 196.43
Правильный ответ: 198.26

Тест 49:
Ответ сети: 198.39
Правильный ответ: 197.38

Тест 50:
Ответ сети: 48.61
Правильный ответ: 48.17

Тест 51:
Ответ сети: 145.75
Правильный ответ: 143.20

Тест 52:
Ответ сети: 63.95
Правильный ответ: 62.55

Тест 53:
Ответ сети: 24.14
Правильный ответ: 23.84

Тест 54:
Ответ сети: 232.93
Правильный ответ: 226.44

Тест 55:
Ответ сети: 180.59
Правильный ответ: 179.03

Тест 56:
Ответ сети: 259.11
Правильный ответ: 255.74

Тест 57:
Ответ сети: 158.13
Правильный ответ: 157.50

Тест 58:
Ответ сети: 90.59
Правильный ответ: 88.87

Тест 59:
Ответ сети: 126.60
Правильный ответ: 124.54

Тест 60:
Ответ сети: 119.06
Правильный ответ: 116.71

Тест 61:
Ответ сети: 468.61
Правильный ответ: 465.86

Тест 62:
Ответ сети: 174.97
Правильный ответ: 174.00

Тест 63:
Ответ сети: 196.52
Правильный ответ: 195.19

Тест 64:
Ответ сети: 20.54
Правильный ответ: 19.42

Тест 65:
Ответ сети: 27.22
Правильный ответ: 26.87

Тест 66:
Ответ сети: 46.41
Правильный ответ: 46.71

Тест 67:
Ответ сети: 183.13
Правильный ответ: 182.93

Тест 68:
Ответ сети: 42.90
Правильный ответ: 42.61

Тест 69:
Ответ сети: 391.41
Правильный ответ: 391.25

Тест 70:
Ответ сети: 37.75
Правильный ответ: 36.67

Тест 71:
Ответ сети: 335.55
Правильный ответ: 332.44

Тест 72:
Ответ сети: 175.85
Правильный ответ: 174.33

Тест 73:
Ответ сети: 136.73
Правильный ответ: 133.87

Тест 74:
Ответ сети: 360.08
Правильный ответ: 357.81

Тест 75:
Ответ сети: 178.79
Правильный ответ: 176.87

Тест 76:
Ответ сети: 90.61
Правильный ответ: 89.38

Тест 77:
Ответ сети: 16.42
Правильный ответ: 17.05

Тест 78:
Ответ сети: 7.23
Правильный ответ: 7.60

Тест 79:
Ответ сети: 3.30
Правильный ответ: 2.36

Тест 80:
Ответ сети: 219.89
Правильный ответ: 219.81

Тест 81:
Ответ сети: 235.40
Правильный ответ: 235.78

Тест 82:
Ответ сети: 33.20
Правильный ответ: 32.41

Тест 83:
Ответ сети: 36.98
Правильный ответ: 35.71

Тест 84:
Ответ сети: 296.49
Правильный ответ: 296.35

Тест 85:
Ответ сети: 223.94
Правильный ответ: 226.16

Тест 86:
Ответ сети: 23.93
Правильный ответ: 22.38

Тест 87:
Ответ сети: 65.84
Правильный ответ: 64.72

Тест 88:
Ответ сети: 257.77
Правильный ответ: 258.12

Тест 89:
Ответ сети: 274.57
Правильный ответ: 274.04

Тест 90:
Ответ сети: 38.37
Правильный ответ: 38.03

Тест 91:
Ответ сети: 97.71
Правильный ответ: 96.95

Тест 92:
Ответ сети: 109.59
Правильный ответ: 110.28

Тест 93:
Ответ сети: 48.10
Правильный ответ: 47.88

Тест 94:
Ответ сети: 75.43
Правильный ответ: 75.26

Тест 95:
Ответ сети: 40.13
Правильный ответ: 39.26

Тест 96:
Ответ сети: 18.44
Правильный ответ: 17.88

Тест 97:
Ответ сети: 43.31
Правильный ответ: 41.87

Тест 98:
Ответ сети: 16.66
Правильный ответ: 16.02

Тест 99:
Ответ сети: 387.04
Правильный ответ: 387.40

Тест 100:
Ответ сети: 62.66
Правильный ответ: 60.63

Коэффициент обучения: 0.01
Количество эпох: 500
Средняя процентная ошибка: 2.41%
время выполнения программы 18.539221799373628'''
