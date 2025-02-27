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
epochs = 1000
number_of_trains = 10000


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