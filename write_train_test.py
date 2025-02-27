import csv
import random
input = list()

with open('input_train.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    for i in range(15000):
        v_0 = random.uniform(1, 100)
        sin_alpha = random.uniform(0.00001, 0.99999)
        H = random.uniform(0, 100)
        g = 9.81
        ans = v_0 * pow(1-sin_alpha*sin_alpha, 0.5) * pow(2*(H + (v_0*sin_alpha)*(v_0*sin_alpha)/2/g)/g, 0.5)
        input.append(v_0)
        input.append(sin_alpha)
        input.append(H)
        input.append(g)
        input.append(ans)
        writer.writerow(input)
        input.clear()

print("Данные успешно записаны в файл input_train.csv")

with open('input_test.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    for i in range(100):
        v_0 = random.uniform(1, 100)
        sin_alpha = random.uniform(0.00001, 0.99999)
        H = random.uniform(0, 100)
        g = 9.81
        ans = v_0 * pow(1-sin_alpha*sin_alpha, 0.5) * pow(2*(H + (v_0*sin_alpha)*(v_0*sin_alpha)/2/g)/g, 0.5)
        input.append(v_0)
        input.append(sin_alpha)
        input.append(H)
        input.append(g)
        input.append(ans)
        writer.writerow(input)
        input.clear()

print("Данные успешно записаны в файл input_test.csv")

