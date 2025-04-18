import csv
import random
import math
input = list()

with open('input_train.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    for i in range(75000):
        x_0 = random.uniform(10, 100)
        v_0 = random.uniform(10, 100)
        m = random.uniform(10, 100)
        F_0 = random.uniform(10, 100)
        k = random.uniform(1, 5)
        w_0 = random.uniform(10, 100)
        if abs(w_0*w_0 - k/m) < 1e-6:
            w_0 = random.uniform(10, 100)
        t = random.uniform(0, math.pi/2)
        ans = pow(x_0*x_0 + (v_0*v_0)/k*m, 0.5) * math.sin(k/m*t + math.atan(x_0/v_0*pow(k/m, 0.5))) + F_0/k/(1 - w_0*w_0/k*m)*math.sin(w_0*t)
        ans += random.uniform(-0.05, 0.05)* ans
        input.append(x_0)
        input.append(v_0)
        input.append(m)
        input.append(F_0)
        input.append(k)
        input.append(w_0)
        input.append(t)
        input.append(ans)
        writer.writerow(input)
        input.clear()

print("Данные успешно записаны в файл input_train.csv")

with open('input_test.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    for i in range(100):
        x_0 = random.uniform(10, 100)
        v_0 = random.uniform(10, 100)
        m = random.uniform(10, 100)
        F_0 = random.uniform(10, 100)
        k = random.uniform(1, 5)
        w_0 = random.uniform(10, 100)
        if abs(w_0*w_0 - k/m) < 1e-6:
            w_0 = random.uniform(10, 100)
        t = random.uniform(0, math.pi/2)
        ans = pow(x_0*x_0 + (v_0*v_0)/k*m, 0.5) * math.sin(k/m*t + math.atan(x_0/v_0*pow(k/m, 0.5))) + F_0/k/(1 - w_0*w_0/k*m)*math.sin(w_0*t)
        ans += random.uniform(-0.05, 0.05)* ans
        input.append(x_0)
        input.append(v_0)
        input.append(m)
        input.append(F_0)
        input.append(k)
        input.append(w_0)
        input.append(t)
        input.append(ans)
        writer.writerow(input)
        input.clear()

print("Данные успешно записаны в файл input_test.csv")

