import numpy as np
import random
import functools
from numpy.linalg import solve
from scipy.stats import f, t
import prettytable as p
from art import tprint


# Номер варіанту 207
x_range = [(-5, 15), (-35, 10), (-35, -10)]

# Знайдемо середні значення
x_aver_min = sum([i[0] for i in x_range]) / 3
x_aver_max = sum([i[1] for i in x_range]) / 3

# Знайдемо y_max та y_min
y_max = 200 + int(x_aver_max)
y_min = 200 + int(x_aver_min)


def regression(x, b):
    """
    Функція для знаходження регресії
    """
    y = sum([x[i]*b[i] for i in range(len(x))])
    return y


def plan_matrix(n, m):
    """
    Функція для знаходження матриці планування
    """
    y = np.zeros(shape=(n,m))
    for i in range(n):
        for j in range(m):
            y[i][j] = random.randint(y_min, y_max)
    x_norm = np.array([[1, -1, -1, -1],
                       [1, -1,  1,  1],
                       [1,  1, -1,  1],
                       [1,  1,  1, -1],
                       [1, -1, -1,  1],
                       [1, -1,  1, -1],
                       [1,  1, -1, -1],
                       [1,  1,  1,  1]])
    x_norm = x_norm[:len(y)]

    x = np.ones(shape=(len(x_norm), len(x_norm[0])))
    for i in range(len(x_norm)):
        for j in range(1, len(x_norm[i])):
            if x_norm[i][j] == -1:
                x[i][j] = x_range[j-1][0]
            else:
                x[i][j] = x_range[j-1][1]

    print('\nМатриця планування')
    matrix = np.concatenate((x,y),axis=1)
    table = p.PrettyTable()
    
    yFieldNames = []
    yFieldNames += (f'Y{i+1}' for i in range(m))

    fieldNames = ["X0c", "X1c", "X2c", "X3c"] + yFieldNames
    table.field_names = fieldNames

    for i in range(len(matrix)):
        table.add_row(matrix[i])

    print(table)

    return x, y, x_norm


def find_coefficient(x, y_aver, n):
    """
    Функція для знаходження коефіціентів B
    """
    mx1 = sum(x[:, 1]) / n
    mx2 = sum(x[:, 2]) / n
    mx3 = sum(x[:, 3]) / n
    my = sum(y_aver) / n
    a12 = sum([x[i][1] * x[i][2] for i in range(len(x))]) / n
    a13 = sum([x[i][1] * x[i][3] for i in range(len(x))]) / n
    a23 = sum([x[i][2] * x[i][3] for i in range(len(x))]) / n
    a11 = sum([i ** 2 for i in x[:, 1]]) / n
    a22 = sum([i ** 2 for i in x[:, 2]]) / n
    a33 = sum([i ** 2 for i in x[:, 3]]) / n
    a1 = sum([y_aver[i] * x[i][1] for i in range(len(x))]) / n
    a2 = sum([y_aver[i] * x[i][2] for i in range(len(x))]) / n
    a3 = sum([y_aver[i] * x[i][3] for i in range(len(x))]) / n

    X = [[  1, mx1, mx2, mx3],
         [mx1, a11, a12, a13], 
         [mx2, a12, a22, a23], 
         [mx3, a13, a23, a33]]

    Y = [my, a1, a2, a3]
    # Вирішимо систему рівнянь для коефіціентів
    B = [round(i, 2) for i in solve(X, Y)]
    print('\nРівняння регресії')
    print(f'{B[0]} + {B[1]}*x1 + {B[2]}*x2 + {B[3]}*x3')

    return B


def s_kv(y, y_aver, n, m):
    """
    Функція для знаходження квадратної дисперсії
    """
    res = []
    for i in range(n):
        s = sum([(y_aver[i] - y[i][j])**2 for j in range(m)]) / m
        res.append(s)
    return res


def kriteriy_cochrana(y, y_aver, n, m):
    """
    Обчислення коефіцієнту Gp. 
    Коефіцієнт Gp показує, яку частку в загальній сумі дисперсій у рядках має максимальна з них.
    """
    S_kv = s_kv(y, y_aver, n, m)
    Gp = max(S_kv) / sum(S_kv)
    print('\nПеревірка за критерієм Кохрена')
    return Gp


def bs(x, y, y_aver, n):
    """
    Функція для знаходження оцінки коефіціентів (Betta s)
    """
    res = [sum(1 * y for y in y_aver) / n]
    for i in range(3): # 3 - ксть факторів (без X0)
        b = sum(j[0] * j[1] for j in zip(x[:,i], y_aver)) / n
        res.append(b)
    return res


def kriteriy_studenta(x, y, y_aver, n, m):
    """
    Функція для знаходження критерія стьюдента
    """
    S_kv_b = s_kv(y, y_aver, n, m)
    S_kv_b_aver = sum(S_kv_b) / n
 
    # Статиcтична оцінка дисперсії
    s_Bs = (S_kv_b_aver / (n * m)) ** 0.5
    Bs = bs(x, y, y_aver, n)

    ts = [abs(B) / s_Bs for B in Bs]

    return ts


def kriteriy_fishera(y, y_aver, y_new, n, m, d):
    """
    Функція для знаходження критерія фішера
    """
    S_kv_ad = (m / (n - d)) * sum([(y_new[i] - y_aver[i])**2 for i in range(len(y))])
    S_kv_b = s_kv(y, y_aver, n, m)
    S_kv_b_aver = sum(S_kv_b) / n

    return S_kv_ad / S_kv_b_aver


def cohren(f1, f2, q=0.05):
    """
    Функція для знаходження табличного значення критерія кохрена
    """
    q1 = q / f1
    fisher_value = f.ppf(q=1 - q1, dfn=f2, dfd=(f1 - 1) * f2)

    return fisher_value / (fisher_value + f1 - 1)


def main(n, m):
    tprint("Lab 3")
    tprint("Danilyuk  Denis")

    f1 = m - 1
    f2 = n
    f3 = f1 * f2
    q = 0.05

    ###
    # Табличні значення

    # (1 + 0.95) / 2 використовується для знаходження значення
    # таблиці стьюдента при рівні значимості q = 0.05
    t_student = t.ppf(df = f3, q = (1 + 0.95) / 2)

    G_kr = cohren(f1, f2)
    ###

    x, y, x_norm = plan_matrix(n, m)
    y_aver = [round(sum(i) / len(i), 2) for i in y]

    B = find_coefficient(x, y_aver, n)

    Gp = kriteriy_cochrana(y, y_aver, n, m)
    print(f'Gp = {Gp}')
    if Gp < G_kr:
        print(f'З ймовірністю {1-q} дисперсії однорідні.')
    else:
        print("Необхідно збільшити кількість дослідів")
        main(n, m + 1)


    ts = kriteriy_studenta(x_norm[:,1:], y, y_aver, n, m)
    print('\nКритерій Стьюдента:\n', [round(i, 4) for i in ts])

    res = [t for t in ts if t > t_student]
    final_k = [B[ts.index(i)] for i in ts if i in res]
    print('Коефіцієнти {} статистично незначущі, тому ми виключаємо їх з рівняння.'.format([i for i in B if i not in final_k]))

    
    y_new = []
    for j in range(n):
        y_new.append(regression([x[j][ts.index(i)] for i in ts if i in res], final_k))

    print(f'\nЗначення "y" з коефіцієнтами {final_k}')
    print(y_new)

    d = len(res)
    f4 = n - d
    F_p = kriteriy_fishera(y, y_aver, y_new, n, m, d)

    # Табличне значення
    f_t = f.ppf(dfn=f4, dfd=f3, q=1 - 0.05) 

    print('\nПеревірка адекватності за критерієм Фішера')
    print('Fp =', F_p)
    print('F_t =', f_t)
    if F_p < f_t:
        print('Математична модель адекватна експериментальним даним')
    else:
        print('Математична модель не адекватна експериментальним даним')



if __name__ == '__main__':
    main(4, 4)
