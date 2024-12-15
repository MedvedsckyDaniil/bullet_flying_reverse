# Обратная задача внешней баллистики
# Полёт пули

## Цели работы

Изучить и смоделировать полёт пули в зависимости от заданных начальных условий и встречаемых её на своём пути препятствий.

## Структура проекта

#### Папки

`/build` – автоматически сгенерированная PyCharm-ом папка с единственным файлом `/build/setup.py`.

`/butcher` – папка с текстовыми файлами, содержащими таблицы Бутчера:
* `/butcher/midpoint` – таблица Бутчера для метода средней точки;
* `/butcher/RK4` – таблица Бутчера для метода Рунге-Кутты 4-ого порядка;
* `/butcher/RK5` – таблица Бутчера для метода Рунге-Кутты 5-ого порядка;
* `/butcher/DP5` – таблица Бутчера для метода Дормана-Принца 5-ого порядка;
* `/butcher/DP8` – таблица Бутчера для метода Дормана-Принца 8-ого порядка.

Так же папка `/butcher` содержит изображение `/butcher/butcher_tables.jpg` для более наглядного сравнения таблиц.

![butcher_tables.jpg](/butcher/butcher_tables.jpg)

#### Файлы

`/main.py` – файл с кодом программы прямой задачи внешней баллистики.

`/reverse.py` – файл с текущим кодом программы.

#### Функции `/main.py`

| Название функции                                                    | Описание функции                                                                                                                                                                   |
|---------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `def transpose(a)`                                         | Функция преобразовывает файл с таблицей Бутчера в np.array и возвращает его                                                                                                                   |
| `def mul_mat(a, b)`                                                       | cx – вспомогательная функция закона сопротивления воздуха. <br/>Возвращает коэффициент, необходимый `def Force(F_para, S, speed)`                                                             |
| `def mul_const(a, b)`                                       | Функция рассчитывает и возвращает модуль силы лобового сопротивления воздуха <br/>(см. Математические выкладки  -> Сопротивление воздуха)                                                     |
| `def compute_jacobi(functions, args, step)`    | Функция пошагового расчёта траектории пули в препятствии <br/>(см. Математические выкладки  -> Рассматриваемые системы дифференциальных уравнений). <br/>Возвращает массивы времени и высоты. |
| `def g_funk(xdq) # 0-x, 1-speed, 2-adge`                                             | Вспомогательная функция для рисоврения графиков. Она используется чтобы указать на графике максимум. В случае если был номер иттерации не совпадает с переданным значением x возвращается 0   |
| `def g_funk_k(cords)`                                                    | Функция задаёт все начальные характеристики тела, системы отсчёта и препятствия, запускает полёт в воздухе до препятствия, запускает встречу с препятствием, отрисовывает графики             |
| `def cheb_norm(arr)`                                                    | |
| `def func_norm_raspr_obr(r,mu,sigma)`                                                    | |

## Математические выкладки

#### Формулировка задачи

p – вектор начальных условий стрельбы и параметров внешней среды. В расчетах учитываются начальная высота пули, начальная скорость пули, начальный угол между вектором скорости и плоскостью горизонта, плотность препятствия, предельное напряжение для среды, поэтому <br/>
p =  (y₀, v₀, θ₀, ρ, σ)

Суть обратной задачи состоит в восстановлении вектора  по набору значений , где .

|                                | Дано:                                                                                                                                                                                      | 
|--------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Характеристики тела            | m – масса пули; <br/>S – площадь миделева сечения тела; <br/>v₀ – начальная скорость пули;                                                                                                 |
| Характеристики системы отсчёта | y₀ – начальная высота пули; <br/>v₀ – начальная скорость пули; <br/>θ₀ – начальный угол между вектором скорости и плоскостью горизонта; <br/>l – горизонтальное расстояние до препятствия; | 
| Характеристики препятствия     | ρ – плотность препятствия; <br/>σ – предельное напряжение для среды.                                                                                                                       | 


Задача: построить траекторию движения пули.



#### 


#### 



## Некоторые результаты работы программы

Демонстрация траекторий при выстреле с воздухом и без. Вертикальная черта указывает на наивысшую точку.


***

## Список использованных источников

1) Козлитин И.А. – Восстановление входных параметров расчета внешней баллистики тела по результатам траекторных измерений. URL: <br/>
https://www.mathnet.ru/links/f9757b34505280236c2be0aa2743fa93/mm3892.pdf

2) Ефремов А. К. – Аппроксимация закона сопротивления воздуха 1943 г. <br/>
https://cyberleninka.ru/article/n/approksimatsiya-zakona-soprotivleniya-vozduha-1943-g/viewer

    2.1) Точные значения функции закона сопротивления cx "образца 1943 г." <br/>
    https://popgun.ru/viewtopic.php?p=8041476

3) Кривченко А.Л. – Ударно-волновые процессы взаимодействия Высокоскоростных элементов с конденсированными средами. URL: <br/>
http://d21221701.samgtu.ru/sites/d21221701.samgtu.ru/files/aleksenceva_dis.pdf
