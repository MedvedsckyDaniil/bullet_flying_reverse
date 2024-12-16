# Обратная задача внешней баллистики
# Полёт пули

<div style="text-align: justify">

## Цели работы

По конечным параметрам с наложенным на них шумом восстановить начальные для прямой задачи полёта пути.

## Структура проекта

#### Папки

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

`/main_reverse.py` – файл с текущим кодом программы.

#### Функции `/main.py`

| Название функции                              | Описание функции                                                                                                          |
|-----------------------------------------------|---------------------------------------------------------------------------------------------------------------------------|
| `def transpose(a)`                            | Функция принимает матрицу a и возвращает её транспонированную версию.                                                     |
| `def mul_mat(a, b)`                           | Функция реализует перемножение двух матриц a и b.                                                                         |
| `def mul_const(a, b)`                         | Функция умножает матрицу a на скалярное значение b.                                                                       |
| `def kholetsky(matrix,y)`                     | Функция реализует метод Ходельского для разложения матриц.                                                                |
| `def compute_jacobi(functions, args, step)`   | Функция вычисляет матрицу Якоби для системы функций.                                                                      |
| `def g_funk(xdq) # 0-x, 1-speed, 2-adge`      | Функция вычисляет среднеквадратичное угловое отклонения по вертикали (см. математические выкладки / постановка задачи).   |
| `def g_funk_k(cords)`                         | Функция вычисляет арктангенс отношения y и x координат.                                                                   |
| `def cheb_norm(arr)`                          | Функция вычисляет норму Чебышева (максимальную) для массива arr.                                                          |
| `def func_norm_raspr_obr(r,mu,sigma)`         | Функция использует нормального распределения.                                                                             |

## Математические выкладки

#### Суть задачи

Суть обратной задачи состоит в восстановлении вектора p по набору значений F(x,p), где p – вектор начальных условий стрельбы и параметров внешней среды. <br/>
В прямой задаче нам дан весь ветор p.

|                                | Дано:                                                                                                                                                                                      | 
|--------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Характеристики тела            | m – масса пули; <br/>S – площадь миделева сечения тела;                                                                                                 |
| Характеристики системы отсчёта | y₀ – начальная высота пули; <br/>v₀ – начальная скорость пули; <br/>θ₀ – начальный угол между вектором скорости и плоскостью горизонта; <br/>l – горизонтальное расстояние до препятствия; | 
| Характеристики препятствия     | ρ – плотность препятствия; <br/>σ – предельное напряжение для среды.                                                                                                                       | 

p =  (y₀, v₀, θ₀, l, ρ, σ) <br/>
В обратной же задаче даны только l, ρ, σ, <br/>
а y₀, v₀, θ₀ необходимо восстановить из конечного положения пули.

#### Формулировка задачи

Можно предложить различные варианты постановки обратной задачи. <br/>
Пусть проведено K выстрелов. Для каждого выстрела известна пара значений (xₖ,yₖ), где xₖ – расстояние до мишени, а yₖ – вертикальная координата точки попадания. Каждому из  выстрелов соответствует вектор начальных данных и параметров атмосферы pₖ, известный с некоторой погрешностью. <br/>
pₖ =  (yₖ, vₖ, θₖ, l, ρ, σ) <br/>
Тогда задача формулируется как задача минимизации среднеквадратичного углового отклонения по вертикали <br/>
![Формула 11](/images/11.png) <br/>
по набору векторных параметров p₁,...,pₖ. Так как параметры l, ρ, σ имеют одинаковые и известные значения, pₖ-ые различаются только на неизвестные yₖ, vₖ, θₖ. Тогда предыдущую формулу можно конкретизировать как 
![Формула 12](/images/12.png) <br/>
где q =  (y₀, v₀, θ₀, l, ρ, σ).
Вектор q₀, минимизирующий среднеквадратичное угловое отклонение, даёт решение обратной задачи:
![Формула 13](/images/13.png) <br/>

#### Статистическая постановка задачи

Допустим, что при постановке задачи нам дополнительно известна плотность распределения вероятностей для каждого из неизвестных параметров y₀, v₀, θ₀. Обозначим эти плотности как f_y₀, f_v₀, f_θ₀. Также предполагается известной плотность распределения точек попадания по вертикали f_Ψ, которая характеризует рассеяние оружия и задается в угловых единицах. Предположив, что рассмотренные нами факторы, влияющие на результат стрельбы, независимы друг от друга, получим совокупную плотность распределения вероятностей <br/>
![Формула 14](/images/14.png) <br/>
В соответствии с принципом максимального правдоподобия наиболее вероятный набор параметров y₀, v₀, θ₀, l, ρ, σ соответствует максимуму функции совокупной плотности распределения f. <br/>
Предположение о независимости факторов влияния (определяющих факторов задачи) очень важно, поскольку если оно не выполняется, то вместо формулы (14) следует писать более сложное выражение. Это приведет к тому, что в алгоритм решения обратной задачи придется вносить серьезные изменения, которые могут сказаться, например, на его устойчивости. Сама идея использования принципа максимального правдоподобия остается, но ее реализация сильно усложняется. <br/>
Существуют и другие подходы, заключающиеся в поиске комбинаций определяющих факторов. Вопрос их сравнительной эффективности требует серьезных исследований и выходит за рамки данной работы. <br/>
Задача легко решается для случая, когда все случайные величины распределены нормально. Удобнее вместо максимума функции f искать максимум 
![Формула 15](/images/15.png) <br/>

ǭ <br/>

![Формула 16](/images/16.png) <br/>

![Формула 17](/images/17.png) <br/>

![Формула 18](/images/18.png) <br/>

![Формула 19](/images/19.png) <br/>

![Формула 20](/images/20.png) <br/>


## Пример работы программы

↓ Изначальная скорость ↓ Изначальный угол в радианах 
```
0 1002.5274308557114 -0.008550314285454351 
1 1064.0987352652594 -0.03753359288801369 
2 980.1790007670896 0.015216923424321986 
3 1019.1834340151822 -0.0194959647646088 
4 994.6724282973784 -0.0006616966551808199 
5 1056.444923421869 -0.03511132022372046 
6 971.2689634818145 0.029794168283288326 
7 1010.5054004300816 -0.015356461311691752 
8 992.7400490632452 0.0028439474443737347 
9 1001.5486289289053 -0.00967445679437939
0 1.4266188370360868
1 0.8466029615831104
2 1.9019905837702957
3 1.2076621878018214
4 1.5844009127383722
5 0.8951155473266885
6 2.193686574544111
7 1.2904679624936644
8 1.6545215464998995
9 1.4041214778054267
```
↑ Вертикальная координата точки попадания

Соответсвующий значению 1.4266188370360868 вывод прямой задачи:
```
air on
air: adge= -0.5011721306219734 (degress), y= 1.4270270047838414, x= 20.0, speed= 994.1572920953475
block: adge= -0.3579718390669916 (degress), y= 1.426952049627081, x= 20.009799999999977, speed= 0
779.9196243286133
```
![1.4266188370360868.jpg](/images/1.4266188370360868.jpg)

1.4266 ~ 1.4270

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

</div>
