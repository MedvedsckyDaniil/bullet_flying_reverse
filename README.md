# Обратная задача внешней баллистики
# Полёт пули

<div align="justify">

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

`/images` – папка с изображениями, используемыми в README.md.

#### Файлы

`/main.py` – файл с кодом программы прямой задачи внешней баллистики.

`/main_reverse.py` – файл с текущим кодом программы.

#### Функции `/main.py`

| Название функции                              | Описание функции                                                                                                                                  |
|-----------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------|
| `def transpose(a)`                            | Функция принимает матрицу a и возвращает её транспонированную версию. Данное действие происходит с помощью срезов                                 |
| `def mul_mat(a, b)`                           | Функция реализует перемножение двух матриц a и b. Умножение происходит по правилу строка умножить(скалярное произведений) на столбец              |
| `def mul_const(a, b)`                         | Функция умножает матрицу a на значение b.                                                                                                         |
| `def kholetsky(matrix,y)`                     | Функция реализует метод Ходельского для разложения матриц и решения слау. Для этого правая часть раскладывается как A=LL^(-1) где L нижнетреугольная матрица |
| `def compute_jacobi(functions, args, step)`   | Функция вычисляет значения матрицы Якоби в необходимой точке для системы функций.                                                                 |
| `def g_funk(xdq) # 0-x, 1-speed, 2-adge`      | Функция вычисляет арктангенс отношения y полученного при переданных начальных параметрах в прямую задачу и x координат.                           |
| `def g_funk_k(cords)`                         | Функция вычисляет арктангенс отношения y и x координат.                                                                                           |
| `def cheb_norm(arr)`                          | Функция вычисляет норму Чебышева (максимальную по модул.) для массива arr.                                                                        |
| `def func_norm_raspr_obr(r,mu,sigma)`         | Функция вычисляет обратную функцию нормального распределения.                                                                                     |

#### Описание основного кода программы

Инициализируем начальные условия. Скорость и наклон задаются с помощью нормального распределения. <br>
Затем происходит инициальзация изначального gk в нем содержатся арктангенсы отношений координат попадания в мишени. Число строк это число серий выстрелов, а число столбцов – число выстрелов в серии. В нашем случае это вектор 1х10. <br>
Далее инициализируется а соответсвующее g(0) (запуску прямой задачи при начальных значениях) <br>
Затем в условии цикла while проверяется условие на то, что мы нашли ответ с определенной точностью левая и правая часть условия находятся в конце цикла. Затем происходит проверка на то, не увеличилась ли чебышева норма разности изначальных параметров и текущих. Если нет, то вычисляется новое gk(сделано для цикла), после чего находятся стандартные отклонения по строкам gk(sigma_m) и величине пси(оценка совокупного рассеяния). Затем находится вектор Ge с помощью умножения матрицы, состоящей из gk на ветор столбец, каждая строка при этом делится на число выстрелов в серии (тем самым мы усредняем значения). После этого находится матрица омега (в коде sigma_q_aprior). Это квадратная матрица, у которой на главной диагонали находятся стандартные отклонения по параметрам, возведенным в -2 степень. После этого находится матрица Якоби. Для этого в функцию compute_jacobi передается функция расчета прямой задачи, затем ее аргументы (текущие) и шаг иттерации, это происходит для всех выстрелов в сериях. После этого находятся левые и правые часли уравнения для поиска dq и затем с помощью разложения Холецкого проиходит поиск необходимого вектора. Затем, если чебышева норма не увеличилась, мы присваиваем новые значения (добавляем dq к старому q), а если нет, то возвращаем старые значения. Цикл повторяется до тех пор, пока не будут найдены необходимые значения или пока не будет достигнуто огранечения иттераций. Программа находит решение за 1-5 шагов при близких значениях и >100 в остальных случаях. <br>

## Математические выкладки

#### Суть задачи

Суть обратной задачи состоит в восстановлении вектора p по набору значений F(x,p), где p – вектор начальных условий стрельбы. <br/>

|                                | Дано:                                                                                                                                                                                      | 
|--------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Характеристики тела            | m – масса пули; <br/>S – площадь миделева сечения тела;                                                                                                 |
| Характеристики системы отсчёта | y₀ – начальная высота пули; <br/>v₀ – начальная скорость пули; <br/>θ₀ – начальный угол между вектором скорости и плоскостью горизонта; <br/>l – горизонтальное расстояние до препятствия; | 

p =  (v₀, θ₀) <br/>
В обратной же задаче даны только l, y <br/>
а v₀, θ₀ необходимо восстановить из конечного положения пули. <br/>

#### Формулировка задачи

Можно предложить различные варианты постановки обратной задачи. <br/>
Пусть проведено K выстрелов. Для каждого выстрела известна пара значений (xₖ,yₖ), где xₖ – расстояние до мишени, а yₖ – вертикальная координата точки попадания. Каждому из  выстрелов соответствует вектор начальных данных pₖ, известный с некоторой погрешностью. <br/>
pₖ =  (vₖ, θₖ) <br/>
Тогда задача формулируется как задача минимизации среднеквадратичного углового отклонения по вертикали <br/>
![Формула 11](/images/11.png) <br/>
по набору векторных параметров p₁,...,pₖ. Так как параметр l имеет одинаковое и известное значение, pₖ-ые различаются только на неизвестные vₖ, θₖ. Тогда предыдущую формулу можно конкретизировать как <br/>
![Формула 12](/images/12.png) <br/>
где q =  (v₀, θ₀).
Вектор q₀, минимизирующий среднеквадратичное угловое отклонение, даёт решение обратной задачи: <br/>
![Формула 13](/images/13.png) <br/>

#### Статистическая постановка задачи

Допустим, что при постановке задачи нам дополнительно известна плотность распределения вероятностей для каждого из неизвестных параметров v₀, θ₀. Обозначим эти плотности как f_v₀, f_θ₀. Также предполагается известной плотность распределения точек попадания по вертикали f_Ψ, которая характеризует рассеяние оружия и задается в угловых единицах. Предположив, что рассмотренные нами факторы, влияющие на результат стрельбы, независимы друг от друга, получим совокупную плотность распределения вероятностей <br/>
![Формула 14](/images/14.png) <br/>
В соответствии с принципом максимального правдоподобия наиболее вероятный набор параметров v₀, θ₀, l соответствует максимуму функции совокупной плотности распределения f. <br/>
Предположение о независимости факторов влияния (определяющих факторов задачи) очень важно, поскольку если оно не выполняется, то вместо формулы (14) следует писать более сложное выражение. Это приведет к тому, что в алгоритм решения обратной задачи придется вносить серьезные изменения, которые могут сказаться, например, на его устойчивости. Сама идея использования принципа максимального правдоподобия остается, но ее реализация сильно усложняется. <br/>
Существуют и другие подходы, заключающиеся в поиске комбинаций определяющих факторов. Вопрос их сравнительной эффективности требует серьезных исследований и выходит за рамки данной работы. <br/>
Задача легко решается для случая, когда все случайные величины распределены нормально. Удобнее вместо максимума функции f искать максимум <br/>
![Формула 15](/images/15.png) <br/>
где ![Формула 15.1](/images/15.1.png) (qᵢср.) – априорные значения параметров, а σ_qᵢ – их стандартное отклонение, σ_Ψ – стандартное отклонение точек попадания, вызванное рассеением. При этом ǭ = {qᵢср.} <br/>
Обозначив Ψ*(q) = -In f(q), запишем статическую подстановку обратой задачи. <br/>
![Формула 16](/images/16.png) <br/>
Вектор параметров q₀, на котором достигается минимум по этой формуле, будет наиболее вероятным для набора исходных данных (xₖ,yₖ) и является решением обратной задачи. <br/>
Задача q₀ в статистической постановке имеет преимущество над задачей q₀ при обычной её формулировке: при наличии двух разных решений задачи q₁ и q₂, дающих равные Ψ(q₁) и Ψ(q₂) (Ψ(q₁) == Ψ(q₂)), они, скорее всего, будут иметь различные вероятности реализации, а значит, из них будет выбрано решение с большей вероятностью. Таким образом, использование априорной информации о решении существенно сужает множество решений обратной задачи, хотя и не гарантирует единственность ее решения. <br/>

#### Решение обратной задачи в статистической постановке

Для нахождения максимума In f рассмотрим <br/>
![Формула 17](/images/17.png) <br/>
Приравнивая частные производные к нулю, получим <br/>
![Формула 18](/images/18.png) <br/>
Выражение можно записать в векторной форме <br/>
![Формула 19](/images/19.png) <br/>
где ![Формула 19.1](/images/19.1.png) – матрица Якоби, а Ω – матрица, у которой на главной диагонали значения {(σ_qᵢ)⁻²}, а остальные элементы нулевые. Данная формула является нелинейным векторным уравнением относительно неизвестного вектора Δq, которое решается методом Ньютона. <br/>
Естественно взять нулевое начальное приближение. Пренебрегая нелинейностью g(Δq) в окрестности 0, запишем формулу первой итерации метода Ньютона-Гаусса <br/>
![Формула 20](/images/20.png) <br/>
Далее вычисляем q = Δq + ǭ. Это q становится новым ǭ и процесс повторяется. <br/>
При расчете по последней формуле наиболее трудоемкой операцией является вычисление матрицы Якоби  δg/δq. Время ее расчета существенно сокращается в случае, когда K выстрелов можно разделить на M групп, в каждой из которой условия стрельбы одинаковы, а различие точек попадания вызвано исключительно рассеянием оружия (например, несколько последовательных выстрелов по одной мишени в течение короткого времени). Тогда вместо K строк матрица будет содержать M строк, что дает существенную экономию при большом размере каждой из групп. Разбиение на группы также удобно тем, что внутри каждой из них можно провести оценку рассеяния, а усреднив эту оценку по всем группам, получить оценку совокупного рассеяния σ_Ψ , как было описано выше. <br/>
Пусть все M групп имеют одинаковое число элементов N, тогда матрица Якоби δg/δq будет содержать M строк. Элементы исходного вектора g₀ разумно поместить в матрицу G_(M×N), каждая строка которой соответствует одной группе. Для этого случая формула примет вид <br/>
![Формула 21](/images/21.png) <br/>
где e – вектор-столбец длины N. Как легко видеть, формула 1/N * Ge производит осреднение элементов матрицы G по строкам. Множитель 1/N перед (σ_Ψ)² появляется из-за того, что стандартное отклонение средней точки попадания в группе из N выстрелов будет в √(N) раз меньше стандартного отклонения σ_Ψ одиночных попаданий. Величина σ_Ψ берется из таблиц стрельбы, либо вычисляется по формуле <br/>
![Формула 22](/images/22.png) <br/>
где σ_m – оценка стандартного отклонения по выборке, состоящей из элементов строки G_m, вычисляемая следующим образом: <br/>
![Формула 23](/images/23.png) <br/>
<br/>
![Формула 29](/images/29.png) <br/>

## Формулы и их расчёт в коде

| Формулы                        | Место в коде, где реализовано их вычисление            | 
|--------------------------------|--------------------------------------------------------|
| ![Формула 21](/images/21.png)  | Строки 190-194 <br/> ![190-194](/images/190-194.png)   |
| Ge усредненная по строкам матрица арктангенсов(у/х)   | Строки 170-173 <br/> ![170-173](/images/170-173.png)                 |
| ![Формула 22](/images/22.png)  | Строка 168     <br/> ![168](/images/168.png)           |
| ![Формула 23](/images/23.png)  | Строки 160-166 <br/> ![160-166](/images/160-166.png)   |
| ![Формула 29](/images/29.png)  | Строки 201-203 <br/> ![201-203](/images/201-203.png)   | 


## Пример работы программы

↓ Изначальная скорость ↓ Изначальный угол в радианах 
```
промежуточные результаты
0 912.2215541132144 -0.002607993873921007 
1 912.9118129417562 -0.04102408551685154 
2 910.769511175065 -0.047348154720878284 
3 912.46717116884 0.017855275409924834 
4 911.8143741871042 -0.010346075594626592 
5 913.4382279792203 1.5522524588677356e-05 
6 911.747963086093 -0.018490637738238457 
7 913.2475900041912 0.003533286530811492 
8 912.4748031542522 -0.01479214563572712 
9 911.6026028838556 -0.018037512531428032 
itteration 0
finding jacobi
cholesky method
0 930.2512602308165 -0.002607994296077368 
1 948.3882023371616 -0.041024087168613076 
2 952.9820942807985 -0.04734815708768483 
3 926.7847341088652 0.01785527514506051 
4 931.8172239459013 -0.010346076115771383 
5 930.8855649204786 1.552213108030507e-05 
6 934.3630824321012 -0.01849063840593889 
7 929.9716457500224 0.003533286169293287 
8 933.8229278998967 -0.014792146228703033 
9 934.0544123946905 -0.018037513189731375 
itteration 1
finding jacobi
cholesky method
0 948.3217820515624 -0.0026079942961116397 
1 984.1798842851208 -0.041024087168740744 
2 995.7299597267524 -0.047348157087864495 
3 941.1227041210276 0.017855275145038763 
4 951.8758674740785 -0.01034607611581344 
5 948.3698819465776 1.5522131048305103e-05 
6 957.0589638789402 -0.018490638405992365 
7 946.7282589726352 0.003533286169263824 
8 955.2389359917353 -0.014792146228750677 
9 956.585237503792 -0.018037513189784127 
itteration 2
finding jacobi
cholesky method
0 966.3913121724794 -0.002607994296113424 
1 1019.9643809642605 -0.041024087168747045 
2 1038.4658613318268 -0.047348157087873176 
3 955.4601754466177 0.017855275145037618 
4 971.9331609505068 -0.010346076115815617 
5 965.8533027900778 1.5522131046635272e-05 
6 979.7529069354108 -0.01849063840599511 
7 963.4840814421166 0.003533286169262283 
8 976.6533118757197 -0.014792146228753134 
9 979.1141641886171 -0.018037513189786837 
itteration 3
finding jacobi
cholesky method
0 984.4599028522993 -0.0026079942961137435 
1 1055.7424174181638 -0.04102408716874811 
2 1081.1912200598588 -0.047348157087874626 
3 969.7971688903077 0.01785527514503741 
4 991.9891832732536 -0.010346076115816004 
5 983.3358730961402 1.5522131046335887e-05 
6 1002.4450393566294 -0.018490638405995595 
7 980.2391517511999 0.0035332861692620064 
8 998.0661571584508 -0.014792146228753569 
9 1001.6413167239533 -0.018037513189787316 
itteration 4
finding jacobi
cholesky method
конечный результат
0 1002.5276054015183 -0.002607994296113837 
1 1091.5146445195044 -0.04102408716874841 
2 1123.907269298532 -0.04734815708787502 
3 984.1337054985746 0.017855275145037347 
4 1012.0440109028251 -0.010346076115816116 
5 1000.8176378658115 1.552213104624796e-05 
6 1025.135482985498 -0.018490638405995734 
7 996.9935081824668 0.003533286169261925 
8 1019.4775694832845 -0.014792146228753696 
9 1024.166813733558 -0.018037513189787455 
0 1.5454693460795152
1 0.7766866417828686
2 0.6499450507517444
3 1.954773340494417
4 1.3906980878932456
5 1.59794612653024
6 1.2277711706016532
7 1.668300682816293
8 1.3017656629973176
9 1.236835980053562
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
</div>
