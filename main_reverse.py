import numpy as np

from main import *

def transpose(a):
    c=np.zeros((a.shape[1],a.shape[0]))
    for i in range(a.shape[1]):
        c[i,:]=a[:,i]
    return c

def mul_mat(a, b):
    c=np.zeros((a.shape[0],b.shape[1]))
    for i in range(a.shape[0]):
        for j in range(b.shape[1]):
            c[i,j]=np.dot(a[i,:],b[:,j])
    return c

def mul_const(a, b):
    c=np.zeros(a.shape)
    c[:,:]=np.dot(a[:,:],b)
    return c

def add_mat(a, b):
    c=np.zeros(a.shape)
    c[:, :] = a[:, :]+ b[:,:]
    return c

def kholetsky(matrix,y):
    tp=np.zeros((matrix.shape[0],matrix.shape[1]+y.shape[1]))
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            tp[i,j]=matrix[i,j]
        tp[i,-1]=y[i,-1]
    matrix=tp
    n = len(matrix)
    # Формирование матрицы t элементов
    t_arr = np.zeros((matrix.shape))

    t_arr[0] = matrix[0] / np.sqrt(matrix[0][0])
    for i in range(1, n):
        t_arr[i][i] = np.sqrt(matrix[i][i] - (t_arr[:, i] ** 2).sum())
        t_arr[i, i + 1:] = (matrix[i, i + 1:] - (t_arr[:i, i].reshape(-1, 1) * t_arr[:i, i + 1:]).sum(0)) / t_arr[i][
            i]

    # Находим решение
    b = t_arr[:, n].reshape(-1, 1)
    t_arr = t_arr[:, :n]
    x_arr = np.zeros((n, 1))
    for i in range(1, n + 1):

        x_arr[-i, 0] = (b[-i] - (t_arr[-i, -i:] * x_arr[-i:, 0]).sum())[0] / t_arr[-i, -i]
    return x_arr

def compute_jacobi(functions, args, step):
    func_count = len(functions)
    var_count = len(args)

    jacobi = np.zeros((func_count, var_count))

    for i in range(func_count):
        cur_f_value = functions[i](args)
        for j in range(var_count):
            unit_vector = np.zeros(var_count)
            unit_vector[j] = 1
            jacobi[i, j] = (functions[i](args + step * unit_vector) - cur_f_value) / step

    return jacobi[:,:-1]#по х возвращать не надо

def g_funk(xdq):#0-x,1-speed,2-adge
    mass = 0.0061
    S = 6.6 * 10 ** (-5)
    startX=0
    startY=1.6
    l=20
    butcher = parse('DP8')
    F_para=True
    return np.arctan(air_K(mass, S, startX, startY, xdq[1], xdq[2], l, butcher, F_para)[2]/xdq[0])
def g_funk_k(cords):
    return np.arctan(cords[1]/cords[0])

def cheb_norm(arr):
    tmp = np.abs(arr)
    return np.max(tmp)

def func_norm_raspr_obr(r,mu,sigma):
    return mu-np.sqrt(-2*sigma**2*np.log(sigma*np.sqrt(2*np.pi)*r))


def inverse_matrix(matrix_origin):
    n = matrix_origin.shape[0]
    m = np.hstack((matrix_origin, np.eye(n)))

    for nrow, row in enumerate(m):
        divider = row[nrow]
        row /= divider
        for lower_row in m[nrow + 1:]:
            factor = lower_row[nrow]
            lower_row -= factor * row
    for k in range(n - 1, 0, -1):
        for row_ in range(k - 1, -1, -1):
            if m[row_, k]:
                m[row_, :] -= m[k, :] * m[row_, k]
    return m[:, n:].copy()

adge= -0.01339246052355161
y= 1.5976693190983655
x= 20
speed= 912.0705826611311

param_M=1#число серий
num_of_exp_N=10#число выстрелов в серии
param_q=2#скорость, угол

cords = np.zeros((2, num_of_exp_N))

q_aprior = np.zeros((param_q, num_of_exp_N)) #2 число исследуемых параметров, 10 число результатов
q_aprior_buff = np.zeros((param_q, num_of_exp_N)) #2 число исследуемых параметров, 10 число результатов

sigma = 1    # Стандартное отклонение
size = 10  # Количество сгенерированных значений

q_aprior[0]=np.random.normal(speed, sigma, size)             #data_speed
q_aprior[1]=np.random.normal(adge, sigma, size)              #data_adge
q_aprior[1]=np.deg2rad(q_aprior[1])
mass = 0.0061
S = 6.6 * 10 ** (-5)
startX = 0
startY = 1.6
l = 20
butcher = parse('DP8')
F_para = True

cords[0]=x                                                                                     #data_x
for i in range(size):
    cords[1,i]= air_K(mass, S, startX, startY, q_aprior[0,i], q_aprior[1,i], l, butcher, F_para)[2]            #data_y

check1=0
check2=0
itter=0

gk = np.zeros((1, num_of_exp_N))
gk[:] = g_funk_k(cords)
a = g_funk([cords[0, 0], q_aprior[0, 0], q_aprior[1, 0]])
cheb=float('inf')
print("промежуточные результаты")
while(check1>=check2 and itter<5):
    for i in range(num_of_exp_N):
        print(i,end=" ")
        for j in range(param_q):
            print(q_aprior[j,i],end=" ")
        print("")
    ch=cheb_norm(a-gk)
    if (cheb>=ch):
        cheb= cheb_norm(a-gk)
        print(f"itteration {itter}")

        gk=np.zeros((1, num_of_exp_N))
        gk[:]=g_funk_k(cords)

        mu_m = np.zeros((param_M, 1))
        for i in range(param_M):
            mu_m[i]= np.sum(gk[i]) / num_of_exp_N

        sigma_m = np.zeros((param_M, 1))
        for i in range(param_M):
            sigma_m[i]=(np.sum((gk[i]-mu_m[i])**2))/(num_of_exp_N - 1)

        sigma_psi= np.sum(sigma_m[:]) / param_M

        e=np.zeros((num_of_exp_N, 1))
        e[0,0]=1
        Ge=mul_mat(gk,e)
        Ge=mul_const(Ge, 1 / num_of_exp_N)


        sigma_q_aprior=np.zeros((param_q, param_q))

        mu_q = np.zeros((param_q, 1))
        for i in range(param_q):
            mu_q[i]= np.sum(q_aprior[i]) / num_of_exp_N

        for i in range(param_q):
            sigma_q_aprior[i,i]=((np.sum((q_aprior[i]-mu_q[i])**2))/(num_of_exp_N - 1))**(-2)

        step = 1e-5
        functions=[g_funk]
        print("finding jacobi")
        jacobi=[compute_jacobi(functions, [cords[0,i],q_aprior[0,i],q_aprior[1,i]], step) for i in range(num_of_exp_N)]

        dq=[mul_mat(-1*transpose(jacobi[i]),(a-Ge)) for i in range(num_of_exp_N)]

        tmp =[param_M/num_of_exp_N*(sigma_psi**2)*sigma_q_aprior+mul_mat(transpose(jacobi[i]),jacobi[i]) for i in range(num_of_exp_N)]
        print("cholesky method")
        dq = [kholetsky(tmp[i],dq[i]) for i in range(num_of_exp_N)]
        q_aprior_buff=q_aprior.copy()
        for i in range(num_of_exp_N):
            q_aprior[0, i] = dq[i][0][0] + q_aprior[0,i]
            q_aprior[1, i] = dq[i][1][0] + q_aprior[1,i]


        check1=np.sqrt(num_of_exp_N)*cheb_norm(a-gk)/sigma_psi
        alpha=0.09
        check2=func_norm_raspr_obr((((1-alpha)**(1/param_M)+1)/2),mu_m,sigma_psi)#альфа это вероятномть пропуска цели
        itter+=1
    else:
        q_aprior=q_aprior_buff
        break

print("конечный результат")
for i in range(num_of_exp_N):
    print(i,end=" ")
    for j in range(param_q):
        print(q_aprior[j,i],end=" ")
    print("")

for i in range(num_of_exp_N):
    print(i, cords[1,i])
