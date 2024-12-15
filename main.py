import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.patches import *
import time


def parse(file_name: str):
    file = open(file_name, 'r')
    butcher = np.zeros((16, 16))
    j = 0
    while file:
        line = file.readline()
        if line == "":
            break
        tmp = line.split()
        for i in range(len(tmp)):
            if '/' in tmp[i]:
                ch, zn = tmp[i].split('/')
                butcher[j, i] = float(ch) / float(zn)
            else:
                butcher[j, i] = float(tmp[i])
        j += 1
    butcher[-1] = butcher[j - 1]
    butcher[j - 1] = np.zeros(16)
    return butcher


def cx(mah):
    c_x = 0
    if mah < 0.73:
        c_x = 0.157
    elif mah < 0.82:
        c_x = 0.033 * mah + 0.133
    elif mah < 0.91:
        c_x = 0.161 + 3.9 * (mah - 0.823) ** 2
    elif mah < 1:
        c_x = 1.5 * mah - 1.176
    elif mah < 1.18:
        c_x = 0.386 - 1.6 * (mah - 1.176) ** 2
    elif mah < 1.62:
        c_x = 0.384 * np.sin(1.85 / mah)
    elif mah < 3.06:
        c_x = 0.29 / mah + 0.172
    elif mah < 3.53:
        c_x = 0.316 - 0.016 * mah
    else:
        c_x = 0.259
    # return c_x # (если без нормирования)
    if mah < 0.44:
        return 0.61 * c_x
    elif mah < 0.733:
        return 0.58 * c_x
    elif mah < 0.88:
        return 0.48 * c_x
    elif mah < 1:
        return 0.6 * c_x
    elif mah < 1.173:
        return 0.57 * c_x
    elif mah < 1.466:
        return 0.5 * c_x
    elif mah < 2.053:
        return 0.45 * c_x
    else:
        return 0.48 * c_x


def Force(F_para, S, speed):
    if F_para:
        i = 0.47    # i – коэффициент формы
        # cx – функция закона сопротивления воздуха;
        a = 335     # a – скорость звука (н.у. 335 м/сек);
        p = 1.2754  # ρ – плотность атмосферы (н.у. 1,2754 кг/м³).
        return i * S * p * speed ** 2 * cx(speed / a) / 2
    else:
        return 0



def air(mass, S, startX, startY, speed, teta, l, butcher, F_para):
    # Обработка начальных условий           # u0, γ0, y0, t0, v0
    Xspeed = speed * np.cos(teta)           # u0
    adge = np.tan(teta)                     # γ0
    y = startY                              # y0
    t = 0                                   # t0
    speed = speed                           # v0

    h = 0.002
    # h = 1
    bounds = 10000
    # Arrays
    ys = np.zeros(int(bounds / h))
    ys[0] = y
    ts = np.arange(0, bounds, h)
    ks = np.zeros((5,butcher.shape[1]))

    # Система дифференциальных уравнений:
    # du/dx = -F(v)/(m*v);
    # dγ/dx = -g/(u^2);
    # dy/dx = γ;
    # dt/dx = 1/u;
    # v = u * sqrt(1+γ^2)

    itter = 0
    while y > 0 and ts[itter] < l:          # (y > 0 && x < l)
        if ts[itter] + h > l:
            h = l-ts[itter]
        # Рассчёт k0, k1, k2, k3, k4:
        for i in range(15):
            #                    i-1
            # k[i] = f(x(t) + h * Σ a[i][j] * k[j])
            #                    j=0
            sp = speed + h * np.dot(ks[4, :i + 1], butcher[i, :i + 1])
            spx = Xspeed + h * np.dot(ks[0, :i + 1], butcher[i, :i + 1])
            ks[0, i] = (-Force(F_para,S, sp) / (mass * sp))                     # dx_speed  # u
            ks[1, i] = (-g / ((spx) ** 2))                                      # adge      # γ
            ks[2, i] = (adge + h * np.dot(ks[1, :i + 1], butcher[i, :i + 1]))   # y         # y
            ks[3, i] = 1 / (spx)                                                # t         # t
            ks[4, i] = (ks[0, i]) * np.sqrt(1 + (ks[1, i]) ** 2)                # speed     # v
        # Рассчёт x0, x1, x2, x3, x4:
        #                    len_
        # x(t+h) = x(t) + h * Σ a[len_][j] * k[j]
        #                    j=0
        Xspeed += h * np.dot(ks[0],butcher[-1]) # u
        adge += h * np.dot(ks[1],butcher[-1])   # γ
        y += h * np.dot(ks[2],butcher[-1])      # y
        t += h * np.dot(ks[3],butcher[-1])      # t
        speed += h * np.dot(ks[4],butcher[-1])  # v

        itter += 1
        if y >= 0 and itter < int(bounds / h):
            ys[itter] = y
        else:
            break

    num = min(len(ts), itter+1)
    return ts[:num], ys[:num],speed,np.arctan(adge)


def block(mass, S, startX, startY, speed, teta, ρ, σ, butcher):
    # Обработка начальных условий           # u0, γ0, y0, θ0, t0, v0
    x = startX
    y = startY
    adge = np.tan(teta)
    teta = np.rad2deg(teta)
    t = 0
    Xspeed = speed * np.cos(teta)

    h = 0.0002
    # h = 1
    bounds = 10000
    # Arrays
    ys = np.zeros(int(bounds / h))
    ys[0] = startY
    ts = np.arange(startX, bounds, h)
    ks = np.zeros((7,butcher.shape[1]))

    Ex = (mass * Xspeed ** 2) / 2
    ks[1, 0] = Ex

    # Система дифференциальных уравнений:
    # E_med_v/dx = π*R^2*w * (σ/2 + ρ*(v*R/L));
    # E_med_u/dx = π*R^2*w * (σ/2 + ρ*(u*R/L));
    # dv/dx = -sqrt(2*E_med_v/m);
    # du/dx = -sqrt(2*E_med_u/m);
    # dγ/dx = θ^2;
    # dy/dx = γ;
    # dt/dx = 1/u;

    itter = 0
    while speed >= 0:  # y
        # k
        for i in range(15):
            sp = speed + h*np.dot(ks[1,:i+1], butcher[i,:i+1])
            spx = Xspeed + h*np.dot(ks[2,:i+1], butcher[i,:i+1])
            E_med_v = S * (σ / 2 + ρ * sp) * 2000                       # E_med_v
            ks[0, i] = S * (σ/2 + ρ * spx) * 2000                       # E_med_u
            ks[1, i] = -np.sqrt(2*E_med_v/mass)                         # speed     # v
            ks[2, i] = -np.sqrt(2*ks[0,i]/mass)                         # dx_speed  # u
            ks[3, i] = teta ** 2                                        # adge      # γ
            ks[4, i] = (adge + h*np.dot(ks[3,:i+1], butcher[i,:i+1]))   # y         # y
            ks[5, i] = 1 / (spx)                                        # t         # t
        # Рассчёт x1, x2, x3, x4, x5:
        #                    len_
        # x(t+h) = x(t) + h * Σ a[len_][j] * k[j]
        #                    j=0
        #Ex-=h*np.dot(ks[1],butcher[-1])
        Xspeed += h*np.dot(ks[2],butcher[-1])   # u
        adge += h*np.dot(ks[3],butcher[-1])     # γ
        teta += h*np.dot(ks[3],butcher[-1])     # θ
        y += h*np.dot(ks[4],butcher[-1])        # y
        t += h*np.dot(ks[5],butcher[-1])        # t
        speed += h*np.dot(ks[1],butcher[-1])    # v

        itter += 1
        if y >= 0 and itter < int(bounds / h):
            ys[itter] = y
        else:
            break
    num = min(len(ts), itter)
    return ts[:num], ys[:num]


def dd(i, argX, argY):
    if i == argX:
        return argY
    else:
        return 0


def research():
# {
    # Дано:
    # Характеристики тела:
    mass = 0.0061           # mass – масса пули
    S = 6.6 * 10 ** (-5)    # S – площадь миделева сечения тела
    speedi = 420
    # Характеристики системы отсчёта:
    startX = 0
    startY = 1.6            # startY – начальная высота пули
    teta = 0                # teta – угол в градусах
    teta = np.deg2rad(teta)
    l = 20                  # l – горизонтальное расстояние до препятствия

    # Характеристики препятствия:
    ρ = 7850  # 666               # ρ – плотность препятствия
    σ = 450000000  # 66000000          # σ – предельное напряжение для среды


    # Parser
    F_on = True
    butcher = parse('butcher/DP8')
    print(butcher)

    # AIR
    first_stamp = time.time()
    a = air(mass, S, startX, startY, speedi, teta, l, butcher, F_on)
    # BLOCK
    aa = []
    if a[1][-1] > 0:
        aa = block(mass, S, l, a[1][-1], a[2], a[3], ρ, σ, butcher)
    end_stamp = time.time()

    print((end_stamp - first_stamp) * 1000)


    F_on = False
    # AIR
    first_stamp = time.time()
    b = air(mass, S, startX, startY, speedi, teta, l, butcher, F_on)
    end_stamp = time.time()
    print((end_stamp - first_stamp) * 1000)


    # PLOT
    ay_max = max(a[1])
    ax_max = a[0][a[1].argmax()]

    aay_max = max(aa[1])

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    p = patches.Rectangle(
        (l, 0), 2, max(aay_max,ay_max)+2,
        fill=True, clip_on=False,color="y",alpha=0.2
        )

    ax1.add_patch(p)

    cy = [dd(i, ax_max, ay_max) for i in a[0]]
    by_max = max(b[1])
    bx_max = b[0][b[1].argmax()]
    dy = [dd(i, bx_max, by_max) for i in b[0]]
    ax1.plot(a[0], cy, 'g')
    ax1.plot(aa[0], aa[1], 'r')
    ax1.plot(b[0], dy, 'y')
    ax1.plot(l, max(ay_max,by_max), 'b')

    ax1.plot(a[0], a[1], label='normal')
    ax1.plot(b[0], b[1], 'm', label='F(air)=0')
    ax1.legend()
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(a[0], a[1], color='tab:blue', label='air')
    ax.plot(aa[0], aa[1], color='tab:red', label='block')
    #ax.plot(b[0], b[1], color='tab:red', label='angle F(air)=0')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    #ax.set_ylim(startY - 1, startY + 1)
    plt.legend()
    plt.show()
# }


if __name__ == "__main__":
    # int main()
    # {
    g = 9.80665
    research()
    # return 0
    # }

def air_K(mass, S, startX, startY, speed, teta, l, butcher, F_para):
    Xspeed = speed * np.cos(teta)           # u0
    adge = np.tan(teta)
    y = startY                              # y0
    t = 0                                   # t0
    speed = speed                           # v0
    g = 9.80665

    h = 0.002
    # h = 1
    bounds = 100
    # Arrays
    ys = np.zeros(int(l / h))
    ys[0] = y
    ts = np.arange(0, l, h)
    ks = np.zeros((5,butcher.shape[1]))

    itter = 0
    while y > 0 and ts[itter] < l:          # (y > 0 && x < l)
        if ts[itter] + h > l:
            h = l-ts[itter]
        for i in range(15):
            #                    i-1
            # k[i] = f(x(t) + h * ОЈ a[i][j] * k[j])
            #                    j=0
            sp = speed + h * np.dot(ks[4, :i + 1], butcher[i, :i + 1])
            spx = Xspeed + h * np.dot(ks[0, :i + 1], butcher[i, :i + 1])
            ks[0, i] = (-Force(F_para,S, sp) / (mass * sp))                     # dx_speed  # u
            ks[1, i] = (-g / ((spx) ** 2))                                      # adge 
            ks[2, i] = (adge + h * np.dot(ks[1, :i + 1], butcher[i, :i + 1]))   # y         # y
            ks[3, i] = 1 / (spx)                                                # t         # t
            ks[4, i] = (ks[0, i]) * np.sqrt(1 + (ks[1, i]) ** 2)                # speed     # v

        Xspeed += h * np.dot(ks[0],butcher[-1]) # u
        adge += h * np.dot(ks[1],butcher[-1])  
        y += h * np.dot(ks[2],butcher[-1])      # y
        t += h * np.dot(ks[3],butcher[-1])      # t
        speed += h * np.dot(ks[4],butcher[-1])  # v

        itter += 1
        if y >= 0 and itter < int(l / h):
            ys[itter] = y
        else:
            break
    return np.arctan(adge),speed,y

def block_K(mass, S, startX, startY, speed, teta, ПЃ, Пѓ, butcher):
    x = startX
    y = startY
    adge = np.tan(teta)
    teta = np.rad2deg(teta)
    t = 0
    Xspeed = speed * np.cos(teta)

    h = 0.0002
    # h = 1
    bounds = 10000
    # Arrays
    ys = np.zeros(int(bounds / h))
    ys[0] = startY
    ts = np.arange(startX, bounds, h)
    ks = np.zeros((7,butcher.shape[1]))

    Ex = (mass * Xspeed ** 2) / 2
    ks[1, 0] = Ex

    itter = 0
    while speed >= 0:  # y
        # k
        for i in range(15):
            sp = speed + h*np.dot(ks[1,:i+1], butcher[i,:i+1])
            spx = Xspeed + h*np.dot(ks[2,:i+1], butcher[i,:i+1])
            E_med_v = S * (Пѓ / 2 + ПЃ * sp) * 2000                       # E_med_v
            ks[0, i] = S * (Пѓ/2 + ПЃ * spx) * 2000                       # E_med_u
            ks[1, i] = -np.sqrt(2*E_med_v/mass)                         # speed     # v
            ks[2, i] = -np.sqrt(2*ks[0,i]/mass)                         # dx_speed  # u
            ks[3, i] = teta ** 2                                        # adge      # Оі
            ks[4, i] = (adge + h*np.dot(ks[3,:i+1], butcher[i,:i+1]))   # y         # y
            ks[5, i] = 1 / (spx)                                        # t         # t

        Xspeed += h*np.dot(ks[2],butcher[-1])
        adge += h*np.dot(ks[3],butcher[-1]) 
        teta += h*np.dot(ks[3],butcher[-1]) 
        y += h*np.dot(ks[4],butcher[-1])
        t += h*np.dot(ks[5],butcher[-1])
        speed += h*np.dot(ks[1],butcher[-1])

        itter += 1
        if y >= 0 and itter < int(bounds / h):
            ys[itter] = y
        else:
            break
    num = min(len(ts), itter)
    return np.arctan(adge),speed,ts[num-1],y
