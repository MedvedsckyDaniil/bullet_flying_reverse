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
    # return c_x # (РµСЃР»Рё Р±РµР· РЅРѕСЂРјРёСЂРѕРІР°РЅРёСЏ)
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
        i = 0.47    # i вЂ“ РєРѕСЌС„С„РёС†РёРµРЅС‚ С„РѕСЂРјС‹
        # cx вЂ“ С„СѓРЅРєС†РёСЏ Р·Р°РєРѕРЅР° СЃРѕРїСЂРѕС‚РёРІР»РµРЅРёСЏ РІРѕР·РґСѓС…Р°;
        a = 335     # a вЂ“ СЃРєРѕСЂРѕСЃС‚СЊ Р·РІСѓРєР° (РЅ.Сѓ. 335 Рј/СЃРµРє);
        p = 1.2754  # ПЃ вЂ“ РїР»РѕС‚РЅРѕСЃС‚СЊ Р°С‚РјРѕСЃС„РµСЂС‹ (РЅ.Сѓ. 1,2754 РєРі/РјВі).
        return i * S * p * speed ** 2 * cx(speed / a) / 2
    else:
        return 0



def air(mass, S, startX, startY, speed, teta, l, butcher, F_para):
    # РћР±СЂР°Р±РѕС‚РєР° РЅР°С‡Р°Р»СЊРЅС‹С… СѓСЃР»РѕРІРёР№           # u0, Оі0, y0, t0, v0
    Xspeed = speed * np.cos(teta)           # u0
    adge = np.tan(teta)                     # Оі0
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

    # РЎРёСЃС‚РµРјР° РґРёС„С„РµСЂРµРЅС†РёР°Р»СЊРЅС‹С… СѓСЂР°РІРЅРµРЅРёР№:
    # du/dx = -F(v)/(m*v);
    # dОі/dx = -g/(u^2);
    # dy/dx = Оі;
    # dt/dx = 1/u;
    # v = u * sqrt(1+Оі^2)

    itter = 0
    while y > 0 and ts[itter] < l:          # (y > 0 && x < l)
        if ts[itter] + h > l:
            h = l-ts[itter]
        # Р Р°СЃСЃС‡С‘С‚ k0, k1, k2, k3, k4:
        for i in range(15):
            #                    i-1
            # k[i] = f(x(t) + h * ОЈ a[i][j] * k[j])
            #                    j=0
            sp = speed + h * np.dot(ks[4, :i + 1], butcher[i, :i + 1])
            spx = Xspeed + h * np.dot(ks[0, :i + 1], butcher[i, :i + 1])
            ks[0, i] = (-Force(F_para,S, sp) / (mass * sp))                     # dx_speed  # u
            ks[1, i] = (-g / ((spx) ** 2))                                      # adge      # Оі
            ks[2, i] = (adge + h * np.dot(ks[1, :i + 1], butcher[i, :i + 1]))   # y         # y
            ks[3, i] = 1 / (spx)                                                # t         # t
            ks[4, i] = (ks[0, i]) * np.sqrt(1 + (ks[1, i]) ** 2)                # speed     # v
        # Р Р°СЃСЃС‡С‘С‚ x0, x1, x2, x3, x4:
        #                    len_
        # x(t+h) = x(t) + h * ОЈ a[len_][j] * k[j]
        #                    j=0
        Xspeed += h * np.dot(ks[0],butcher[-1]) # u
        adge += h * np.dot(ks[1],butcher[-1])   # Оі
        y += h * np.dot(ks[2],butcher[-1])      # y
        t += h * np.dot(ks[3],butcher[-1])      # t
        speed += h * np.dot(ks[4],butcher[-1])  # v

        itter += 1
        if y >= 0 and itter < int(bounds / h):
            ys[itter] = y
        else:
            break

    num = min(len(ts), itter+1)
    print(f"air: adge= {np.rad2deg(np.arctan(adge))} (degress), y= {y}, x= {ts[num-1]}, speed= {speed}")
    return ts[:num], ys[:num],speed,np.arctan(adge)


def block(mass, S, startX, startY, speed, teta, ПЃ, Пѓ, butcher):
    # РћР±СЂР°Р±РѕС‚РєР° РЅР°С‡Р°Р»СЊРЅС‹С… СѓСЃР»РѕРІРёР№           # u0, Оі0, y0, Оё0, t0, v0
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

    # РЎРёСЃС‚РµРјР° РґРёС„С„РµСЂРµРЅС†РёР°Р»СЊРЅС‹С… СѓСЂР°РІРЅРµРЅРёР№:
    # E_med_u/dx = ПЂ*R^2*w * (Пѓ/2 + ПЃ*(u*R/L))
    # E_med_v/dx = ПЂ*R^2*w * (Пѓ/2 + ПЃ*(v*R/L))
    # dv/dx = -sqrt(2*E_med_v/m);
    # du/dx = -sqrt(2*E_med_u/m);
    # dОі/dx = Оё^2;
    # dy/dx = Оі;
    # dt/dx = 1/u;

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
        # Р Р°СЃСЃС‡С‘С‚ x1, x2, x3, x4, x5:
        #                    len_
        # x(t+h) = x(t) + h * ОЈ a[len_][j] * k[j]
        #                    j=0
        #Ex-=h*np.dot(ks[1],butcher[-1])
        Xspeed += h*np.dot(ks[2],butcher[-1])
        adge += h*np.dot(ks[3],butcher[-1]) # Оі
        teta += h*np.dot(ks[3],butcher[-1]) # Оё
        y += h*np.dot(ks[4],butcher[-1])
        t += h*np.dot(ks[5],butcher[-1])
        speed += h*np.dot(ks[1],butcher[-1])

        itter += 1
        if y >= 0 and itter < int(bounds / h):
            ys[itter] = y
        else:
            break
    num = min(len(ts), itter)
    print(f"block: adge= {np.rad2deg(np.arctan(adge))} (degress), y= {y}, x= {ts[num-1]}, speed= 0")
    return ts[:num], ys[:num]


def dd(i, argX, argY):
    if i == argX:
        return argY
    else:
        return 0


def research():
# {
    # Р”Р°РЅРѕ:
    # РҐР°СЂР°РєС‚РµСЂРёСЃС‚РёРєРё С‚РµР»Р°:
    mass = 0.0061           # mass вЂ“ РјР°СЃСЃР° РїСѓР»Рё
    S = 6.6 * 10 ** (-5)    # S вЂ“ РїР»РѕС‰Р°РґСЊ РјРёРґРµР»РµРІР° СЃРµС‡РµРЅРёСЏ С‚РµР»Р°
    speedi = 920#920#460#315
    # РҐР°СЂР°РєС‚РµСЂРёСЃС‚РёРєРё СЃРёСЃС‚РµРјС‹ РѕС‚СЃС‡С‘С‚Р°:
    startX = 0
    startY = 1.6            # startY вЂ“ РЅР°С‡Р°Р»СЊРЅР°СЏ РІС‹СЃРѕС‚Р° РїСѓР»Рё
    teta = 0                # teta вЂ“ СѓРіРѕР» РІ РіСЂР°РґСѓСЃР°С…
    teta = np.deg2rad(teta)
    l = 20                  # l вЂ“ РіРѕСЂРёР·РѕРЅС‚Р°Р»СЊРЅРѕРµ СЂР°СЃСЃС‚РѕСЏРЅРёРµ РґРѕ РїСЂРµРїСЏС‚СЃС‚РІРёСЏ

    # РҐР°СЂР°РєС‚РµСЂРёСЃС‚РёРєРё РїСЂРµРїСЏС‚СЃС‚РІРёСЏ:
    ПЃ = 7850  # 666               # ПЃ вЂ“ РїР»РѕС‚РЅРѕСЃС‚СЊ РїСЂРµРїСЏС‚СЃС‚РІРёСЏ
    Пѓ = 450000000  # 66000000          # Пѓ вЂ“ РїСЂРµРґРµР»СЊРЅРѕРµ РЅР°РїСЂСЏР¶РµРЅРёРµ РґР»СЏ СЃСЂРµРґС‹
    """
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
9 1.4041214778054267"""
    speedi=1064.0987352652594
    teta=-0.03753359288801369
    """
0 1004.4862050377168 0.021727307567034507 
1 999.5711975822754 0.0247675221973402 
2 1052.6275251198442 -0.014670451453917474 
3 998.3161724629178 0.02782060723250134 
4 1042.2010485048258 -0.008775842396325527 
5 1017.5985336155214 0.005908425622385527 
6 1011.0193420882584 0.012230229452026653 
7 1051.7725442634744 -0.014810791102266095 
8 995.1178820516174 0.031176750899602222 
9 1031.2161929278243 -0.0038241472956954885 
0 2.0322513904796877
1 2.0930761187522897
2 1.3041992599901888
3 2.154185901469636
4 1.422114016891909
5 1.7157907124436749
6 1.8422407405981738
7 1.301385824108574
8 2.221363317364925
9 1.521138184833243
    """
    # Parser
    F_on = True
    butcher = parse('DP8')
    #print(butcher)
    print("air on")

    # AIR
    first_stamp = time.time()
    a = air(mass, S, startX, startY, speedi, teta, l, butcher, F_on)
    # BLOCK
    aa = []
    if a[1][-1] > 0:
        aa = block(mass, S, l, a[1][-1], a[2], a[3], ПЃ, Пѓ, butcher)
    end_stamp = time.time()

    print((end_stamp - first_stamp) * 1000)


    F_on = False
    print("air off")
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
    # РћР±СЂР°Р±РѕС‚РєР° РЅР°С‡Р°Р»СЊРЅС‹С… СѓСЃР»РѕРІРёР№           # u0, Оі0, y0, t0, v0
    Xspeed = speed * np.cos(teta)           # u0
    adge = np.tan(teta)                     # Оі0
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

    # РЎРёСЃС‚РµРјР° РґРёС„С„РµСЂРµРЅС†РёР°Р»СЊРЅС‹С… СѓСЂР°РІРЅРµРЅРёР№:
    # du/dx = -F(v)/(m*v);
    # dОі/dx = -g/(u^2);
    # dy/dx = Оі;
    # dt/dx = 1/u;
    # v = u * sqrt(1+Оі^2)

    itter = 0
    while y > 0 and ts[itter] < l:          # (y > 0 && x < l)
        if ts[itter] + h > l:
            h = l-ts[itter]
        # Р Р°СЃСЃС‡С‘С‚ k0, k1, k2, k3, k4:
        for i in range(15):
            #                    i-1
            # k[i] = f(x(t) + h * ОЈ a[i][j] * k[j])
            #                    j=0
            sp = speed + h * np.dot(ks[4, :i + 1], butcher[i, :i + 1])
            spx = Xspeed + h * np.dot(ks[0, :i + 1], butcher[i, :i + 1])
            ks[0, i] = (-Force(F_para,S, sp) / (mass * sp))                     # dx_speed  # u
            ks[1, i] = (-g / ((spx) ** 2))                                      # adge      # Оі
            ks[2, i] = (adge + h * np.dot(ks[1, :i + 1], butcher[i, :i + 1]))   # y         # y
            ks[3, i] = 1 / (spx)                                                # t         # t
            ks[4, i] = (ks[0, i]) * np.sqrt(1 + (ks[1, i]) ** 2)                # speed     # v
        # Р Р°СЃСЃС‡С‘С‚ x0, x1, x2, x3, x4:
        #                    len_
        # x(t+h) = x(t) + h * ОЈ a[len_][j] * k[j]
        #                    j=0
        Xspeed += h * np.dot(ks[0],butcher[-1]) # u
        adge += h * np.dot(ks[1],butcher[-1])   # Оі
        y += h * np.dot(ks[2],butcher[-1])      # y
        t += h * np.dot(ks[3],butcher[-1])      # t
        speed += h * np.dot(ks[4],butcher[-1])  # v

        itter += 1
        if y >= 0 and itter < int(l / h):
            ys[itter] = y
        else:
            break

    #num = min(len(ts), itter+1)
    #print(f"air: adge= {np.arctan(adge)} (degress), y= {y}, x= {ts[-1]}, speed= {speed}")
    #plt.plot(ts,ys)
    #plt.show()
    return np.arctan(adge),speed,y

def block_K(mass, S, startX, startY, speed, teta, ПЃ, Пѓ, butcher):
    # РћР±СЂР°Р±РѕС‚РєР° РЅР°С‡Р°Р»СЊРЅС‹С… СѓСЃР»РѕРІРёР№           # u0, Оі0, y0, Оё0, t0, v0
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
        adge += h*np.dot(ks[3],butcher[-1]) # Оі
        teta += h*np.dot(ks[3],butcher[-1]) # Оё
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
