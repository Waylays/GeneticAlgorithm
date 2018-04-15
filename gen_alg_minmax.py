import random
import math
import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm


def obj_func(x):
    a = 20
    b = 0.2
    c = 2*np.pi
    d = 5.7
    f = 0.8
    n = 2
    return (1./f)*(-a*np.exp(-b*np.sqrt(np.mean(x*x)))-np.exp(np.mean(np.cos(c*x)))+a+np.exp(1)+d)

def nbits(a, b, dx):
    count = (b-a)/dx
    B = np.rint(np.log2(count))
    result = []
    result.append((b-a)/(2**B-1))
    result.append(int(B))
    return result

def gen_population(P, N, B):
    # P - l. osobnikow
    # N - l. zmiennych
    # B - l. bitów
    pop = np.random.randint(2,size=(P,N*B))
    return pop

def decodeIndividualVariable(x,B,a,b):
    return (a + ((b - a) * binaryToInt(x)) / (2 ** B - 1))

def decodeIndividual(osobnik, N, B, a, b):
    splitted = np.split(osobnik, N)
    decoded = []
    for x in splitted:
        decoded.append(decodeIndividualVariable(x,B,a,b))
    return np.array(decoded)

def decode_population(pop, N, B, a, b):
    out = []
    for i, osobnik in enumerate(pop):
        out.append(decodeIndividual(osobnik,N, B, a, b))

    return out

def evaluate_population(pop, func):
    evaluated_pop = []
    for i, arr in enumerate(pop):
        evaluated_pop.append(func(arr))
    return evaluated_pop

def get_best(pop, evaluated_pop):
    min = evaluated_pop[0]
    index = 0;
    for i, e in enumerate(evaluated_pop):
        if(e < min):
            min = e
            index = i
    return pop[index]

def get_best_1(evaluated_pop):
    min = evaluated_pop[0]
    index = 0;
    for i, e in enumerate(evaluated_pop):
        if(e < min):
            min = e
            index = i
    return evaluated_pop[index]

def sum(arr):
    sum = 0
    for a in arr:
        sum += a
    return sum

def binaryToInt(bits):
    out = 0
    for bit in bits:
        out = (out << 1)|bit
    return out

def cross(pop, pk):

    for i in range(0,len(pop)-1, 2):
        o = random.random()
        if(o < pk):
            temp1 = list(pop[i])
            temp2 = list(pop[i + 1])
            x = random.randint(1, len(pop[i]) - 1)
            res1 = list(temp1[:x]) + list(temp2[x:])
            res2 = list(temp2[:x]) + list(temp1[x:])
            pop[i] = np.array(res1)
            pop[i+1] = np.array(res2)
    return pop

def roulette(pop, pop_eval):
    v = 0
    ranges = []
    new_pop = []
    max = np.amax(pop_eval)

    for x in pop_eval:
        ranges.append((v, v + (max-x)))
        v += (max-x)
    for i in range(len(pop)):
        rn = np.random.random() * v
        for j in range(len(ranges)):
            if(rn >= ranges[j][0] and rn <= ranges[j][1]):
                new_pop.append(pop[j])
                break

    return new_pop

def mutate(pop, pm ):
    x = 0
    for i,row in enumerate(pop):
        for j,col in enumerate(row):
            if random.random() <= pm:
                x += 1
                pop[i][j] = not pop[i][j]

P = 150
N = 2
a = -1.5
b = 1.5
max_iter = 150

dx, B = nbits(a, b, 0.001)


pop = gen_population(P,N,B)

decoded_pop = decode_population(pop, N, B, a, b)
decoded_pop = np.array(decoded_pop)

evaluated_pop = evaluate_population(decoded_pop, obj_func)

best = get_best(pop, evaluated_pop)

vectorBest = np.zeros(max_iter,dtype=float)
vectorAvg = np.zeros(max_iter,dtype=float)

for x in range(max_iter):
    pop = roulette(pop,evaluated_pop)
    pop = cross(pop,0.7)
    mutate(pop,0.002)

    decoded_pop = decode_population(pop, N, B, a, b)
    decoded_pop = np.array(decoded_pop)
    evaluated_pop = evaluate_population(decoded_pop,obj_func)

    best_tmp = get_best(pop, evaluated_pop)
    if obj_func(decodeIndividual(best, N, B, a, b)) > obj_func(decodeIndividual(best_tmp, N, B, a, b)):
        best = best_tmp

    vectorBest[x] = obj_func(decodeIndividual(best,N,B,a,b))
    vectorAvg[x] = np.average(evaluated_pop)

    print("best:", best)
    print("best_decoded:", np.array(decodeIndividual(best,N,B,a,b)))
    print("min:", obj_func(decodeIndividual(best,N,B,a,b)))
    print("avg: ", vectorAvg[x])
    print("")


n = 20
sampled = np.linspace(a, b, n)
x, y = np.meshgrid(sampled, sampled)
z = np.zeros((len(sampled),len(sampled)))

for i in range(len(sampled)):
    for j in range(len(sampled)):
        z[i, j] = obj_func(np.array([x[i,j],y[i,j]]))


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='viridis', linewidth=0.4, antialiased=True)
cset = ax.contourf(x, y, z, zdir='z', offset= 7, cmap='viridis', alpha=0.5)
cset = ax.contourf(x, y, z, zdir='x', offset= 2, cmap='viridis', alpha=0.5)
cset = ax.contourf(x, y, z, zdir='y', offset= 2, cmap='viridis', alpha=0.5)

ax.view_init(30, 200)
plt.show()

plt.plot(vectorBest)
plt.xlabel('Iteracja')
plt.ylabel('Wartość funkcji dla najlepszego')
plt.title('Zmiana najlepszego osobnika')
plt.axis([0,max_iter,np.min(vectorBest)-0.5,np.max(vectorBest)+0.5])
plt.grid(True)
plt.show()


plt.plot(vectorAvg)
plt.xlabel('Iteracja')
plt.ylabel('Wartość średnia')
plt.title('Zmiana średniej')
plt.axis([0,max_iter,np.min(vectorAvg)-2,np.max(vectorAvg) + 2])
plt.grid(True)
plt.show()


obj_func(np.zeros(2))