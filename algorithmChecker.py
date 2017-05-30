from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np

#functions for project
def f1(x):
    """High Conditioned Elliptic Function"""
    sum = 0.0
    for i in range(1, len(x)+1):
        sum += (10**6)**((i-1)/(len(x)-1)) * x[i-1]**2
    return sum

def f2(x):
    """Bent cigar function"""
    sum = 0.0
    sum += x[0]**2
    for i in range(2, len(x)+1):
        sum += x[i-1]**2
    sum *= (10**6)
    return sum

def f3(x):
    """discus function"""
    sum = 0.0
    sum += (x[0]**2)*(10**6)
    for i in range(2, len(x)+1):
        sum += x[i-1]**2
    return sum

def f4(x):
	"""F8 Rosenbrock's saddle"""
	sum = 0.0
	for i in range(len(x)-1):
		sum += 100*((x[i]**2)-x[i+1])**2+\
		(1-x[i])**2
	return sum

def f5(x):
    """Ackley's Function"""
    sum1, sum2 = 0.0, 0.0
    for i in range(0, len(x)):
        sum1 += x[i]**2
    sum1 = sum1 / float(len(x))
    for i in range(0, len(x)):
        sum2 += np.cos(2*np.pi*x[i])
    sum2 = sum2 / float(len(x))

    # Calculate first exp
    exp1 = -20.0 * (np.e ** (-0.2 * sum1))
    exp2 = np.e ** sum2

    # Calculate final result
    result = exp1 - exp2 + 20 + np.e
    return result

def f6(x):
        sum1, sum2, sum3 = 0.0, 0.0, 0.0
        a = 0.5
        b = 3
        kmax = 20
        for i in range(len(x)):
            for k in range(0, kmax):
                sum2 += (a ** k) * np.cos(2 * np.pi * (b ** k) * (x[i] + 0.5))
                sum3 += (a ** k) * np.cos(2 * np.pi * (b ** k) * 0.5)
        sum1 += sum2 - (len(x) * sum3)
        return sum1

def f7(x):
    """Griewank's function"""
    sum = 0
    for i in x:
        sum += i * i
    product = 1
    for j in xrange(len(x)):
        product *= np.cos(x[j] / np.sqrt(j + 1))
    return 1 + sum / 4000 - product

def f8(x):
    """Rastrigin's Function"""
    sum = 0.0
    for i in range(0, len(x)):
        sum += (x[i]**2 - 10 * np.cos(2*np.pi*x[i]) + 10)
    return sum

def f9(x):
    """Katsuura Function"""
    product = 1
    for i in range(0, len(x)):
        sum = 0
        for j in range(1,33):
            term = np.power(2,j) * x[i]
            sum += np.abs(term - np.round(term))/(np.power(2,j))
        product *= np.power(1+((i+1)*sum),10.0/ np.power(len(x),1.2))
    return (10/len(x) * len(x) * product - (10/len(x) * len(x)))

#graphs for part 1
"""
#Function 1
X = np.linspace(-100, 100, 100)            # points from 0..10 in the x axis
Y = np.linspace(-100, 100, 100)            # points from 0..10 in the y axis
X, Y = np.meshgrid(X, Y)               # create meshgrid
Z = f1([X, Y])                         # Calculate Z

# Plot the 3D surface for first function from project
fig = plt.figure()
ax = fig.gca(projection='3d')         # set the 3d axes
ax.plot_surface(X, Y, Z,
                rstride=3,
                cstride=3,
                alpha=0.3,
                cmap='hot')
plt.show()

#Function 2
X = np.linspace(-100, 100, 100)            # points from 0..10 in the x axis
Y = np.linspace(-100, 100, 100)            # points from 0..10 in the y axis
X, Y = np.meshgrid(X, Y)               # create meshgrid
Z = f2([X, Y])                         # Calculate Z

# Plot the 3D surface for first function from project
fig = plt.figure()
ax = fig.gca(projection='3d')         # set the 3d axes
ax.plot_surface(X, Y, Z,
                rstride=3,
                cstride=3,
                alpha=0.3,
                cmap='hot')
plt.show()

#Function 3
X = np.linspace(-100, 100, 100)            # points from 0..10 in the x axis
Y = np.linspace(-100, 100, 100)            # points from 0..10 in the y axis
X, Y = np.meshgrid(X, Y)               # create meshgrid
Z = f3([X, Y])                         # Calculate Z

# Plot the 3D surface for first function from project
fig = plt.figure()
ax = fig.gca(projection='3d')         # set the 3d axes
ax.plot_surface(X, Y, Z,
                rstride=3,
                cstride=3,
                alpha=0.3,
                cmap='hot')
plt.show()

#Function 4
X = np.linspace(-100, 100, 100)            # points from 0..10 in the x axis
Y = np.linspace(-100, 100, 100)            # points from 0..10 in the y axis
X, Y = np.meshgrid(X, Y)               # create meshgrid
Z = f4([X, Y])                         # Calculate Z

# Plot the 3D surface for first function from project
fig = plt.figure()
ax = fig.gca(projection='3d')         # set the 3d axes
ax.plot_surface(X, Y, Z,
                rstride=3,
                cstride=3,
                alpha=0.3,
                cmap='hot')
plt.show()

#Function 5
X = np.linspace(-100, 100, 100)            # points from 0..10 in the x axis
Y = np.linspace(-100, 100, 100)            # points from 0..10 in the y axis
X, Y = np.meshgrid(X, Y)               # create meshgrid
Z = f5([X, Y])                         # Calculate Z

# Plot the 3D surface for first function from project
fig = plt.figure()
ax = fig.gca(projection='3d')         # set the 3d axes
ax.plot_surface(X, Y, Z,
                rstride=3,
                cstride=3,
                alpha=0.3,
                cmap='hot')
plt.show()

#Function 6
X = np.linspace(-100, 100, 100)            # points from 0..10 in the x axis
Y = np.linspace(-100, 100, 100)            # points from 0..10 in the y axis
X, Y = np.meshgrid(X, Y)               # create meshgrid
Z = f6([X, Y])                         # Calculate Z

# Plot the 3D surface for first function from project
fig = plt.figure()
ax = fig.gca(projection='3d')         # set the 3d axes
ax.plot_surface(X, Y, Z,
                rstride=3,
                cstride=3,
                alpha=0.3,
                cmap='hot')
plt.show()

#Function 7
X = np.linspace(-100, 100, 100)            # points from 0..10 in the x axis
Y = np.linspace(-100, 100, 100)            # points from 0..10 in the y axis
X, Y = np.meshgrid(X, Y)               # create meshgrid
Z = f7([X, Y])                         # Calculate Z

# Plot the 3D surface for first function from project
fig = plt.figure()
ax = fig.gca(projection='3d')         # set the 3d axes
ax.plot_surface(X, Y, Z,
                rstride=3,
                cstride=3,
                alpha=0.3,
                cmap='hot')
plt.show()

#Function 8
X = np.linspace(-100, 100, 100)            # points from 0..10 in the x axis
Y = np.linspace(-100, 100, 100)            # points from 0..10 in the y axis
X, Y = np.meshgrid(X, Y)               # create meshgrid
Z = f8([X, Y])                         # Calculate Z

# Plot the 3D surface for first function from project
fig = plt.figure()
ax = fig.gca(projection='3d')         # set the 3d axes
ax.plot_surface(X, Y, Z,
                rstride=3,
                cstride=3,
                alpha=0.3,
                cmap='hot')
plt.show()

#Function 9
X = np.linspace(-100, 100, 100)            # points from 0..10 in the x axis
Y = np.linspace(-100, 100, 100)            # points from 0..10 in the y axis
X, Y = np.meshgrid(X, Y)               # create meshgrid
Z = f9([X, Y])                         # Calculate Z

# Plot the 3D surface for first function from project
fig = plt.figure()
ax = fig.gca(projection='3d')         # set the 3d axes
ax.plot_surface(X, Y, Z,
                rstride=3,
                cstride=3,
                alpha=0.3,
                cmap='hot')
plt.show()"""