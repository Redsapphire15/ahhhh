import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import math

import numpy as np

def qr_decomposition(A):
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))

    for j in range(n):
        v = A[:, j]
        for i in range(j):
            R[i, j] = np.dot(Q[:, i], A[:, j]) / np.linalg.norm(Q[:, i])**2
            v = v - R[i, j] * Q[:, i]

        R[j, j] = np.linalg.norm(v)
        Q[:, j] = v / R[j, j]

    return Q, R

def back_substitution(U, b):
    n = len(b)
    x = np.zeros(n)

    for i in range(n - 1, -1, -1):
        x[i] = b[i] / U[i, i]
        for j in range(i - 1, -1, -1):
            b[j] -= U[j, i] * x[i]

    return x

dataset = np.loadtxt('data1.txt')

for i in range(2,5,1):
    print("Case:")
    print(i)
    a = np.zeros([200,i])
    b = np.zeros([200,1])
    pt = np.zeros([200,1])

    if i == 2:        
        a[:,0] = dataset[:,0]
        a[:,1] = 1
    if i == 3:
        for l in range(0,200,1):
            a[l][0] = math.pow(dataset[l,0],2)
            a[l][1] = l+1
        a[:,2] = 1
    if i == 4:
        for l in range(0,200,1):
            a[l][0] = math.pow(dataset[l,0],3)
            a[l][1] = math.pow(l+1,2)
            a[l][2] = l+1
        a[:,3] = 1

    b[:,0] = dataset[:,1]

    for j in range(1,201,1):
        pt[j-1][0] = j

    Q, R = qr_decomposition(a)
    Q_sp, R_sp = sp.linalg.qr(a, mode = 'economic')
    U_svd, S_svd, V_svd = np.linalg.svd(a, full_matrices=False)
    print(V_svd)

    #print("Matrix A:")
    #print(a)
    #print("\nMatrix Q:")
    #print(Q)
    #print("\nMatrix R:")
    #print(R)
    #print(R_sp)
    #print("\nQ * R:")
    #print(np.dot(Q_sp, R_sp))

    Q_b = np.transpose(Q) @ b


    X = back_substitution(R,Q_b)

    for h in range(i):
        plt.plot(pt, Q[:,h], label = f"$Q_{h+1}$")
        plt.plot(pt,Q_sp[:,h], linestyle='--', label = f"$Q-SP_{h+1}$")
    plt.legend()
    plt.title("Columns of the Q matrix")
    plt.show()
    
    y_plt = np.zeros((200,1))
    X_rev = np.zeros((i,1))
    for h in range(i):
        for p in range(i-1,-1,-1):
            if h+p==i-1:
                X_rev[h][0] = X[p]
    
    for h in range(200):
        for w in range(i):
            y_plt[h][0] += X_rev[w][0]*(pt[h][0]**w)
    
    d1 = dataset[:,[0]]
    d2 = dataset[:,[1]]
    plt.plot(dataset[:,0],dataset[:,1])
    plt.plot(pt,y_plt, linestyle='--')
    plt.title("Regression fit")
    plt.show()
    for h in range(i):
        plt.plot(pt, U_svd[:,h], label = f"$U_{h}$")
    plt.title("U_SVD graph")
    plt.legend()
    plt.show()
    for h in range(len(V_svd)):
        plt.plot(pt[:len(V_svd),0], V_svd[:,h], linestyle = '--', label = f"$V_{h}$")
    plt.title("V_svd graph")
    plt.legend()
    plt.show()

    '''
   
    X = np.zeros([i,1])
    if i==2:
        X[1][0] = Q_b[1][0]/R_sp[1][1]
        X[0][0] = (Q_b[0][0] -(R_sp[0][1]*X[1][0]))/R_sp[0][0]
    if i==3:
        X[2][0] = Q_b[2][0]/R_sp[2][2]
        X[1][0] = (Q_b[1][0] - (R_sp[1][2] * X[2][0]))/R_sp[1][1]
        X[0][0] = (Q_b[0][0] - R_sp[0][1]*X[1][0] - R_sp[0][2]*X[2][0])/R_sp[0][0]
    if i==4:
        X[3][0] = Q_b[3][0]/R_sp[3][3]
        X[2][0] = (Q_b[2][0] - R_sp[2][3]*X[3][0])/R_sp[2][2]
        X[1][0] = (Q_b[1][0] - R_sp[1][3]*X[3][0] - R_sp[1][2]*X[2][0])/R_sp[1][1]
        X[0][0] = (Q_b[0][0] - R_sp[0][3]*X[3][0] - R_sp[0][2]*X[2][0] - R_sp[0][1]*X[1][0])/R_sp[0][0]
    print(X) 
'''