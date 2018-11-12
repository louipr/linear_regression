import numpy as np
import matplotlib.pyplot as plt

def mse(m,b,xv,yv):
    N = len(xv)
    error = 0
    for i in range(N):
        error += (yv[i] - (m*xv[i] + b))**2
    error /= N
    return error 

def stepGradient(b_current,m_current,xv,yv,gamma):
    #gradient descent 
    b_gradient = 0
    m_gradient = 0
    N = len(xv)
    for i in range(N):
        m_gradient += xv[i]*(yv[i]-(m_current*xv[i] + b_current))
        b_gradient += (yv[i]-(m_current*xv[i] + b_current))
    m_gradient = m_gradient*(-2.0/N)
    b_gradient = b_gradient*(-2.0/N)

    m_new = m_current - gamma*m_gradient
    b_new = b_current - gamma*b_gradient
    #stdout = "dEdm=%f, dEdb=%f"%(m_gradient,b_gradient)
    #print(stdout)
    return [m_new,b_new,m_gradient,b_gradient]

def get_mse_weights(b_init,m_init,xv,yv,gamma,iterations):
    m = m_init
    b = b_init
    
    mse_vec = []
    m_gradient_vec = []
    b_gradient_vec = []
    for i in range(iterations):
        m,b,m_gradient,b_gradient = stepGradient(b,m,xv,yv,gamma)
        mse_vec.append(mse(m,b,xv,yv))
        m_gradient_vec.append(m_gradient)
        b_gradient_vec.append(b_gradient)

    return [m,b,mse_vec,m_gradient_vec,b_gradient_vec]    

def main():
    #Number of points 
    N = 100

    #(xi,yi) generate
    #Xv = np.random.randint(1,10,size=N)
    #Yv = np.random.randint(1,10,size=N)
    mu,sigma = 2,5
    Xv = np.random.normal(mu,sigma,size=N)
    Yv = np.random.normal(mu,sigma,size=N)
    
    for i in range(N): 
        stdout = "(X%d,Y%d)=(%d,%d)"%(i,i,Xv[i],Yv[i])
        print(stdout)

    mse_vec = []
    m,b,mse_vec,m_gradient_vec,b_gradient_vec = get_mse_weights(1,1,Xv,Yv,0.01,10000)

    x_min = np.min(Xv)
    x_max = np.max(Xv)
    xl = np.arange(x_min,x_max)
    yl = []
    for x in xl: 
        yl.append(x*m + b)

    #print last 
    stdout = "mse_final=%f, dEdm_final=%f, dEdb_final=%f"%(mse_vec[-1],m_gradient_vec[-1],b_gradient_vec[-1])
    print(stdout)
    
    #plot
    plt.figure(1)
    plt.plot(Xv,Yv,'ro')
    plt.plot(xl,yl)

    #figure 2
    plt.figure(2)
    plt.subplot(221)
    plt.plot(mse_vec)
    plt.xlabel('mse')
    plt.subplot(222)
    plt.plot(m_gradient_vec,color='blue')
    plt.xlabel('m_gradient')
    plt.subplot(224)
    plt.plot(b_gradient_vec,color='green')
    plt.xlabel('b_gradient')
    plt.show()




if __name__ == "__main__":
    main()
