import pickle5 as pickle
import matplotlib.pyplot as plt
import numpy as np

def counts2power(count_freq,lamda):

    #lamda given in m
    #count_freq in Hz
    
    h = 6.626070040e-34 #Joules*sec
    c = 3e8 #m/s
    power = count_freq*h*c/(lamda)
    print('count frequency: ',count_freq,'[Hz]')
    print('power: ',power,'[Watts]')
    return power

def power2counts(power, lamda):

    #lamda given in m
    #power in watts
    h = 6.626070040e-34 #Joules*sec
    c = 3e8 #m/s
    count_freq = power*(lamda)/(h*c) 
    print('power: ',power,'[Watts]')
    print('count frequency: ',count_freq,'[Hz]')
    return count_freqß

def save_fig(fig,filename):

    pickle.dump(fig, open( filename + '.fig.pickle', 'wb'))
    f = open(filename+'.pyx',"w+")
    fig.clf()
    f.write("import pickle5 as pickle\n")
    f.write("import matplotlib.pyplot as plt\n")
    f.write("import addcopyfighandler as copyfig\n")
    file_string = "figx = pickle.load(open(r'"+filename+".fig.pickle','rb'))"
    f.write(file_string + '\n')
    #f.write("dummy = plt.figure()\n")
    #f.write("new_manager = dummy.canvas.manager\n")
    #f.write("new_manager.canvas.figure = figx\n")
    #f.write("figx.set_canvas(new_manager.canvas)\n")
    #fig.clf()
    f.write("copyfig.copyfig()\n")
    f.write("plt.show()\n")
    
def skew_gauss(x,a,b,c1,c2):

    c = np.linspace(c1,c2,len(x))
    y = a*np.exp(-(x-b)**2/(2*c**2))
    t = 1
    return y

def lorentz(x,a,b,c):
    lor = (a*b**2/(b**2+(x-c)**2))
    return lor

def gauss(x,a,b,c):
    gau = a*np.exp(-(x-c)**2/(2*b**2))
    return gau

def norma(a):
    a = np.asarray(a)
    return a/np.max(a)

