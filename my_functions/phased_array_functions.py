
import numpy as np
import random
import matplotlib.pyplot as plt 
import scipy
from scipy import misc
from scipy.ndimage import rotate
from matplotlib import pyplot as mp
from matplotlib.patches import Circle
from scipy import signal
from mayavi import mlab
from scipy import interpolate
import numpy as np
import matplotlib.pyplot as plt  
import cmath
from matplotlib.patches import Circle


def read_bauschi_kspace(filename):
    f = open(filename)

    x = []
    y = []
    a = []

    for line in f:
        x.append(np.float(line.split(',')[0]))
        y.append(np.float(line.split(',')[1]))
        a.append(np.float(line.split(',')[2]))

    a = np.asarray(a)
    a = a.reshape(512,512)

    return a

def make_cell_rod( n1, n2, d1, d2, w, l, mode, sharpness ) :
    cell = np.zeros( ( n1, n2 ) )*1j
    start1 = (d1-w)/2*n1/d1
    stop1 = (d1+w)/2*n1/d1
    start2 =  (d2-l)/2*n2/d2
    stop2 =  (d2+l)/2*n2/d2
    middle2 = n2/2
    middle1 = n1/2
    fourth =  (stop2-start2)/4+start2

    
    if mode == 0:
        for i in range(n1-1):
            for j in range(n2-1):
                if i > start2 and i <= stop2 and j > start1 and j <= stop1 :
                        cell[i,j] = 1
         
    if mode == 1:
        for i in range(n1-1):
            for j in range(n2-1):
                if i > start2 and i < stop2 and j > start1 and j < stop1 :
                    if i > middle1:
                        cell[i,j] = (i-middle1)/(stop2-middle1)*1j
                    if i <= middle1:
                        cell[i,j] = (i-middle1)/(stop2-middle1)*1j

    if mode == 2:
        for i in range(n1-1):
            for j in range(n2-1):
                if i > start2 and i < stop2 and j > start1 and j < stop1 :
                    if i < middle1:
                        cell[i,j] = (i-fourth)/(start2-fourth)
                    if i >= middle1:
                        cell[i,j] = -(i-fourth-(middle1-start2))/(start2-fourth)
        
    if mode == 3:
        for i in range(n1-1):
            for j in range(n2-1):
                if i > start2 and i < stop2 and j > start1 and j < stop1:
                    if i > middle1:
                        cell[i,j] = (j-middle2)*(i-middle1)/(stop2-middle2)*1j
                    if i <= middle1:
                        cell[i,j] = (i-middle1)/(stop2-middle2)*1j*(j-middle2)
        
    if mode == 4:
        for i in range(n1-1):
            for j in range(n2-1):
                if i > start2 and i < stop2 and j > start1 and j < stop1:
                    if i > middle1:
                        cell[i,j] = np.abs((j-middle2))*(i-middle1)/(stop2-middle2)*1j
                    if i <= middle1:
                        cell[i,j] = (i-middle1)/(stop2-middle2)*1j*np.abs((j-middle2))
        
    if mode == 5:
        for i in range(n1-1):
            for j in range(n2-1):
                if i > start2 and i < stop2 and j > start1 and j < stop1:
                    if i > middle1:
                        cell[i,j] = (i-middle1)/(stop2-middle1)*1j
                    if i <= middle1:
                        cell[i,j] = -(i-middle1)/(stop2-middle1)*1j
        
    if mode == 6:
        for i in range(n1-1):
            for j in range(n2-1):
                
                if i > start2 and i < stop2 and j > start1 and j < stop1:
                    
                    if j > middle2:
                        cell[i,j] = (j-middle2)/(stop1-middle2)*1j
                        
                    if j <= middle2:
                        cell[i,j] = (j-middle2)/(stop1-middle2)*1j
        
        cell1 = cell
        cell = np.zeros( ( n1, n2 ) )*1j
        for i in range(n1-1):
            for j in range(n2-1):
                if i > start2 and i < stop2 and j > start1 and j < stop1 :
                    if i > middle1:
                        cell[i,j] = (i-middle1)/(stop2-middle1)*1j
                    if i <= middle1:
                        cell[i,j] = (i-middle1)/(stop2-middle1)*1j
        cell2 = np.abs(cell)

        cell = cell1*cell2

    print('* cell ready')
    print('')
    #if mode == 1 : cell = np.abs(cell)+(1j*0)
    cell = cell**sharpness
    cell = cell/(np.max(np.max(cell)))
    #cell = -cell + 1 
    return cell

def make_array(N1, N2, cell):
  
    array_row = cell 
    for i in range(N1-1):
        array_row = np.concatenate((array_row,cell))

    array = array_row
    for j in range(N2-1):
        array = np.concatenate((array,array_row),axis=1)
    
    print('* array ready')
    print('')
    return array

def apply_noise_per_cell(array, N, std_cell): # multiplicative noise on every cell
    if std_cell == 0:
        return array
    n = np.int( np.shape(array)[0]/N )
    mean = 1
    for i in range(N):
        for j in range(N):
            cell = array[i*n:(i+1)*n, j*n:(j+1)*n ]
            noise = np.random.normal(mean,std_cell,1)
            if noise < 0 : noise = -noise # making sure that multiplicative noise doesn't take negative values
            cell = cell*noise
            array[i*n:(i+1)*n, j*n:(j+1)*n ] = cell
    print('* noise per cell ready')
    print('')
    return array

def apply_noise_per_halfcell(array, N, std_cell): # multiplicative noise on every cell
    if std_cell == 0:
        return array
    n = np.int( np.shape(array)[0]/N )
    mean = 1
    for i in range(N):
        for j in range(N):
            # first halfcell
            cell1 = array[i*n:np.int((i+1/2)*n), j*n:np.int((j+1)*n) ]
            noise = np.random.normal(mean,std_cell,1)
            if noise < 0 : noise = -noise # making sure that multiplicative noise doesn't take negative values
            cell1 = cell1*noise
            array[i*n:np.int((i+1/2)*n), j*n:np.int((j+1)*n) ] = cell1
            # second halfcell
            cell2 = array[np.int((i+1/2)*n):np.int((i+1)*n), j*n:np.int((j+1)*n) ]
            noise = np.random.normal(mean,std_cell,1)
            if noise < 0 : noise = -noise # making sure that multiplicative noise doesn't take negative values
            cell2 = cell2*noise
            array[np.int((i+1/2)*n):np.int((i+1)*n), j*n:np.int((j+1)*n) ] = cell2

    print('* noise per halfcell ready')
    print('')
    return array

def apply_noise_per_point(array,std_point): # additive noise on every point 

    if std_point == 0:
        return array

    noise = np.random.normal(0, std_point, np.shape(array))
    array = array + noise
    print('* noise per point ready')
    print('')
    return array

def apply_linear_phase_per_cell(array, dx, N, lamda, angle_x, angle_y, consider_polarization, n_substrate):
    
    if angle_x == 0 and angle_y == 0:
        return array
    
    n = np.int( np.shape(array)[0]/N )
    d1 = dx*n
    k = 2*np.pi/lamda*n_substrate
    ax = -angle_y*np.pi/180 # i invert them to match the axis of basuchi
    ay = angle_x*np.pi/180

    for i in range(N):
        for j in range(N):

            cell = array[i*n:(i+1)*n, j*n:(j+1)*n]
            projection_factor = 1
            if consider_polarization : projection_factor = np.cos(ax)**2
            phasex = np.exp(1j*k*i*d1*np.sin(ax))
            cell = cell*phasex*projection_factor
            phasey = np.exp(1j*k*j*d1*np.sin(ay))
            cell = cell*phasey
            array[i*n:(i+1)*n, j*n:(j+1)*n] = cell
    print('* linear phase per cell ready')
    print('')
    return array
    
def apply_linear_phase_per_point(array, dx, N, lamda, angle_x, angle_y,n_substrate):

    if angle_x == 0 and angle_y == 0:
        return array
    
    k = k = 2*np.pi/lamda*n_substrate
    n = np.int( np.shape(array)[0]/N )
    
    d1 = dx*n
    ax = -angle_y*np.pi/180 # i invert them to match the axis of basuchi
    ay = angle_x*np.pi/180
    length = np.shape(array)[0]
    for i in range(length):
        for j in range(length):
            
            phasex = np.exp(1j*k*((2*i+1)*dx/2*np.sin(ax)))
            array[i,j] =array[i,j]*phasex
            phasey = np.exp(1j*k*((2*j+1)*dx/2*np.sin(ay)))
            array[i,j] = array[i,j]*phasey
    print('* linear phase per point ready')
    print('')
    return array

def apply_gaussian_mask(a, ga_factor,N,stepped):
    if ga_factor == 0 : return a
    
    n1 = np.int(np.size(a,0))
    n2 = np.int(np.size(a,1))
    a_gaussed = np.zeros((n1,n2)) + 0*1j
    if stepped :
        n1 = np.int(np.size(a,0)/N)
        n2 = np.int(np.size(a,1)/N)
        ga_factor = ga_factor/(n1*N*3)

    if not stepped : 
        
        n1 = np.int(np.size(a,0))
        n2 = np.int(np.size(a,1))
        N = n1
        
    for i in range(N):
        for j in range(N):
            if stepped : 
                a_gaussed[i*n1:(i+1)*n1,j*n1:(j+1)*n1] = a[i*n1:(i+1)*n1,j*n1:(j+1)*n1]*((np.exp(-((i-N/2+0.5)**2 + (j-N/2+0.5)**2)/ga_factor)))#*np.exp( (-((i-N/2+0.5)**2  + (j-N/2+0.5)**2))/ga_factor) 
                #print(np.exp( (-(i-N/2+0.5)**2  - (j-N/2+0.5)**2)/ga_factor))
            if not stepped : a_gaussed[i][j] = ((np.exp(-((i-n1/2)**2 + (j-n2/2)**2)/ga_factor)))*a[i][j]
            
            
    print('* gaussian mask ready')
    print('')
    return np.asarray(a_gaussed)

def plot_cell(cell,dx,filepath): 
    n1 = cell.shape[0]
    d1 = cell.shape[0]*dx
    cell[0][0] = 0
    plt.imshow(norma(np.abs(cell)),extent=[0,d1*1e9,0,d1*1e9])
    plt.arrow(50,50,50,0,color='white',head_width=10, head_length=10)
    plt.arrow(50,50,0,50,color='white',head_width=10, head_length=10)
    plt.text(50,120,'$y$',color='white')
    plt.text(120,50,'$x$',color='white')
    plt.tight_layout()
    #plt.xlabel('kx/k0')
    #plt.ylabel('ky/k0')
    plt.axis('off')
    #plt.title('|E|^2')
    plt.colorbar(ticks=[0, 1])
    plt.xlabel('x [nm]')
    plt.ylabel('y [nm]')
    #plt.title('|E|')
    #plt.colorbar()
    plt.savefig(filepath + 'cell_abs_2D.pdf')
    plt.savefig(filepath+ 'cell_abs_2D.png')
    plt.clf()

    plt.imshow(np.angle(cell),extent=[0,d1*1e9,0,d1*1e9])
    plt.xlabel('x [nm]')
    plt.ylabel('y [nm]')
    plt.title('phase{E}')
    plt.colorbar()
    plt.savefig(filepath + 'cell_phase_2D.pdf')
    plt.savefig(filepath + 'cell_phase_2D.png')
    plt.clf()

    plt.imshow(np.real(cell),extent=[0,d1*1e9,0,d1*1e9])
    plt.xlabel('x [nm]')
    plt.ylabel('y [nm]')
    plt.title('Re{E}')
    plt.colorbar()
    plt.savefig(filepath + 'cell_real_2D.pdf')
    plt.savefig(filepath + 'cell_real_2D.png')
    plt.clf()

    plt.imshow(np.imag(cell),extent=[0,d1*1e9,0,d1*1e9])
    plt.xlabel('x [nm]')
    plt.ylabel('y [nm]')
    plt.title('Im{E}')
    plt.colorbar()
    plt.savefig(filepath + 'cell_imag_2D.pdf')
    plt.savefig(filepath + 'cell_imag_2D.png')
    plt.clf()

    plt.plot(np.linspace(0,d1*1e9,n1),np.abs(cell[:,np.int(n1/2)]))
    plt.xlabel('y [nm]')
    plt.ylabel('|E|')
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.savefig(filepath + 'cell_abs_1D.pdf')
    plt.savefig(filepath + 'cell_abs_1D.png')
    plt.clf()

    plt.plot(np.linspace(0,d1*1e9,n1),np.angle(cell[:,np.int(n1/2)]))
    plt.xlabel('y [nm]')
    plt.ylabel('phase{E}')
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.savefig(filepath + 'cell_phase_1D.pdf')
    plt.savefig(filepath + 'cell_phase_1D.png')
    plt.clf()

    plt.plot(np.linspace(0,d1*1e9,n1),np.real(cell[:,np.int(n1/2)]))
    plt.xlabel('y [nm]')
    plt.ylabel('Re{E}')
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.savefig(filepath + 'cell_real_1D.pdf')
    plt.savefig(filepath + 'cell_real_1D.png')
    plt.clf()

    plt.plot(np.linspace(0,d1*1e9,n1),np.imag(cell[:,np.int(n1/2)]))
    plt.xlabel('y [nm]')
    plt.ylabel('Im{E}')
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.savefig(filepath + 'cell_imag_1D.pdf')
    plt.savefig(filepath + 'cell_imag_1D.png')
    plt.clf()

def plot_array(array,dx,N,filepath):
    n1 = np.int(array.shape[0]/N)
    d1 = n1*dx  
    plt.clf()
    plt.imshow(np.abs(array),extent=[0,d1*1e9*N,0,d1*1e9*N])
    #plt.arrow(1000,1000,2000,0,color='white',head_width=10*20, head_length=10*20)
    #plt.arrow(1000,1000,0,2000,color='white',head_width=10*20, head_length=10*20)
    #plt.text(1000,3500,'$y$',color='white')
    #plt.text(3500,1000,'$x$',color='white')
    plt.tight_layout()
    #plt.xlabel('kx/k0')
    #plt.ylabel('ky/k0')
    plt.axis('off')
    #plt.title('|E|^2')
    #plt.colorbar(ticks=[0, 1])
    plt.xlabel('x [nm]')
    plt.ylabel('y [nm]')
    #plt.title('|E|')
    
    plt.savefig(filepath + 'array_abs_2D.pdf',dpi = 1200)
    plt.savefig(filepath + 'array_abs_2D.png')
    plt.clf()

    plt.imshow(np.angle(array),extent=[0,d1*1e9*N,0,d1*1e9*N])
    plt.xlabel('x [nm]')
    plt.ylabel('y [nm]')
    plt.title('phase{E}')
    plt.colorbar()
    plt.savefig(filepath + 'array_phase_2D.pdf')
    plt.savefig(filepath + 'array_phase_2D.png')
    plt.clf()

    plt.imshow(np.real(array),extent=[0,d1*1e9*N,0,d1*1e9*N])
    plt.xlabel('x [nm]')
    plt.ylabel('y [nm]')
    plt.title('Re{E}')
    plt.colorbar()
    plt.savefig(filepath + 'array_real_2D.pdf')
    plt.savefig(filepath + 'array_real_2D.png')
    plt.clf()

    plt.imshow(np.imag(array),extent=[0,d1*1e9*N,0,d1*1e9*N])
    plt.xlabel('x [nm]')
    plt.ylabel('y [nm]')
    plt.title('Im{E}')
    plt.colorbar()
    plt.savefig(filepath + 'array_imag_2D.pdf')
    plt.savefig(filepath + 'array_imag_2D.png')
    plt.clf()

    plt.plot(np.linspace(0,d1*1e9*N,n1*N),np.abs(array[:,np.int(n1*(N-1)/2)]))
    plt.xlabel('y [nm]')
    plt.ylabel('|E|')
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.savefig(filepath + 'array_abs_1D.pdf')
    plt.savefig(filepath + 'array_abs_1D.png')
    plt.clf()

    plt.plot(np.linspace(0,d1*1e9*N,n1*N),np.angle(array[:,np.int(n1*(N-1)/2)]))
    plt.xlabel('y [nm]')
    plt.ylabel('phase{E}')
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.savefig(filepath + 'array_phase_1D.pdf')
    plt.savefig(filepath + 'array_phase_1D.png')
    plt.clf()

    plt.plot(np.linspace(0,d1*1e9*N,n1*N),np.real(array[:,np.int(n1*(N-1)/2)]))
    plt.xlabel('y [nm]')
    plt.ylabel('Re{E}')
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.savefig(filepath + 'array_real_1D.pdf')
    plt.savefig(filepath + 'array_real_1D.png')
    plt.clf()

    plt.plot(np.linspace(0,d1*1e9*N,n1*N),np.imag(array[:,np.int(n1*(N-1)/2)]))
    plt.xlabel('y [nm]')
    plt.ylabel('Im{E}')
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.savefig(filepath + 'array_imag_1D.pdf')
    plt.savefig(filepath + 'array_imag_1D.png')
    plt.clf()

def crop_farray(dx,lamda,farray,NA_crop):
    kx_max = np.pi/dx
    k0 = 2*np.pi/lamda
    kx_k = np.linspace(-kx_max/k0, kx_max/k0, farray.shape[0])
    ky_k = np.linspace(-kx_max/k0, kx_max/k0, farray.shape[0])
    args_low = np.argwhere(kx_k > -NA_crop) 
    args_high = np.argwhere(kx_k[args_low] < NA_crop)
    args_cropped = args_high[:,0] + args_low[0]
    kx_k_cropped = kx_k[args_cropped]
    ky_k_cropped = ky_k[args_cropped]
    low = args_cropped[0]
    high = args_cropped[-1]
    
    farray_cropped = farray[low:high+1,low:high+1]
    return kx_k_cropped, ky_k_cropped, farray_cropped

def clip_circle(kx_k_cropped, ky_k_cropped, farray_cropped,NA):

    KX_cropped,KY_cropped = np.meshgrid(kx_k_cropped,ky_k_cropped)

    for i in range(farray_cropped.shape[0]):
        for j in range(farray_cropped.shape[0]):
            if np.sqrt(KX_cropped[i,j]**2+KY_cropped[i,j]**2)>=NA: farray_cropped[i,j] = 0

    return farray_cropped

def plot_fourier_space(kx_k_cropped,ky_k_cropped, image, NA, filepath, iteration_var):
    
    
    fig, ax = plt.subplots(figsize=(5.0,4.0))

    circ1 = Circle((0, 0), NA, facecolor='None', edgecolor='k')
    circ2 = Circle((0, 0), 1.0, facecolor='None', edgecolor='k')

    vmaxs = [1, 0.8, 0.6, 0.4, 0.2]
    #ax.add_patch(circ1)
    #ax.add_patch(circ2)
    print("sum:")
    #f = open("intensity_integral of SHG.txt","a+")
    #f.write("%d\r\n" % np.sum(np.sum(image)))
    #image = norma(image)
    plt.imshow(image,extent=[kx_k_cropped[0],kx_k_cropped[-1],ky_k_cropped[0],ky_k_cropped[-1]],cmap = 'viridis')#,vmax = vmaxs[int(iteration_var)])
    plt.axis('equal')

    plt.arrow(-1.3,-1.3,0.4,0,color='white',head_width=0.05, head_length=0.1)
    plt.arrow(-1.3,-1.3,0,0.4,color='white',head_width=0.05, head_length=0.1)
    plt.text(-0.75,-1.4,'$k_x$',color='white')
    plt.text(-1.4,-0.70,'$k_y$',color='white')
    #plt.xlabel('kx/k0')
    #plt.ylabel('ky/k0')
    plt.axis('off')
    #plt.title('|E|^2')

    #plt.colorbar(ticks=[0, 1])
    plt.colorbar()

    plt.savefig(filepath + 'fourier_abs_2D_'+str(iteration_var)+'.pdf')
    plt.savefig(filepath + 'fourier_abs_2D_'+str(iteration_var)+'.png')
    plt.clf()

    n_cropped = np.size(ky_k_cropped)
    plt.plot(ky_k_cropped, image[:,np.int(n_cropped/2)])
    plt.plot(ky_k_cropped, image[np.int(n_cropped/2),:])
    #plt.ylim((0,0.5))
    
    plt.axvline(x=NA,color='r')
    plt.axvline(x=-NA,color='r')
    plt.xlabel('ky/k0')
    plt.ylabel('|E|^{2}')
    plt.autoscale(enable=True, axis='x', tight = True)
    plt.savefig(filepath + 'fourier_abs_1D_'+str(iteration_var)+'.pdf')
    plt.savefig(filepath + 'fourier_abs_1D_'+str(iteration_var)+'.png')
    plt.clf()

    liney = image[:,np.int(n_cropped/2)]
    
    c = np.int(n_cropped/2*0.75/1.47)
    liney_cropped = liney[np.int(n_cropped/2)-c:c+np.int(n_cropped/2)]
    
    #print('line_y: ',np.max(liney_cropped))
    linex = image[np.int(n_cropped/2),:]
    linex_cropped = linex[np.int(n_cropped/2)-c:c+np.int(n_cropped/2)]
    #print('line_x: ',np.max(linex_cropped))
    #plt.plot(liney_cropped)
    #plt.show()
    #plt.clf()

def plot_real_space(image, d1,n1,fN1,rN1,N):


    # real space is plotted shifted because its the ifft of an fftshifted fft.The cropping is not symmetrical
    x_max = fN1*d1/n1
    
    dx = x_max/rN1
    
    n_max = np.int(d1*N/dx)

  
    image = image/np.max(np.max(image))
    fig, ax = plt.subplots(figsize=(5.0,4.0))
    
    offset = n_max/N*3 # just puts an offset that stays stable for different resolutions. For bigger offset just put change the constant 
    low = np.int(rN1/2-offset)
    high = np.int(rN1/2+n_max+offset)
    plt.imshow(image[low:high,low:high])
    plt.axis('equal')
    plt.axis('off')
    plt.colorbar(ticks=[0, 1])
    plt.savefig('PLOTS/phased_array_py/real_space_abs_2D.pdf')
    plt.clf()

def fresnel(n_i, n_t, theta_i, kind ):
    n1 = n_i
    n2 = n_t
    cos_theta_t = np.sqrt(1-(n1/n2*cmath.sin(theta_i))**2) 
    cos_theta_i = cmath.cos(theta_i)

    rs = (n1*cos_theta_i-n2*cos_theta_t)/(n1*cos_theta_i+n2*cos_theta_t)
    ts = 2*n1*cos_theta_i/(n1*cos_theta_i+n2*cos_theta_t)
    rp = (n2*cos_theta_i-n1*cos_theta_t)/(n2*cos_theta_i+n1*cos_theta_t)
    tp = 2*n1*cos_theta_i/(n2*cos_theta_i+n1*cos_theta_t)
    if kind == 'rs': return rs
    if kind == 'rp': return rp
    if kind == 'ts': return ts
    if kind == 'tp': return tp

def give_farfield(N,lamda,NA_crop,NA,delta,orientation,n_substrate, **kwargs):


        if NA > n_substrate : 
            print('ATTENTION: NA = '+str(NA)+' is physically prohibited.')
            print('NA = '+str(n_substrate)+' is assumed instead')
            NA = n_substrate

        cos = cmath.cos
        sin = cmath.sin
        
        k = 2*np.pi/lamda
        
        n1 = 1.0
        n2 = n_substrate

        
        if orientation == 'hor' : 
            PHI = 0
            THETA = np.pi/2
        if orientation == 'ver' : 
            PHI = np.pi/2
            THETA = np.pi/2
        if orientation == 'perp' : 
            THETA = 0
            PHI = 0
        if orientation == 'custom' :
            #print(kwargs)
            THETA = kwargs.get('THETA')
            PHI = kwargs.get('PHI')
            print('theta : ', THETA)
            print('phi : ', PHI)
            
        Is = []
        Es_array = []
        Ep_array = []
        theta_max = cmath.asin(NA_crop/n2)
        r_max = np.sin(theta_max)
        apod_array = []
        x_s = np.linspace(-r_max,r_max,N)
        y_s = np.linspace(-r_max,r_max,N)
        
        for y in y_s:

                for x in x_s:
                        r = np.sqrt(x**2+y**2)
                        theta = cmath.asin(r) 
                        phi = cmath.atan(y/x)   
                        theta_s = cmath.asin(n2/n1*sin(theta))
                        #print(theta_s)
                        PIs = np.exp(1j*k*n1*cos(theta_s)*delta)
                        c1 = (n2/n1)**2*cos(theta)/cos(theta_s)*fresnel(n1,n2,theta_s,'tp')*PIs
                        c2 = (n2/n1)*fresnel(n1,n2,theta_s,'tp')*PIs
                        c3 =  - n2/n1*cos(theta)/cos(theta_s)*fresnel(n1,n2,theta_s,'ts')*PIs
                        c = np.sqrt(1/cos(theta))

                        apodization_factor = 1/cos(theta)
                        Ep = (c1*cos(THETA)*sin(theta) + c2*sin(THETA)*cos(theta)*cos(phi-PHI))
                        Es = c3*sin(THETA)*sin(phi-PHI)
                        I = np.abs(apodization_factor*(Ep*np.conj(Ep)+Es*np.conj(Es)))
                        
                        apod_array.append(apodization_factor)
                        
                                

                        Is.append(I) 
                        Es_array.append(Es)
                        Ep_array.append(Ep)

                
            

        apod_array = np.asarray(apod_array).reshape(N,N)
        Es_array = np.asarray(Es_array).reshape(N,N)
        Ep_array = np.asarray(Ep_array).reshape(N,N)
        Is_array = np.asarray(Is).reshape(N,N)

        kx_max = np.sin(np.abs(theta_max))*n2
        kx_k = np.linspace(-kx_max,kx_max,N)
        ky_k = np.linspace(-kx_max,kx_max,N)
        
        Es_array = clip_circle(kx_k, ky_k,Es_array,NA)
        Ep_array = clip_circle(kx_k, ky_k,Ep_array,NA)       
        Is_array = clip_circle(kx_k, ky_k,Is_array,NA)    

        return Es_array, Ep_array, Is_array, apod_array

def give_incoherent_farfield(Nsize,lamda,NA_crop,NA, delta, pol_ratio,n_substrate):
    Es_array, Ep_array, Is_array = give_farfield(Nsize,lamda,NA_crop,NA, delta, orientation = 'ver', n_substrate = n_substrate)
    Es_array2, Ep_array2, Is_array2 = give_farfield(Nsize,lamda,NA_crop,NA, delta, orientation = 'hor', n_substrate = n_substrate)
    I_array =  np.abs(Es_array)**2 + np.abs(Ep_array)**2 + np.abs(Es_array2*pol_ratio)**2 + np.abs(Ep_array2*pol_ratio)**2
    return I_array

def norma(a):
    a = a/np.max(np.max(a))
    return a

def solve_farfield( cell, dx, N, lamda, lamda_exc,  angle_x_cell, angle_y_cell, angle_x_point, angle_y_point, std_cell, std_halfcell, std_point, ga_factor,g_ratio, NA, NA_crop, fN, rN, n_substrate, consider_polarization_projection, dipole_pol, distance_from_substrate ):
    
    array = make_array(N,N,cell)
    array = apply_linear_phase_per_point(array, dx, N, lamda, angle_x = angle_x_point, angle_y = angle_y_point, n_substrate = n_substrate)
    apply_linear_phase_per_cell(array, dx, N, lamda, angle_x_cell, angle_y_cell, consider_polarization_projection, n_substrate = n_substrate)
    array = apply_noise_per_cell(array,N, std_cell)
    array = apply_noise_per_halfcell(array,N, std_halfcell)
    array = apply_noise_per_point(array, std_point)

    array1 = apply_gaussian_mask(array, ga_factor,N, stepped = True)
    array2 = apply_gaussian_mask(array, ga_factor,N, stepped = False)

    array = array1*g_ratio + array2*(1-g_ratio)
    
    
    farray =  np.fft.fftshift(np.fft.fft2(array,s=[fN,fN]))
    kx_k_cropped, ky_k_cropped, farray_cropped = crop_farray(dx,lamda,farray, NA_crop)
    farray_cropped = clip_circle(kx_k_cropped, ky_k_cropped, farray_cropped,NA)
    Nsize = np.shape(farray_cropped)[0]
    Es_array, Ep_array, Is_array = give_farfield( Nsize, lamda, NA_crop, NA, delta = distance_from_substrate, orientation = dipole_pol, n_substrate = n_substrate)


    

    return array, kx_k_cropped, ky_k_cropped, farray_cropped, Es_array, Ep_array

def add_incoherent_background(I, lamda, NA_crop, NA, delta , pol_ratio , n_substrate, bkg_ratio):
    if bkg_ratio == 0 : return I
    I_array = I
    Nsize = np.shape(I)[0]
    I_incoherent = give_incoherent_farfield(Nsize, lamda, NA_crop, NA, delta = 20e-9, pol_ratio = pol_ratio, n_substrate = n_substrate)
    I_array_ = norma(I_array)
    I_incoherent = norma(I_incoherent)
    I_center_sum = np.sum(np.sum(I_array))
    I_incohsum = np.sum(np.sum(I_incoherent))
    I_incoherent =  I_incoherent*bkg_ratio*I_center_sum/I_incohsum
    I_array = I_array + I_incoherent
    return I_array

def read_cell(filename):
    f = open(filename)
    x = []
    y = []
    Ex = []
    Ey = []

    for line in f: 
        if line.split()[0] == '%': continue
        x.append(np.float(line.split()[0]))
        y.append(np.float(line.split()[1]))
        ex = line.split()[2]
        ey = line.split()[3]
        ex = ex.replace('i','j')
        ey = ey.replace('i','j')
        #print(ex)
        ex = complex(ex)
        ey = complex(ey)
        Ex.append(ex)
        Ey.append(ey)

    dx= x[1]-x[0]
    length = len(x)
    n = np.int(np.sqrt(length))
    x = np.asarray(x).reshape((n,n))
    Ey = np.asarray(Ey).reshape((n,n))
    Ex = np.asarray(Ex).reshape((n,n))
    y = np.asarray(y).reshape((n,n))
    
    return x, y, Ex, Ey, dx
    
def resample_cell(cell,n,dx):

    dx = dx*n
    cell_sampled = []
    nx = np.int(np.shape(cell)[0]/n)
    for i in range(nx):
        for j in range(nx):
            cell_sampled.append(cell[i*n,j*n])

    cell_sampled = np.asarray(cell_sampled).reshape((nx,nx))
    return cell_sampled, dx

def read_2D_comsol(filename):
    # can read a txt generated by comsol that has Ex and Ey complex values of a field in a 2D plane
    #returns 2D arrays of x,y,Ex,Ey the field values being complex
    f = open(filename)

    x = []
    y = []

    Ex = []
    Ey = []

    for line in f:
        if line.split()[0] == '%' : continue

        x.append(np.float(line.split()[0]))
        y.append(np.float(line.split()[1]))

        Ex.append(np.float(line.split()[2])+1j*np.float(line.split()[3]))
        
        Ey.append(np.float(line.split()[4])+1j*np.float(line.split()[5]))


        n = np.int(np.sqrt(len(x)))

    x = np.asarray(x).reshape((n,n))/1e-9
    y = np.asarray(y).reshape((n,n))/1e-9
    Ex = np.asarray(Ex).reshape((n,n))
    Ey = np.asarray(Ey).reshape((n,n))
    
    return x, y, Ex, Ey