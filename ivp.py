# Math and graphing tools
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from math import pi

class Initial_Value_Problem:
    
    def __init__(self,equation_name,NE,n,M,CFL,Nt,initial_functions_array,tol,shots,results_dir,graphs_dir):
        """
        This is the constructor of the Initial_Value_Problem object.

        Input: Number of Equations, number of qubits, number of frequencies
        Courant-Friedrich-Levy factor, number of time steps, initial value
        of each unknown function, path of the results directory and path
        of the plots directory.
        """
        # We define global useful quantities
        self.equation_name = equation_name # advection, wave
        self.NE = NE

        # Spatial domain
        self.n  = n
        self.N  = 2**n
        self.dx = 1.0/float(self.N)
        self.x  = np.arange(self.N)*self.dx

        # Temporal domain
        self.Nt = Nt
        self.CFL = CFL
        self.dt = CFL/float(self.N)
        self.t = np.arange(Nt)*self.dt

        # Space of parameters
        self.M = M
        self.NofParams = 2*M+1  # Number of free parameters 
        self.shots = shots
        self.tol = tol
        
        # ------ Initial conditions -------
        # Direct Space
        self.u0 = [initial_functions_array[i](self.x) for i in range(NE)]
        self.u_p = self.u0.copy()
        self.u_star = self.u0.copy()
        # Space of parameters
        self.l0 = np.zeros((self.NE,self.NofParams),dtype=float)
        self.set_initial_parameters()

        
        self.formalism = 2
        self.print_timeSteps = False

        self.results_paths = ["{}classical_n{}M{}.npy".format(results_dir,n,M),
            "{}state_vector_n{}M{}.npy".format(results_dir,n,M),
            "{}sampling_n{}M{}.npy".format(results_dir,n,M)]
        self.graphs_paths = {
        "2d":"{}2dPlot_n{}M{}.png".format(graphs_dir,n,M)
        }

        self.titlefont = 20
        self.axisfont = 18
        self.legendfont = 18
        self.ticksfont = 18

               
    def set_initial_parameters(self):
        """
        This method computes the vector of parameters of each
        equation at the initial time.
        
        The vector of parameters is defined as:

        [ Re{c_0}, Re{c_1}, ..., Re{c_{M}}, Im{c_1}, ..., Im{c_{M}} ]

        Where the c_i are the first M Fourier coeficients of f.
        """
        M = self.M
        # For each of the equations
        for i in range(self.NE):
            # We calculate the FFT of the initial condition
            u0_fft = np.fft.fft(self.u0[i])
            # We place the real parts first
            self.l0[i][:M+1] = np.real(u0_fft[:M+1])
            # We place the imaginary parts after
            self.l0[i][M+1:] = np.imag(u0_fft[1:M+1])

    
    def FastStateVector(self,l_loc):
        """
        Input: Vector of parameters l.

        Resurns: The function in represents in the direct space
        on a numpy array.
        """
        N,M = self.N,self.M
        # We are going to rebuild the Fourier Transform
        # but with truncated terms set to 0
        u_loc_fft = np.zeros(N,dtype=complex)
        
        u_loc_fft[0] = l_loc[0]
        # Positive frequencies
        u_loc_fft[1:M+1] = l_loc[1:M+1] + 1j * l_loc[M+1:]

        # Negative frequencies
        u_loc_fft[-M:] = u_loc_fft[M:0:-1].conj()

        u_loc = np.fft.ifft(u_loc_fft)
        
        return np.real(u_loc)


    def evolveClassically(self):
        """
        This function returns the evolution as a function of time.
        It uses the concept of sparse matrix from scipy, which is 
        a matrix with a lot of ceros
        """
        # Rename useful quantities
        N = self.N
        NE = self.NE
        dx = self.dx
        dt = self.dt
        Nt = self.Nt
        
        u = np.zeros((NE,Nt,N))

        u[:,0,:] = self.u0.copy()
        def rhs(u_loc):
            rhs = np.zeros((NE,N))
            if self.equation_name == 'wave':
                rhs[0] = 0.5*(np.roll(u_loc[1],-1) - np.roll(u_loc[1],1))/dx
                rhs[1] = 0.5*(np.roll(u_loc[0],-1) - np.roll(u_loc[0],1))/dx
                rhs[2] = u_loc[0]
            elif self.equation_name == 'advection':
                rhs[0] = -0.5*(np.roll(u_loc[0],-1) - np.roll(u_loc[0],1))/dx
            return rhs

        for i in range(1,Nt):
            u_star = u[:,i-1,:] + dt*rhs(u[:,i-1,:])
            u[:,i,:] = 0.5*(u[:,i-1,:]+u_star + dt*rhs(u_star))
            
        np.save(self.results_paths[0], u)
        
    def evolveWithStateVector(self,sampling=False):
        """
        This is the main loop of the VQA. If we were using
        a real quantum computer, we would just keep track of
        the parameters as functions of time. In this implementation,
        since we are computing everything on a classical computer,
        it saves some time to save the function in the direct space.
        The result is exactly the same, the cost function gives the
        exact same result, the execution is just way faster.
        """
        # Auxiliary parameter used in CF
        self.formalism = 2 if sampling else 1
        # Rename useful quantities
        NE = self.NE
        N  = self.N
        Nt = self.Nt
        NofParams = self.NofParams
        
        # We will store the all the functions at each time step
        u = np.zeros((NE,Nt,N))
        u[:,0,:] = self.u0
        
        # We will keep track of the previous and interminiate
        # sets of parameters
        l_p = self.l0.copy()
        l_star = self.l0.copy()
        
        # Keeping track of the functions they represent is (optional)
        # but it helps save time when calling the CF.
        self.u_star = np.zeros((NE,N))
        self.u_p = np.zeros((NE,N))
        
        print("Evolution starts") if (self.print_timeSteps) else None

        # Loop Core
        for nTime in range(1,Nt):
            # We recall what the function was in 
            # the last time step (optional)
            self.u_p = u[:,nTime-1,:]
            
            # First rk step
            for EqNum in range(NE):
                # Find parameters
                l_star[EqNum] = minimize(self.CF, l_p[EqNum],args=(1,EqNum), 
                                         method="nelder-mead",tol=self.tol).x
                # Save wave function (optional)
                self.u_star[EqNum] = self.FastStateVector(l_star[EqNum])
            
            # Second rk step
            for EqNum in range(NE):
                # Find parameters
                l_p[EqNum] = minimize(self.CF, l_star[EqNum],args=(2,EqNum), 
                                      method="nelder-mead",tol=self.tol).x
                # Save wave function (optional)
                u[EqNum,nTime,:] = self.FastStateVector(l_p[EqNum])
            
            print("Time step number {}".format(nTime)) if (self.print_timeSteps) else None
        
        np.save(self.results_paths[self.formalism],u)
        
        print("Evolution finished") if (self.print_timeSteps) else None
    
    def evolveWithSampling(self):
        self.evolveWithStateVector(sampling=True)

    # Classical dot product
    def Cdot(self,theta_state,phi_state,power=0):
        """
        This function calculates products of the form:
                Re{ < 0| U†(θ) ⊕^p U(φ)|0 > }
        but classically and much faster

        Input: theta_state, phi_state, p
        Output: Re{ < 0| U†(θ) ⊕^p U(φ)|0 > }
        """
        power = round(power)

        # Apply the shift operator
        phi_state = np.roll(phi_state,-power)

        # Compute the dot product
        return np.real(np.dot(np.conj(theta_state),phi_state))

    # Dot product but with sampling error, just what
    # the hadamard test would output
    def Sdot(self,theta_state,phi_state,power=0):
        """
        This function calculates products of the form:
                Re{ < 0| U†(θ) ⊕^p U(φ)|0 > }
        fast and then simulates monte carlo sampling

        Input: theta_state, phi_state, p
        Output: Re{ < 0| U†(θ) ⊕^p U(φ)|0 > }
        """
        # We will need to normalize the states to include
        # the sampling
        theta_norm = np.sum(theta_state**2)**0.5
        phi_norm = np.sum(phi_state**2)**0.5
        
        # Compute the classical dot product but normalize
        # the result
        exact_expVal = self.Cdot(theta_state,phi_state,power)/(theta_norm*phi_norm)

        # The probability of measuring 0 in the ancilla qubit

        P0 = 0.5*(1.0+exact_expVal)

        if abs(exact_expVal) > 1.001:
            print("alerta")
        if (P0 > 1.0):
            P0 = 1.0
        elif (P0 < 0.0):
            P0 = 0.0
        
        
        # Simulate that we execute (self.shots) times a poisson
        # experiments with P0 probability of sucess.
        counts_of_0 = np.random.binomial(self.shots, P0, 1)[0]
        
        # Aproximate the expectation value
        measured_expVal = -1.0 + 2.0*(counts_of_0)/float(self.shots)
        
        # Return the result with the norms again
        return measured_expVal*theta_norm*phi_norm

    # This is the Cost Function
    def CF(self,params,rk,EqNum):
        """
        This is the Cost Function. Since we are implementing
        a Runge-Kutta of order 2, there will be 2 RK steps.

        Input: params is the vector of parameters,
            rk = 1 or 2 is the step of the Runge-Kutta,
            EqNum = 0,1,2,etc represents the number of the 
            equation being evolved. 
        Output: CF
        """
        # Compute the u funciton at this current i step,
        # that is why it is called ui
        ui = self.FastStateVector(params) # 1d numpy array
        
        # Set of the functions at the previous time step
        u_p = self.u_p # set of 1d numpy arrays (2d numpy array)
        
        # Set of functions at the intermidiate time step
        u_star = self.u_star # set of 1d numpy arrays (2d numpy array)
        
        CFL = self.CFL
        dt  = self.dt

        # Dot function, can be Cdot or Sdot
        def dot(theta_state,phi_state,power=0):
            # self.formalism is a global integer that determines
            # the formalism being used
            # 1 -> State Vector Formalism
            # 2 -> Sampling Error Formalism
            if(self.formalism==1):
                return self.Cdot(theta_state,phi_state,power)
            elif(self.formalism==2):
                return self.Sdot(theta_state,phi_state,power)
            
        
        def rhs(ui,u_loc,EqNum):
            """
            Computes the product:
                   Δt < ui | rhs(u_loc) >
            u_loc can be u_p or u_star
            
            Input:
                ui: 1d numpy array containing the function at
                    the current time step
                u_loc: set of 1d numpy arrays containing values
                    of the system of functions computed at the
                    previous time step or at the intermidiate
                    Runge-Kutta step
            """
            if self.equation_name == 'wave':
                if EqNum == 0:
                    rhs = 0.5 * CFL * (dot(ui,u_loc[1],1) - dot(ui,u_loc[1],-1))
                elif EqNum == 1:
                    rhs = 0.5 * CFL * (dot(ui,u_loc[0],1) - dot(ui,u_loc[0],-1))
                else:
                    rhs = dt * (dot(ui,u_loc[0]))
            elif self.equation_name == 'advection':
                rhs = - 0.5 * CFL * (dot(ui,u_loc[0],1) - dot(ui,u_loc[0],-1))
            return rhs

        if rk == 1:
            result = dot(ui,ui) - 2.0 * dot(ui,u_p[EqNum]) - 2.0 * rhs(ui,u_p,EqNum)
        elif rk == 2:
            result = dot(ui,ui) - dot(ui,u_p[EqNum]) - dot(ui,u_star[EqNum]) \
                   - rhs(ui,u_star,EqNum)
        
        return result 

    def plotEvolution(self,EqNum,formalisms=[0,1,2]):
        """
        Input: Index of the equation to be ploted, list of formalisms
        to be plotted

        Ouput: A graph with three 2D subplots
        """
        N = self.N
        Nt = self.Nt
        u = np.zeros((3,Nt,N))
        for i in formalisms:
            u[i] = np.flipud(np.load(self.results_paths[i])[EqNum])
    
        fig, ax = plt.subplots(nrows = 1,ncols=3,figsize=(12, 4))
        
        plt.subplots_adjust(left=0.07, right=0.95,wspace=0.1,
                            hspace=0.4,top=0.95,bottom=0.1)
        
        xmin = -0.5*self.dx
        xmax = 1.0+xmin
        tmin = -0.5*self.dt
        tmax = tmin + self.dt*Nt
        aspect_ratio = (xmax-xmin) / (tmax-tmin)
        vmax = np.max(np.abs(u[:2,:,:]))

        for i in range(3):
            im = ax[i].imshow(u[i], cmap='coolwarm',
                              extent=[xmin,xmax,tmin,tmax], 
                              vmin=-vmax,vmax=vmax,aspect=aspect_ratio) 
        
        ax[0].set_title("Classical",fontsize=self.titlefont)
        ax[1].set_title("SVF",fontsize=self.titlefont)
        ax[2].set_title("SEF",fontsize=self.titlefont)
        for i in range(3):
            ax[i].set_xlabel('x', fontsize=self.axisfont)
            ax[i].tick_params(axis='both', labelsize=self.axisfont)
            ax[i].set_xticks(np.arange(0, 1, 0.5))
        ax[0].set_ylabel('t', fontsize=self.axisfont)
        for i in range(1,3):
            ax[i].yaxis.set_ticklabels([])  # Set tick labels to an empty list

        # Use any of the images to generate the colorbar
        cbar = fig.colorbar(im, ax=ax.ravel().tolist(),shrink=0.8)  
        cbar.ax.tick_params(labelsize=self.ticksfont)
        
        fig.savefig(self.graphs_paths["2d"], dpi=300)
        plt.show()

    def plotState(self,u_loc):
        """
        Input: Numpy array

        Output: Graph in the direct space
        """
        fig, ax = plt.subplots(figsize=(5, 4))    
        ax.plot(self.x,u_loc, c='#E74C3C',lw=2)
        
        ax.set_xlabel('x', fontsize=self.axisfont)
        ax.set_ylabel('psi', fontsize=self.axisfont)
        ax.set_title("|psi (x) >",fontsize=self.titlefont)
        ax.tick_params(axis='both', labelsize=self.ticksfont)
    
        plt.show()


def convergeceTest(list_n,list_M,EqNum,formalisms,u_exact,results_path,path_for_graph):
    """
    This is a global function. The Initial_Value_Problem object
    is helpful to find the solution of the IVP with one particular
    n, M, etc. Here, we need information about executions with
    different n or M values.

    Input: 
    list_n: is a list of the values n that you want to compare
    list_M: is a list of the M value that corresponds to each
            n value.
    formalisms: list of formalisms to be plotted
    u_exact: function that takes in x (1d numpy array) and t (scalar)
            returns the value of the function evaluated in the
            discrete domain at time t.
    results_path: Directory where the results of the executions
            are found.
    path_for_graph: Directory where you want to save your graph.

    """
    # We are going make three graphs, one for each formalism.
    # if one formalism is not included in the formalisms list,
    # the graph will be empty
    NofGraphs = 3
    
    fig, ax = plt.subplots(nrows = 1,ncols=NofGraphs,figsize=(10, 4))
    plt.subplots_adjust(left=0.12, right=0.95,wspace=0.1,hspace=0.4,top=0.9,bottom=0.2)

    formalisms_names = ["classical","state_vector","sampling"]

    # List of numerical solutions with each formalism
    f_num = [0,0,0]

    # Maximum value of the L1 norm, for plotting purposes
    vmax = 0.0
    # For every execution
    for k in range(len(list_n)):
        n = list_n[k]
        M = list_M[k]
        N = 2**n
        
        for i in formalisms:
            f_num[i] = np.load("{}{}_n{}M{}.npy".format(
                results_path,formalisms_names[i],n,M))[EqNum]
        
            Nt = f_num[i].shape[0]
        dx = 1.0/float(N)
        dt = 0.5*dx
        x = np.arange(N)*dx
        t = np.arange(Nt)*dt

        # Error as a function of time for every formalism
        error = np.zeros((NofGraphs,Nt))
        
        # For each formalism
        for i in formalisms:
            # Compute the L1 norm of the error at each time step
            for j in range(Nt):
                # Compute the exact funtion at time t
                f = u_exact(x,t[j]) # 1d array
                error[i,j] = np.sum(np.abs(f-f_num[i][j]))*dx

            # Plot the L1 norm as a funtion of time
            ax[i].plot(t,error[i],label = "n = {}".format(n))

            # Maximum value, no make all graphs be coherent
            if (1.1* np.max(error) > vmax):
                vmax = 1.1* np.max(error)
    
    # ========================================
    # ------------- Graph --------------------
    # ========================================
    
    # Font sizes
    titulo = 23
    ejes = 20
    legendSize = 15
    ticksSize = 20
    titles = ["Classical","SVF","SEF"]

    # Adjust each graph
    for i in range(NofGraphs):
        ax[i].set_title(titles[i],fontsize=titulo)
        ax[i].set_xlabel('t', fontsize=ejes)
        ax[i].tick_params(axis='both', labelsize=ejes)
        ax[i].legend(fontsize=legendSize,loc='upper right')
        ax[i].set_ylim(0,vmax)
        if i >= 1:
            ax[i].yaxis.set_ticklabels([])  # Set tick labels to an empty list
    
    ax[0].set_ylabel('L1 norm', fontsize=ejes)
    
    fig.savefig(path_for_graph, dpi=300)
    plt.show()
