"""
This is a toy model of a lithium-ion battery. Code defines particles with their respective masses and  
charges, and calculates dynamics considering only Coulomb forces and Newton's law. Moreover, code calculates
temperature in both anode and cathode zones, using basic electromagnetism equation. User must select amount of particles, kind of cathode and electrolyte, for running the simulation
"""
from abc import ABC, abstractmethod
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, RadioButtons
#Abstract class
class Particle(ABC):
    """
Class Particle is a father class, that indicates dynamical and electronical properties to different particles  
"""
    def __init__(self, position, velocity, mass, charge):
        self.position = np.asarray(position, float) #np.asarray is used for converting different kinds of data (as lists or tuples) into arrays, the numpy kind of data. In this case, is used because tridimensional positions will be presented
        self.velocity = np.asarray(velocity, float)
        self.mass = float(mass)
        self.charge = float(charge)

    @abstractmethod
    def color(self) -> str:
        pass #This abstract method asks classes to define their colors, since no method for it is defined.
   #Color is not known by this class, and other classes must implement it when applying this class 

class Electron(Particle):
    """
    Electron properties and methods are given in this class, using heritage of Particle
    """
    def __init__(self, position): #Initial velocity, mass and charge are inner to electrons, so, we only accept external definition of position
        super().__init__(position, np.zeros(3), mass=1.0, charge=-1.0) #Initial velocity is zero in all directions
         #Super is used for bringing Particle methods for dynamical and electronic properties
        """
        Avoiding system collapses, electron mass and charge will be normalized as 1
        Electrons will be blue if not orbiting, and cyan if orbiting
        """
        self.orbiting = False #We introduce an orbiting characteristic for using it later in simulation
        self.orbit_center = None #Both properties are defined just for future use in anode
        self.orbit_angle = 0.0
    def color(self):
        if self.orbiting:
            return 'cyan'
        else:
            return 'blue' #Electron is blue if not orbiting
class LithiumIon(Particle):
    """
    This class provides lithium ions properties, with a charge 1265 times greater than electrons, and same but opposite charge
    Lithium ions will be red
    """
    def __init__(self, position):
        super().__init__(position, np.zeros(3), mass = 1265.0, charge = 1.0)
    def color(self): #We define the abstract method color for lithium ions as red
        return 'red'
class Anode(Particle):
    """
    Three different kinds of anodes will be used. So, in this case, no fixed charge will be defined, but constant 1e9 mass will be defined
    """
    def __init__(self, position, charge):
        super().__init__(position, np.zeros(3), mass = 1e9, charge=charge) #Curiosity is position doesn't need position=position, but charge does
    def color(self):
        return 'limegreen'

#Now, particles have been all created, and their properties have been created.

#--------------------------------------------------------------------------------------------------------------------------------------------------
#CONSTANTS
BIAS_FORCE = {"EC": 9.0, "LiPF6": 22.0, "DMC": 14.0}
"""
Biasing corresponds to a fixed electrical force applied over the system, 
mainly due to electrolyte attractive force. Three different electrolytes are
defined, with LiPF6 the greatest force.
"""     
Anode_Q = {r"$CoO_4$":4.0, "Li_metal": 6.0, "Silicon": 3.0}
"""
Three different materials are defined for anode, without considering their stability, but their approximate charges in terms of vacancies
"""
K = 1.0 #Normalized Coulomb constant
Eps = 0.003 #Softening factor for avoiding division by zero in coulomb's law
Dt         = 0.006      # time per step
Box        = 1.0        # battery size
Max_acc    = 500.0      # maximum acceleration for avoiding crashing
Damp       = 0.9985     # Percentage of every velocity for avoiding crashing
Barrier_x  = 0.5        # Electrons cannot pass this limit
Orbit_r    = 0.04       # Effective radius for anode to catch electrons
Orbit_w    = 12.0       # angular frequency for electrons
V_Bias = 1.5 #This will work as an external applied voltage
R_0= 0.5 #Arbitrary reference resistance
T_0 = 300.0 #Arbitrary temperaure in Kelvin, approimately environmental temperature
Alpha = 0.004 #Thermal coefficient of resistance
#-------------------------------------------------------------------------------------------------------------------------------------------------
class LiIonBattery:
   """
   This class defines battery toy model initial conditions, in terms of number of every particle, electrolyte, anode, time of simulation, number of exit electrons and number of exits. Moreover, initial random positions are defined.
   After this definitions, it calculates forces over every particle.
   """
   def __init__(self, n, electrolyte, anode): #This values will be defined in other step with buttons
       self.n = n
       self.electrolyte = electrolyte
       self.anode = anode
       self.bias_force = BIAS_FORCE[electrolyte]
       self.anode_charge = Anode_Q[anode]
       self.t = 0.0 #Simulation time
       self.run_exit = 0.0 #Electrons crossing in this run
       self.exited = [] #Crossed by every run
       
       self.temp_left = [] #This is used in the future for temperature measurement
       self.temp_anode = [] #[t, T].

       self.initial_particles() #This init is not the constructor, is a class method that will be defined in next line
   def initial_particles(self):
       """
       Random positions will be given for every particle. Electrons and ions are initially located at x<0.4, while anode particles are located randomly in x=0.95, but in different y and z positions
       """
       self.electrons = []
       self.ions = []
       self.anodes = []
       for i in range(self.n):
           self.electrons.append(Electron(np.random.rand(3)*[0.4, 1, 1]))
           self.ions.append(LithiumIon(np.random.rand(3)*[0.4,1,1]))
           self.anodes.append(Anode([0.95, np.random.rand(), np.random.rand()], self.anode_charge)) #rand takes values from zero to one
       self.all = self.electrons + self.ions + self.anodes
           #self.all is the reason why lists of positions were made previously. When calling every Electron(position), and others, it was brought the full amount of properties to the list. When adding them, the whole data is kept in self.all, allowing a simple use for charge and forces calculations. Every element already has information about every particle
   def forces(self):
       """
       Calculated force over every particle, considering electrical force and biasing.
       """
       N = len(self.all) #Number of particles
       P = np.zeros((N,3)) #Creates a zeros list of N rows and 3 columns (3 directions)
       Q = np.zeros(N)
       for i in range(N):
           P[i] = self.all[i].position
           Q[i] = self.all[i].charge #They fill the P and Q lists with every particle positions (rows of P), and charge (Every element of Q)
       diff = np.zeros((N,N,3))#This creates a list of N elements, each one with N elements, and every one with three elements. This will give N lists (one per particle), with N distance (One per particle), and every distance has three components
       for i in range (N):
           for j in range (N):
               diff[i][j] = P[i]-P[j]
       r2 = np.sum(diff**2, axis=2) + Eps**2 #We take every element squared, in the axis 2 (3 elements list), add the squared values, and add eps**2, avoiding same particles differences problems
       inv3 = np.zeros((N,N))
       for i in range(N):
           for j in range(N):
               if i != j:
                   inv3[i][j]=r2[i][j]**(-1.5) #This is made for two reasons: The consideration of r_hat, and avoiding same particle made forces
        
       F_Coulomb = np.zeros((N,3)) #A list for N particles, and the resultant force over it
       for i in range(N):
           for j in range(N):
               F_Coulomb[i]+= K*Q[i]*Q[j]*diff[i][j]*inv3[i][j]
        
       F_bias = np.zeros((N,3))
       for i in range(N):
           F_bias[i][0] = self.bias_force*np.sign(Q[i])
       return F_Coulomb+F_bias
   def boundary(self, p):
       """
       Three rules will be defined:
       - y and z walls, where particles will bounce
       - electrons will exit in left wall, when touching it, and will be reinjected in right side.
       - electron exits right wall, and is slowly reinjected in right side
       - ions reflect on both walls
       """           
       for ax in (1,2): #It iterates on axis 1 and 2, that means, axis y and z
           if p.position[ax]<0:
               p.position[ax] = 0.0
               p.velocity[ax] *= -1
           if p.position[ax] > Box:
               p.position[ax] = Box
               p.velocity[ax]*=-1    
       if isinstance (p, Electron): #It verifies if particle p is an electron
           if 0.45 < p.position[0] < 0.55 and p.velocity[0] > 0:
                p.position[0] = 0.45
                p.velocity[0] *= -1
           elif p.position[0]<=0 and 0.4<p.position[1]<0.6 and 0.4<p.position[2]<0.6:
                self.run_exit += 1
                p.position[0] = 0.99
                p.position[1:] = np.random.uniform(0.3, 0.7, 2)
                p.velocity[:]=np.zeros(3)
                p.orbiting = False
           elif p.position[0]<=0:
                p.position[0]=0.0
                p.velocity[:]*=-1     
           elif p.position[0] >= Box:
                p.position[0]= 0.99
                p.position[1]=p.position[1]
                p.position[2]=p.position[2]
                p.velocity[:] = np.zeros(3)
                p.orbiting = False    
       else:#Verifies other particles limits
            if p.position[0]<0:
                p.position[0] = 0.0
                p.velocity[0] *=-1
            if p.position[0] > Box:
                p.position[0]=Box
                p.velocity[0]*=-1
   def check_orbit(self,e): #e will represent electrons
       """
       Checks if electrons are close enough to Anode's for orbiting around them
       """
       for a in self.anodes:
           dist = np.sqrt((e.position[0]-a.position[0])**2+(e.position[1]-a.position[1])**2+(e.position[2]-a.position[2])**2)
           if dist < Orbit_r and not e.orbiting: #If it is inside the orbiting, and orbiting is false, electron is not orbiting:
               e.orbiting= True
               e.orbit_center = a.position.copy()
               e.orbit_angle = np.random.rand()*2*np.pi #Orbits at a random angle, around anode particle, and now, it is cyan
    
   def step(self): #This defines the different steps conditions, and the amount of steps per time
      """
      The function computes acceleration, updates position, applies boundary conditions, chechs orbits, and updates velocity. When electrons are orbiting, they are ignored from forces calculations, as well as anodes
      """ 
      F = self.forces()
      for i, p in enumerate(self.all):
          if isinstance(p, Anode):
              continue
          if isinstance(p, Electron) and p.orbiting:
              p.orbit_angle += Orbit_w * Dt #Makes electron rotate
              p.position[0]=p.orbit_center[0]
              p.position[1] = p.orbit_center[1]+Orbit_r*np.cos(p.orbit_angle)
              p.position[2] = p.orbit_center[2]+ Orbit_r*np.sin(p.orbit_angle)
              continue
          acc = F[i]/p.mass #It defines acceleration for every particle in terms of the forces
          for i in range(3):
              if abs(acc[i]) > Max_acc:
                  acc[i] = Max_acc*np.sign(acc[i])
          p.position += p.velocity*Dt + 0.5*acc*Dt**2
          #Defines position in terms of classical cinematic equations

          self.boundary(p) #Boundary was waiting for receiving an argument p, and it was given by the enumerate

          if isinstance(p, Electron):
              self.check_orbit(p) #Checks if any electron is now orbiting

          p.velocity = (p.velocity+acc*Dt)*Damp #Calculates new velocity
      self.t += Dt  #Adds the step and the time, with updated positions
      self.record_temperature()
   def electrical(self):
       """
       Calculates electrical properties and temperature, following
       I = total_crossings / time
       R = V/I
       T = T_0+I²R / (alpha*R_0) Where T_0 and R_0 are reference temperature and resistance. And alpha is thermal coefficient of resistance
       """
       probe = np.array([0.5, 0.5, 0.5]) #Central point of measurement
       total_crossings =  sum(self.exited)+self.run_exit #Counts total electrons that have crossed
       I = total_crossings / (self.t +Dt) #Avoids division by zero at self.t = 0
       V_Coulomb = 0.0
       for p in self.all:
           r = np.sqrt((p.position[0]-probe[0])**2 +
                       (p.position[1]-probe[1])**2+
                       (p.position[2]-probe[2])**2) + Eps
           V_Coulomb += K*p.charge / r
       V_total = V_Bias+V_Coulomb
       R = V_total / (I+1e-9) #Avoiding zero division error
       T = T_0 + (I**2*R)/(Alpha*R_0) 
       return I, R, T, V_total
   def record_temperature(self):
       """
       Records temperature in terms of electrons moving in every side of the battery
       """
       if self.temp_left and self.t - self.temp_left[-1][0] < 0.05: #If any time has transcurred, but it isn't 0.05 different from past one, then, jus tignore
           return 
       I, R, T, V = self.electrical()
       n_left = 0
       n_anode = 0
       for e in self.electrons:
           if e.position[0] < 0.4:
               n_left += 1
           if e.position[0] > 0.7:
               n_anode += 1
       w_left = n_left / len(self.electrons) #A fraction of electrons in left side
       w_anode = n_anode / len(self.electrons)

       self.temp_left.append((self.t, T_0 + (T-T_0)*w_left)) #Temperature as a fraction of total temperature change
       self.temp_anode.append((self.t, T_0+(T-T_0)*w_anode))
   def end_run(self):
       """
       Simulation is ended after 5 seconds.
        """
       self.exited.append(self.run_exit)
       self.run_exit = 0#Restarts the counting
       self.t = 0
       self.initial_particles() #Executes again from random choose of positions
   def  gaussian_fit(self):
       """
       Fits a Gaussian curve for exited electrons.
       It will give two values sigma and mu, where
       -sigma = how consistent battery data is
       -mu = average value
       It need three runs for showing the curve
       """
       if len(self.exited) < 3:
           return None
       data = np.array(self.exited)
       mu, sigma =  norm.fit(data) #Use of scipy

       #Here, a smooth x-axis is fixed for the curve
       x_min = max(0, mu -4*sigma)
       x_max = mu +4*sigma
       x = np.linspace(x_min, x_max, 200)
       pdf = norm.pdf(x, mu, sigma) #norm.pdf gives a normal distribution or PDF(Probability Density Function)

       return x, pdf, mu, sigma

#PLOT
fig = plt.figure(figsize=(15,8))
fig.patch.set_facecolor('#000000') #Sets full plot color
gs = gridspec.GridSpec (2,3, figure=fig,
                        hspace = 0.45, wspace=0.35, #wspace gives the space between subplots
                        left = 0.05, right=0.97,
                        top = 0.93, bottom=0.22)


ax3d = fig.add_subplot(gs[0,0], projection='3d') #Locate battery simulation the grid in row 0, column 0
axG = fig.add_subplot(gs[0,1]) #Gaussian plot
axT = fig.add_subplot(gs[0,2]) #Temperature
axI = fig.add_subplot(gs[1,:]) #Rest of the plot for information about simulation

#Style for 2D panels
for ax in (axG, axT, axI):
    ax.set_facecolor('#0D0D1A')
    ax.tick_params(colors = 'white')
    for spine in ax.spines.values(): #Characteristics of borders and other elements around the graph
        spine.set_color('#333333')

ax3d.set_facecolor('#0D0D1A')
fig.suptitle('Lithium ions toy model battery', color='White', fontsize=13, fontweight='bold')


#Buttons
def make_radio(localization, labels, title):
    """
    Creates buttons, considering:
    -localization: position where buttons will be placed [left, bottom, widht, height]
    -labels = options
    -title = label shown in the button
    """
    ax = plt.axes(localization, facecolor='#1A1A2E')
    b = RadioButtons(ax, labels, activecolor='cyan')
    ax.set_title(title, color='white', fontsize=8)
    for lb in b.labels:
        lb.set_color('white')
    return b

#Controls or buttons
b_n = make_radio([0.04, 0.03, 0.10, 0.13], ('5', '10', '15'), 'Number of particles')
b_el = make_radio([0.17, 0.03, 0.10, 0.13], ('EC', 'LiPF6', 'DMC'), 'Electrolyte')
b_an = make_radio([0.33, 0.03, 0.10, 0.13], (r'$CoO_4$', 'Li_metal', 'Silicon'), 'Anode')

#Pause button
ax_pause = plt.axes([0.76,0.05,0.09,0.06])
ax_pause.set_facecolor('#1A1A2E')
btn_pause = Button(ax_pause, 'Pause/Resume', color='#1A1A2E', hovercolor='#333333')
btn_pause.label.set_color('White')

#Now, a save button, for saving figures
ax_save = plt.axes([0.87, 0.05,0.09, 0.06])
ax_save.set_facecolor('#1A1A2E')
btn_save=Button(ax_save, 'Save figures', color='#1A1A2E', hovercolor='#333333')
btn_save.label.set_color('white')


#Simulation state
Battery = [LiIonBattery(10, 'EC', r'$CoO_4$')]
paused = [False]

def rebuild(*_): #Ignora los datos que se le den
    """
    Restart battery when any button changes
    """
    n = int(b_n.value_selected) #The number of particles selected
    el = b_el.value_selected
    an = b_an.value_selected
    Battery[0] = LiIonBattery(n, el, an) #Defines a first value in list for every pressing button, restarting the simulation

b_n.on_clicked(rebuild) #When pressed any n, function rebuild is applied
b_el.on_clicked(rebuild)
b_an.on_clicked(rebuild)
btn_pause.on_clicked(lambda _: paused.__setitem__(0, not paused[0])) #This is equivalent to write paused[0]=[False]

def savefigures(event): #Save plots
    """
    Saves 3D simulation panel and Gaussian-temperature panel in separate PNG files in current directory
    """
    extent_3d = ax3d.get_window_extent().transformed(fig.dpi_scale_trans.inverted()) #Transforms pixels measurement into real big image coordinates
    #Extent saves the axes space qualities
    fig.savefig('Battery.png', bbox_inches=extent_3d.expanded(1.2,1.2), dpi=150) #DPI : Dots per inch

    extent_g = axG.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig('BatteryGaussian.png', bbox_inches=extent_g.expanded(1.4,1.4),dpi=150)
    print('Saved: Battery.png and BatteryGaussian.png')
btn_save.on_clicked(savefigures)


#Animation
def update(_):
    """
    Called by FuncAnimation on every frame 
    """
    if paused[0]:
        return
    
    B = Battery[0]
    for i in range(4): #Four steps done by every frame
        B.step()
    
    if B.t >= 5.0:
        B.end_run()
    I,R,T,V = B.electrical()

    #Graph 3D
    ax3d.cla() #Clean axes
    ax3d.set_facecolor('#0D0D1A')
    ax3d.set_xlim(0, 1); ax3d.set_ylim(0, 1); ax3d.set_zlim(0, 1)
    ax3d.set_xlabel('x', color='white')
    ax3d.set_ylabel('y', color='white')
    ax3d.set_zlabel('z', color='white')
    ax3d.tick_params(colors='white')
    ax3d.set_title(f"t={B.t:.2f}s  run={len(B.exited)+1}  crossings={B.run_exit}",
                   color='white', fontsize=9)

    #Draw particles
    for p in B.all:
        ax3d.scatter(p.position[0], p.position[1], p.position[2], color = p.color(), s=22 if isinstance(p, Anode) else 10, depthshade=False)


    #Gaussian plot
    axG.cla()
    axG.set_facecolor('#0D0D1A')
    axG.set_title("Electrons per run", color='white', fontsize=9)
    axG.set_xlabel("Crossings per run", color='white')
    axG.set_ylabel("Density",           color='white')
    axG.tick_params(colors='white')   
    result = B.gaussian_fit()
    if result:
        x, pdf, mu, sigma = result
        #Histogram of every run
        axG.hist(B.exited, bins='auto', density=True, color='#00B1DD', alpha=0.45, label='Runs') #bins are intervals of division
        
    #Gaussian curve
        axG.plot(x, pdf, 'r-', lw=2, label = f'Average: {mu:.2f}. Spread: {sigma:.2f}')
        axG.legend(fontsize=8, labelcolor='white', facecolor='#1A1A2E')
    else:
        remaining = 3-len(B.exited)
        axG.text(0.5, 0.5, f'{remaining} more runs are required', ha='center', va='center', transform=axG.transAxes, color='#DDDDDD', fontsize=10)

    #Temperature plot
    axT.cla()
    axT.set_facecolor('#0D0D1A')
    axT.set_title("Temperature", color='white', fontsize=9)
    axT.set_xlabel("Time (s)", color='white')
    axT.set_ylabel("T (K)",    color='white')
    axT.tick_params(colors='white')   

    if B.temp_left:
        #This will separate temperature and time from lists
        ts, Tl = zip(*B.temp_left)  #This unpacks, that means, opposite process of zip
        axT.plot(ts, Tl, color='cyan', lw=1.5, label='Left side')
    if B.temp_anode:
        ts, Ta = zip(*B.temp_anode)
        axT.plot(ts, Ta, color='Orange', lw=1.5, label='Anode') 
    axT.legend(fontsize=8, labelcolor='white', facecolor='#1A1A2E')
    
    #Information bar
    axI.cla()
    axI.set_facecolor('#0d0d1a')
    axI.axis('off')   # no axes needed
    axI.text(0.02, 0.65, f"I = {I:.4f} A",      color='cyan',   fontsize=11, transform=axI.transAxes)
    axI.text(0.20, 0.65, f"R = {R:.3f} Ω",       color='yellow', fontsize=11, transform=axI.transAxes)
    axI.text(0.38, 0.65, f"T = {T:.1f} K",        color='tomato', fontsize=11, transform=axI.transAxes)
    axI.text(0.56, 0.65, f"V = {V:.3f} V",        color='white',  fontsize=11, transform=axI.transAxes)
    axI.text(0.02, 0.15,
             f"Electrolyte: {B.electrolyte}   "
             f"Anode: {B.anode}   "
             f"Particles: {B.n}   "
             f"Run {len(B.exited)+1}   "
             f"({max(0, 5.0 - B.t):.1f}s left)",
             color='#aaaaaa', fontsize=9, transform=axI.transAxes)
 
# FuncAnimation calls update() every 30ms
# cache_frame_data=False is needed because our frames are not static
anim = FuncAnimation(fig, update, interval=40, cache_frame_data=False)
 
plt.show()













               

        
           
    


                    

          
        
                            




       
          