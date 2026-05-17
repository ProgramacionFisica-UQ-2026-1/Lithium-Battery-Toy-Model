# Lithium-Battery-Toy-Model
## Sebastián Rincón Martínez
This is a Lithium Battery Toy model, assuming mechanical bouncing on electrolyte, and electrons output zones. Gaussian distribution for electrons output is made, additional to temperature analysis on full battery.

Lithium ion batteries are of great importance nowadays, since they're used in daily used devices, as cellphones, or electrical cars. Actually, usage of lithium has increased substantially in last years, as shown in the image.
<img width="1200" height="1200" alt="image" src="https://github.com/user-attachments/assets/1a3e4e0e-61b6-49bb-a918-2b0453c7f852" />
Increasing in use of electrical cars has a consequence: increase on lithium demand, as shown in the following plot.
<img width="1200" height="1000" alt="image" src="https://github.com/user-attachments/assets/a480b843-0144-416e-a33b-41d5d6a8ef37" />
Problem is lithium is scarce in the nature. Then, it's important to optimize batteries, for reducing as much as possible the lithium waste. That's why this toy model is designed as a starting point for a program able to simulate battery electrical behavior and efficiency when using different materials combinations in anode, cathode and electrolyte, correspondent to the main parts of a functional battery. In this case, only electrical forces will be considered, ignoring chemical reactions or disipative forces.

Additional to the low disponibility problem, lithium has also presented heating problems, since it's highly volatile. In the code, a simulation of temperature due to electrical flux is presented, as an opportunity for determining specifical zones in which major danger may be presented, and for taking future preventive actions.

**CODE**
Following section will explain the code separated by the most important steps.
## Abstract class Particle
This abstract class defines all the mechanical and electronical properties of particles in the system, and asks other classes to define their visual properties (color). This is the main class, that gives information about position, vectorial velocity, mass and charge.
```
from abc import ABC, abstractmethod
import numpy as np

class Particle(ABC):
    def __init__(self, position, velocity, mass, charge):
        self.position = np.asarray(position, float) 
        self.velocity = np.asarray(velocity, float)
        self.mass = float(mass)
        self.charge = float(charge)

    @abstractmethod
    def color(self) -> str:
        pass
```
## Classes Electron, Lithium ion and Anode
This classes define the properties of three particles present in batteries: electrons (with charge -1 and mass 1, for the simulation), lithium ions, and anodes, which are positively charged, and are massive respecting electrons and ions.
```
class Electron(Particle):
    def __init__(self, position): 
        super().__init__(position, np.zeros(3), mass=1.0, charge=-1.0) 
        self.orbiting = False 
        self.orbit_center = None 
        self.orbit_angle = 0.0
    def color(self):
        return 'cyan' if self.orbiting else 'blue'

class LithiumIon(Particle):
    def __init__(self, position):
        super().__init__(position, np.zeros(3), mass = 1265.0, charge = 1.0)
    def color(self): 
        return 'red'

class Anode(Particle):
    def __init__(self, position, charge):
        super().__init__(position, np.zeros(3), mass = 1e9, charge=charge) 
    def color(self):
        return 'limegreen'
```
## Constants
This section establishes the necessary constraints, possible electrolyte and anodes structures for user selections, and numerical values for maintaining simulation stable.
```
BIAS_FORCE = {"EC": 9.0, "LiPF6": 22.0, "DMC": 14.0}
Anode_Q = {r"$CoO_4$":4.0, "Li_metal": 6.0, "Silicon": 3.0}

K = 1.0         # Normalized Coulomb constant
Eps = 0.003     # Softening factor for avoiding division by zero in Coulomb's law
Dt         = 0.006      # time per step
Box        = 1.0        # battery size
Max_acc    = 500.0      # maximum acceleration for avoiding crashing
Damp       = 0.9985     # Percentage of every velocity for avoiding crashing
Barrier_x  = 0.5        # Electrons cannot pass this limit
Orbit_r    = 0.04       # Effective radius for anode to catch electrons
Orbit_w    = 12.0       # angular frequency for electrons
V_Bias = 1.5 # This will work as an external applied voltage
R_0= 0.5 # Arbitrary reference resistance
T_0 = 300.0 # Arbitrary temperature in Kelvin, approximately environmental temperature
Alpha = 0.004 # Thermal coefficient of resistance
```
## Class LiIonBattery Initialization
This acts as the setup manager for the battery. When a new simulation starts, this block takes the user's choices (number of particles, type of electrolyte, and anode material), prepares empty lists to track temperature and performance history, prepares a self.instance for electrons outputs and resets the simulation clock to zero.
```
class LiIonBattery:
   def __init__(self, n, electrolyte, anode): 
       self.n = n
       self.electrolyte = electrolyte
       self.anode = anode
       self.bias_force = BIAS_FORCE[electrolyte]
       self.anode_charge = Anode_Q[anode]
       self.t = 0.0 # Simulation time
       self.run_exit = 0.0 # Electrons crossing in this run
       self.exited = [] # Crossed by every run
       
       self.temp_left = [] # This is used in the future for temperature measurement
       self.temp_anode = [] #[t, T].

       self.initial_particles()
```
## Particle initial positions
This function populates our virtual battery cell. It scatters the requested number of electrons and lithium ions randomly on the left side of the container, while placing the heavy anode particles in a fixed vertical wall on the right side.
```
def initial_particles(self):
       self.electrons = []
       self.ions = []
       self.anodes = []
       for i in range(self.n):
           self.electrons.append(Electron(np.random.rand(3)*[0.4, 1, 1]))
           self.ions.append(LithiumIon(np.random.rand(3)*[0.4,1,1]))
           self.anodes.append(Anode([0.95, np.random.rand(), np.random.rand()], self.anode_charge)) 
       self.all = self.electrons + self.ions + self.anodes
```
## Force calculation
It looks at every single particle, calculates the distances between them, and uses Coulomb's law to determine how much they push or pull on each other based on their electrical charges. It also applies a constant forward or backward push (the bias force) to simulate the chemical gradient of the battery's electrolyte.
```
def forces(self):
       N = len(self.all) 
       P = np.zeros((N,3)) 
       Q = np.zeros(N)
       for i in range(N):
           P[i] = self.all[i].position
           Q[i] = self.all[i].charge 
       diff = np.zeros((N,N,3))
       for i in range (N):
           for j in range (N):
               diff[i][j] = P[i]-P[j]
       r2 = np.sum(diff**2, axis=2) + Eps**2 
       inv3 = np.zeros((N,N))
       for i in range(N):
           for j in range(N):
               if i != j:
                   inv3[i][j]=r2[i][j]**(-1.5) 
        
       F_Coulomb = np.zeros((N,3)) 
       for i in range(N):
           for j in range(N):
               F_Coulomb[i]+= K*Q[i]*Q[j]*diff[i][j]*inv3[i][j]
        
       F_bias = np.zeros((N,3))
       for i in range(N):
           F_bias[i][0] = self.bias_force*np.sign(Q[i])
       return F_Coulomb+F_bias
```
## Boundaries
This is an important section for simulation understanding. Battery is 1x1x1 dimentioned. When any particle touches a wall in y and z direction, it's assumed a mechanical bouncing, by multiplying velocity by -1. There's an output zone of electrons in x=0, in the section 0.4<y<0.6 and 0.4<z<0.6 There, a 'teletransportation' is assumed for electrons, that arrive to the right side, simulating a wire transportation of charge, that enforces ions mobility for recharge process. In the middle of x direction, there's a simulation of wall, that only allows ions to pass, but avoids electrons flux; the wall simulates electrolyte function.
```
def boundary(self, p):
       for ax in (1,2): 
           if p.position[ax]<0:
               p.position[ax] = 0.0
               p.velocity[ax] *= -1
           if p.position[ax] > Box:
               p.position[ax] = Box
               p.velocity[ax]*=-1    
       if isinstance (p, Electron): 
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
       else:
            if p.position[0]<0:
                p.position[0] = 0.0
                p.velocity[0] *=-1
            if p.position[0] > Box:
                p.position[0]=Box
                p.velocity[0]*=-1
```
## Electron orbits
When an electron transported by wire arrives to anode side, they tend to bond with positively charged particles. It is simulated by assuming an effective radius of trapping by anode, in which electron keeps orbiting (but still supports battery with charge) and turns cyan color.
```
def check_orbit(self,e): 
       for a in self.anodes:
           dist = np.sqrt((e.position[0]-a.position[0])**2+(e.position[1]-a.position[1])**2+(e.position[2]-a.position[2])**2)
           if dist < Orbit_r and not e.orbiting: 
               e.orbiting= True
               e.orbit_center = a.position.copy()
               e.orbit_angle = np.random.rand()*2*np.pi
```
## Steps and time integration
This section calculates forces and accelerations after Dt time, and reshapes the simulation to new positions. Then, this section is the hearth of dynamical processes in the model.
```
def step(self): 
      F = self.forces()
      for i, p in enumerate(self.all):
          if isinstance(p, Anode):
              continue
          if isinstance(p, Electron) and p.orbiting:
              p.orbit_angle += Orbit_w * Dt 
              p.position[0]=p.orbit_center[0]
              p.position[1] = p.orbit_center[1]+Orbit_r*np.cos(p.orbit_angle)
              p.position[2] = p.orbit_center[2]+ Orbit_r*np.sin(p.orbit_angle)
              continue
          acc = F[i]/p.mass 
          for i in range(3):
              if abs(acc[i]) > Max_acc:
                  acc[i] = Max_acc*np.sign(acc[i])
          p.position += p.velocity*Dt + 0.5*acc*Dt**2

          self.boundary(p) 

          if isinstance(p, Electron):
              self.check_orbit(p) 

          p.velocity = (p.velocity+acc*Dt)*Damp 
      self.t += Dt  
      self.record_temperature()
```
## Electrical variables
In this section, current, voltage and resistance are calculated. Current and resistance are used for temperature of battery determination. Used formulas are fully based on basic electromagnetism.
```
def electrical(self):
       probe = np.array([0.5, 0.5, 0.5]) 
       total_crossings =  sum(self.exited)+self.run_exit 
       I = total_crossings / (self.t +Dt) 
       V_Coulomb = 0.0
       for p in self.all:
           r = np.sqrt((p.position[0]-probe[0])**2 +
                       (p.position[1]-probe[1])**2+
                       (p.position[2]-probe[2])**2) + Eps
           V_Coulomb += K*p.charge / r
       V_total = V_Bias+V_Coulomb
       R = V_total / (I+1e-9) 
       T = T_0 + (I**2*R)/(Alpha*R_0) 
       return I, R, T, V_total
```
## Localized heat
In terms of fractions of electrons present on left side (cathode) or anode, localized temperature is calculated, in order to compare an approximated temperature behavior in different sections of battery.
```
def record_temperature(self):
       if self.temp_left and self.t - self.temp_left[-1][0] < 0.05: 
           return 
       I, R, T, V = self.electrical()
       n_left = 0
       n_anode = 0
       for e in self.electrons:
           if e.position[0] < 0.4:
               n_left += 1
           if e.position[0] > 0.7:
               n_anode += 1
       w_left = n_left / len(self.electrons) 
       w_anode = n_anode / len(self.electrons)

       self.temp_left.append((self.t, T_0 + (T-T_0)*w_left)) 
       self.temp_anode.append((self.t, T_0+(T-T_0)*w_anode))
```
## Cycle and end of cycles
The code restarts with randomly located particles every 5 seconds. This section saves the gotten data about electrons output in every run, and restarts with different conditions.
```
def end_run(self):
       self.exited.append(self.run_exit)
       self.run_exit = 0
       self.t = 0
       self.initial_particles()
```
## Statistical Gaussian fit using from scipy, function norm
Once the battery has completed at least three full discharge cycles, this function analyzes the historical data to calculate the mathematical average and consistency spread of electron crossings. It uses these values to map out a smooth bell-curve line on the interface.
```
def  gaussian_fit(self):
       if len(self.exited) < 3:
           return None
       data = np.array(self.exited)
       mu, sigma =  norm.fit(data) 

       x_min = max(0, mu -4*sigma)
       x_max = mu +4*sigma
       x = np.linspace(x_min, x_max, 200)
       pdf = norm.pdf(x, mu, sigma) 

       return x, pdf, mu, sigma
```
## Figure and plot
Now, visual setups are determined. Visually, simulation is divided in four sections: dynamical 3D simulation, Gaussian fit, temperatures comparison, and simulation information.
```
fig = plt.figure(figsize=(15,8))
fig.patch.set_facecolor('#000000') 
gs = gridspec.GridSpec (2,3, figure=fig,
                        hspace = 0.45, wspace=0.35, 
                        left = 0.05, right=0.97,
                        top = 0.93, bottom=0.22)

ax3d = fig.add_subplot(gs[0,0], projection='3d') 
axG = fig.add_subplot(gs[0,1]) 
axT = fig.add_subplot(gs[0,2]) 
axI = fig.add_subplot(gs[1,:]) 

for ax in (axG, axT, axI):
    ax.set_facecolor('#0D0D1A')
    ax.tick_params(colors = 'white')
    for spine in ax.spines.values(): 
        spine.set_color('#333333')

ax3d.set_facecolor('#0D0D1A')
fig.suptitle('Lithium ions toy model battery', color='White', fontsize=13, fontweight='bold')
```
## Buttons
For making simulation user-friendly, buttons are applied for determining which electrolyte, and anode will be used, and how many of each particle will be in the simulation.
```
def make_radio(localization, labels, title):
    ax = plt.axes(localization, facecolor='#1A1A2E')
    b = RadioButtons(ax, labels, activecolor='cyan')
    ax.set_title(title, color='white', fontsize=8)
    for lb in b.labels:
        lb.set_color('white')
    return b

b_n = make_radio([0.04, 0.03, 0.10, 0.13], ('5', '10', '15'), 'Number of particles')
b_el = make_radio([0.17, 0.03, 0.10, 0.13], ('EC', 'LiPF6', 'DMC'), 'Electrolyte')
b_an = make_radio([0.33, 0.03, 0.10, 0.13], (r'$CoO_4$', 'Li_metal', 'Silicon'), 'Anode')

ax_pause = plt.axes([0.76,0.05,0.09,0.06])
ax_pause.set_facecolor('#1A1A2E')
btn_pause = Button(ax_pause, 'Pause/Resume', color='#1A1A2E', hovercolor='#333333')
btn_pause.label.set_color('White')

ax_save = plt.axes([0.87, 0.05,0.09, 0.06])
ax_save.set_facecolor('#1A1A2E')
btn_save=Button(ax_save, 'Save figures', color='#1A1A2E', hovercolor='#333333')
btn_save.label.set_color('white')
```
## Rebuild after button pressed
Another important fact about the simulation is that everything restarts if a button (different to pause and save) is pressed. All data is lost, so it's recomendeable to save figures when needed.
```
Battery = [LiIonBattery(10, 'EC', r'$CoO_4$')]
paused = [False]

def rebuild(*_):
    n = int(b_n.value_selected) 
    el = b_el.value_selected
    an = b_an.value_selected
    Battery[0] = LiIonBattery(n, el, an) 

b_n.on_clicked(rebuild) 
b_el.on_clicked(rebuild)
b_an.on_clicked(rebuild)
btn_pause.on_clicked(lambda _: paused.__setitem__(0, not paused[0]))
```
## Save of figures
Battery screeeshot and Gaussian plots are saved to the system when Save figures button is pressed.
```
def savefigures(event): 
    extent_3d = ax3d.get_window_extent().transformed(fig.dpi_scale_trans.inverted()) 
    fig.savefig('Battery.png', bbox_inches=extent_3d.expanded(1.2,1.2), dpi=150) 

    extent_g = axG.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig('BatteryGaussian.png', bbox_inches=extent_g.expanded(1.4,1.4),dpi=150)
    print('Saved: Battery.png and BatteryGaussian.png')
btn_save.on_clicked(savefigures)
```
## Window restart
When simulation time 5 seconds is ended, full interface restarts, but Gaussian plot data is saved.
```
def update(_):
    if paused[0]:
        return
    
    B = Battery[0]
    for i in range(4): 
        B.step()
    
    if B.t >= 5.0:
        B.end_run()
    I,R,T,V = B.electrical()

    # Graph 3D
    ax3d.cla() 
    ax3d.set_facecolor('#0D0D1A')
    ax3d.set_xlim(0, 1); ax3d.set_ylim(0, 1); ax3d.set_zlim(0, 1)
    ax3d.set_xlabel('x', color='white')
    ax3d.set_ylabel('y', color='white')
    ax3d.set_zlabel('z', color='white')
    ax3d.tick_params(colors='white')
    ax3d.set_title(f"t={B.t:.2f}s  run={len(B.exited)+1}  crossings={B.run_exit}",
                   color='white', fontsize=9)

    # Draw particles
    for p in B.all:
        ax3d.scatter(p.position[0], p.position[1], p.position[2], color = p.color(), s=22 if isinstance(p, Anode) else 10, depthshade=False)

    # Gaussian plot
    axG.cla()
    axG.set_facecolor('#0D0D1A')
    axG.set_title("Electrons per run", color='white', fontsize=9)
    axG.set_xlabel("Crossings per run", color='white')
    axG.set_ylabel("Density",           color='white')
    axG.tick_params(colors='white')   
    result = B.gaussian_fit()
    if result:
        x, pdf, mu, sigma = result
        axG.hist(B.exited, bins='auto', density=True, color='#00B1DD', alpha=0.45, label=f'Runs:{len(B.exited)}') 
        axG.plot(x, pdf, 'r-', lw=2, label = f'Average: {mu:.2f}. Spread: {sigma:.2f}')
        axG.legend(fontsize=8, labelcolor='white', facecolor='#1A1A2E')
    else:
        remaining = 3-len(B.exited)
        axG.text(0.5, 0.5, f'{remaining} more runs are required', ha='center', va='center', transform=axG.transAxes, color='#DDDDDD', fontsize=10)

    # Temperature plot
    axT.cla()
    axT.set_facecolor('#0D0D1A')
    axT.set_title("Temperature", color='white', fontsize=9)
    axT.set_xlabel("Time (s)", color='white')
    axT.set_ylabel("T (K)",    color='white')
    axT.tick_params(colors='white')   

    if B.temp_left:
        ts, Tl = zip(*B.temp_left)  
        axT.plot(ts, Tl, color='cyan', lw=1.5, label='Left side')
    if B.temp_anode:
        ts, Ta = zip(*B.temp_anode)
        axT.plot(ts, Ta, color='Orange', lw=1.5, label='Anode') 
    axT.legend(fontsize=8, labelcolor='white', facecolor='#1A1A2E')
    
    # Information bar
    axI.cla()
    axI.set_facecolor('#0d0d1a')
    axI.axis('off')   
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
```
## Animation
Finally, Matplotlib FuncAnimation function is used for dynamical activity of simulation, every 20ms
```
anim = FuncAnimation(fig, update, interval=20, cache_frame_data=False)
 
plt.show()
```

# Interpretation
Simulation is able to show temperature increases localized and in full battery, resistance, current and voltage.
<img width="430" height="267" alt="image" src="https://github.com/user-attachments/assets/592588f3-5b24-423c-998a-351fef344c66" />
<img width="878" height="39" alt="image" src="https://github.com/user-attachments/assets/8282858b-002d-42d5-80e1-a30ddc6c574f" />
<img width="445" height="269" alt="image" src="https://github.com/user-attachments/assets/9735180d-0ca3-4820-887b-0b116d68f665" />

Past images are examples of electric properties data, and temperature's change with the time, for a first run of 10 particles, with electrolyte EC and $CoO_4$ in anode. Is seen that both sides increased their temperatures, suggesting preventie actions shall effectively be taken, particularly in high-mobility zone of cathode, where temperature increased significatively, ignoring, of course, heat disipation. Similar results were seen for other simulations. Current constantly increased, with LiPF6 and $CoO_4$ combination having the higher result, showing why this combination is used currently. But in other situations, as the one of third image ($DMC-CoO_4$ combination), temperature oscillates more, and is hard to control. 

Examples of batteries 3D representation are shown in the following images
<img width="340" height="340" alt="Battery" src="https://github.com/user-attachments/assets/cc0de31c-ec9c-41a6-8aa9-62856f6f6d9f" />
<img width="340" height="340" alt="Battery" src="https://github.com/user-attachments/assets/308bdc63-9538-4727-a98d-6b14c5ac0ae1" />
As seen, simulation shows information about time, number of runs and crossed electrons in current run. Moreover, green particles show anodes, located at a same x position; blue particles represent electrons and red particles represent lithium ion. 

But most attractive results are Gaussian fixes, shown in following images
## $LiPF_6-CoO_4$ combination
<img width="678" height="397" alt="BatteryGaussian" src="https://github.com/user-attachments/assets/c1b3a06a-7a5a-4b4b-aef5-01cb6a38de50" />
## $EC-CoO_4$ combination
<img width="678" height="397" alt="BatteryGaussian" src="https://github.com/user-attachments/assets/fb98a09a-f8bd-4e2c-9e61-9b6e63ced4a3" />
## $DMC-CoO_4$ combination

## $LiPF_6-$Li-metal combination

## $EC_6-$ Li-metalcombination

## $DMC_6-$ Li-metal combination

## $LiPF_6-$ Silicon combination

## $EC-$ Silicon combination

## $DMC-$ Silicon combination

Batteries efficiency depends on many factors, such as generated current, and charge duration. Is seen that $LiPF_6-CoO_4$ combination is not the one with more electrons exits, but it was the one with greatest currents, and is one of the simulations with lower spread. This shows the combination is not only able to generate big currents, but also allows control over the battery, more than other combinations. Moreover, it's not neccesary to have too many electrons exiting at low times, cause it will probably generate more current than needed, and will discharge too fast. Then, simulation allows comparisons for different variables to consider for a battery: charge duration, temperature, current, resistance and stability or certain measurement of control over the battery results.

