# Lithium-Battery-Toy-Model**
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



