import sys
import numpy as np

import matplotlib.pyplot as plt

class universe():
  def __init__(self, epsilon, sigma, T):
    self.e = epsilon
    self.si = sigma
    self.beta = 1/T

  def potential(self, r1, rall, L):
    U = 0
    for i in rall:
      x = np.min([(r1[0]-i[0])**2, (r1[0]-i[0]+L)**2 , (r1[0]-i[0]-L)**2])
      y = np.min([(r1[1]-i[1])**2, (r1[1]-i[1]+L)**2 , (r1[1]-i[1]-L)**2])
      d = np.sqrt(x + y)
      U += 4*self.e*(np.power(self.si/d,12)-np.power(self.si/d,6))
    return U
  


class container(universe):
  def __init__(self,  epsilon, sigma, T, side, particles):
    super().__init__(epsilon, sigma, T)
    self.N = particles
    self.L = side

  def init_position(self):
    lim = np.power(sys.maxsize,-1/48)
    sites = int(self.L//lim) + 1
    lit_l = float(L/sites)
    if sites**2 <= self.N:
      print("Warning")
    initial = np.concatenate((np.zeros(sites**2-self.N),np.ones(self.N)))
    np.random.shuffle(initial)
    r = np.zeros((self.N,2))
    k=0
    for i,j in np.ndenumerate(initial):
      if j ==1:
        x,y = (i[0]%sites)*lit_l + 0.5*lit_l, (i[0]//sites)*lit_l+0.5*lit_l
        r[k] = np.array([x,y])
        k += 1
    return r

  def energy(self, r):
    E_arr = np.zeros(self.N)
    for k in range(self.N):
      rest = np.concatenate((r[:k], r[k+1:]))
      # print(rest)
      E_arr[k] = self.potential(r[k], rest, self.L)
    return E_arr

  def move(self, r, E_arr):
    delta = 0.1
    i = np.random.choice(range(r.shape[0]))
    
    r_f=np.zeros(2)
    r_f[0] = r[i][0] + (np.random.random()-0.5)*delta
    r_f[1] = r[i][1] + (np.random.random()-0.5)*delta
    
    rest = np.concatenate((r[:i], r[i+1:]))
    E_f = self.potential(r_f, rest,self.L)
    delta_E = E_f - E_arr[i]
    
    if np.random.random() <= np.exp(-self.beta*(delta_E)):
      r[i] = r_f
      E_arr[i] = E_f
    return r, E_arr

epsilon = 1
sigma = 1

T = 2

L = 3
N = 15

try1 = container(epsilon, sigma, T, L, N)

r = try1.init_position()
E_arr = try1.energy(r)


def sample(r, E_arr):
  E = np.sum(E_arr)
  return E

Neq = 2000
Ndec = 10
Nsamples = 10000

samples = []

for i in range(Neq+Ndec*Nsamples):
  r, E_arr = try1.move(r, E_arr)
  if (i - Neq) % Ndec == 0 and i > Neq:
    E = sample(r, E_arr)
    samples.append(E)
    
plt.plot(range(len(samples)),samples)
plt.show()



