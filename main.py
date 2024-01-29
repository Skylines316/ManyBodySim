def move():
    pass

def sample():
    pass

Neq = 1000
Ndec = 10
Nsamples = 1000

for i in range(Neq+Ndec*Nsamples):
    move()
    if (i - Neq) % Ndec == 0:
        sample()

