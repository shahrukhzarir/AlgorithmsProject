import numpy as np
import pylab as py
from algorithmChecker import *
import csv


class Particle:
    def __init__(self, dim=10):
        pass
        self.__dim = dim

class PSO:
    def __init__(self, func, bounds, initPos=None):

        # number of particles in swarm
        self.nPart = 100

        # Control Parameters
        self.epsError = 1
        self.maxGen = 3000*10
        self.w = 0.2
        self.phiP = 0.2
        self.phiG = 0.1
        self.default = -1

        # Function to be minimised
        self.problem = func

        # Set up boundary values
        self.minBound = np.array(bounds[0])
        self.maxBound = np.array(bounds[1])

        #Setup Dimensions
        self.dim = len(bounds[0])


        # Initial positions
        if initPos!=None:
            self.initPos = np.array(initPos).reshape((self.default,self.dim))
        else:
            self.initPos = initPos


    def __initPart(self):
        """Initiate particles.
        """

        # Create particles
        self.Particles = []
        for i in range(self.nPart):
            self.Particles.append( Particle(self.dim) )

        # Initiate pos and fit for particles
        for part in self.Particles:

            # Initial position
            if self.initPos == None:
                part.pos = np.random.random(self.dim)*self.maxBound - self.minBound
            else:
                part.pos = self.initPos[0,:]
                self.initPos = np.delete(self.initPos, 0,0)

                # If nothing left on initial pos
                if len(self.initPos) == 0:
                    self.initPos = None

            # Initial velocity
            part.vel = np.random.random(self.dim)*(self.maxBound - self.minBound)
            part.vel *= [-1, 1][np.random.random()>0.5]

            # Initial fitness
            part.fitness = self.problem(part.pos)
            part.bestFit = part.fitness
            part.bestPos = part.pos

        # Global best fitness
        self.globBestFit = self.Particles[0].fitness
        self.globBestPos = self.Particles[0].pos
        for part in self.Particles:
            if part.fitness < self.globBestFit:
                self.globBestFit = part.fitness
                self.globBestPos = part.pos

    def update(self):

        for part in self.Particles:

            # Gen param
            rP, rG = np.random.random(2)

            w, phiP, phiG = self.w, self.phiP, self.phiG

            # Update velocity
            v, pos = part.vel, part.pos
            part.vel = self.w*v + phiP*rP*(part.bestPos-pos) + phiG*rG*(self.globBestPos-pos)

            # New position
            part.pos += part.vel

            # If pos outside bounds
            if np.any(part.pos<self.minBound):
                NFC = part.pos<self.minBound
                part.pos[NFC] = self.minBound[NFC]
            if np.any(part.pos>self.maxBound):
                NFC = part.pos>self.maxBound
                part.pos[NFC] = self.maxBound[NFC]

            # New fitness
            part.fitness = self.problem(part.pos)

        # Global and local best fitness
        for part in self.Particles:

            # Comparing to local best
            if part.fitness < part.bestFit:
                part.bestFit = part.fitness

            # Comparing to global best
            if part.fitness < self.globBestFit:
                self.globBestFit = part.fitness
                self.globBestPos = part.pos

    def optimize(self):
        """ Optimisation function.
            Before it is run, initial values should be set.
        """

        # Initiate particles
        self.__initPart()
        self.listOfPos = []

        NFC = 0
        while(NFC < self.maxGen):
            #print "Run: " + str(NFC) + " Best: " + str(self.globBestFit)

            # Perform search
            self.update()

            # Acceptably close to solution
            #if self.globBestFit < self.epsError:
                #return self.globBestPos, self.globBestFit

            # next gen
            NFC += 1
            self.listOfPos.append(self.globBestFit)
        # Search finished
        return self.globBestPos, self.globBestFit, self.listOfPos

#################################

if __name__ == "__main__":
    N = 100
    outputFile = open('output.csv', 'w')
    outputWriter = csv.writer(outputFile)
    outputWriter.writerow(['Function 1'])
    outputWriter.writerow(['Run','Best Fit', 'Best Solution'])

    for i in range(25):
        t = np.linspace(-100, 100, N)
        minProb = lambda t: f1(t)
        numParam = 4
        bounds = ([0]*numParam, [10]*numParam)

        pso = PSO(minProb, bounds)
        g = pso.optimize()
        outputWriter.writerow([[i+1],g[0], g[1]])
        #print 'bestFit: ', g[0]
        #print 'bestPos: ', g[1]

    ############################
    # Visual results representation
    """
    py.figure()
    py.plot(g[2])
    py.xlabel("NFC")
    py.ylabel("Best Fit Performance")
    py.title("PSO Performance Vs NFC")
    py.show()
    """
