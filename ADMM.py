import numpy as np
from geometric_noise import geometric_noise
from prefixsum import PrefixSum
from sklearn.base import BaseEstimator



class MyADMM(BaseEstimator):
    def __init__(self,
    beta=0.14418361364211363,
    eta=1,
    epsilon=0.1,
    delta=0.1,
    dimension=10,
    start=0) -> None:
        self.beta=beta
        self.eta=eta
        self.scores=[]
        self.avg_scores=[]
        self.dimension=dimension
        self.prefixsum=PrefixSum(epsilon,delta,dimension)
        self.start=start
        self.U=2*np.log2(1/delta)/epsilon


    def estimate_parameter(self, c1, c2):
        #starting point

        self.x=self.start
        self.y = self.x*0
        self.lamb = np.zeros(self.dimension)
        self.scores = []


    def train(self,c1,c2,datastream,N):


        #print(self.x)



        #update for y
        for i in range(len(self.y)):
            if self.lamb[i]-self.beta*self.x[i]+c1>0 and self.lamb[i]-self.beta*self.x[i]-c1<0:
                self.y[i]=0
            elif self.lamb[i]-self.beta*self.x[i]+c1>0 and self.lamb[i]-self.beta*self.x[i]-c1>0:
                self.y[i]=-(self.lamb[i]-self.beta*self.x[i]-c1)/(2*c2+self.beta)
            elif self.lamb[i]-self.beta*self.x[i]+c1<0 and self.lamb[i]-self.beta*self.x[i]-c1<0:
                self.y[i]=-(self.lamb[i]-self.beta*self.x[i]+c1)/(2*c2+self.beta)
            else:
                if (self.lamb[i]-self.beta*self.x[i]+c1)**2>(self.lamb[i]-self.beta*self.x[i]-c1)**2:
                    self.y[i]=-(self.lamb[i]-self.beta*self.x[i]+c1)/(2*c2+self.beta)
                else:
                    self.y[i]=-(self.lamb[i]-self.beta*self.x[i]-c1)/(2*c2+self.beta)

        #update for lambda
        self.lamb=self.lamb-self.beta*(self.x-self.y)


        #update for x
        gradient=0

        for data in (datastream):
          sample_a=(data[0:-1])
          sample_b=data[-1]
          nabla_f=2*sample_a*(((np.dot(np.transpose(sample_a),self.x)-sample_b)))
          gradient=self.prefixsum.query(nabla_f-self.beta*self.y-self.lamb)


        self.x=self.start-gradient*self.eta/(1+self.eta*self.beta)


        return self.x