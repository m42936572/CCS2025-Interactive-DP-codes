import numpy as np
from geometric_noise import geometric_noise
from prefixsum import PrefixSum


class randbin:
  def __init__(self,epsilon,delta,U):
    self.epsilon=epsilon
    self.delta=delta
    self.buffer=[]
    self.bins=[]
    self.E=np.log2(1/self.delta)*30/self.epsilon
    self.true_value=[]
    self.cnt=0
    self.U=U
    self.rho=[geometric_noise(U,epsilon,1)]
    self.prefixsum=PrefixSum(epsilon,delta,1)
    self.S=[self.prefixsum.query(self.rho[0])]


  def query(self,element):
    self.cnt+=1
    self.true_value.append(element)
    self.buffer.append(element)

    if self.cnt>=self.E+self.U+self.S[-1]:
      bin=self.buffer[0:self.rho[-1]-1]
      del self.buffer[0:self.rho[-1]-1]
      self.bins.append(bin)
      self.rho.append(geometric_noise(self.U,self.epsilon,1))
      self.S.append(self.prefixsum.query(self.rho[-1]))
      return bin,1
    return [],0