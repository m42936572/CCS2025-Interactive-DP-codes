import numpy as np
import math


class PrefixSum_fixed_length:
    def __init__(self, T, epsilon,delta,size):
        self.T = T
        self.epsilon = epsilon
        self.delta=delta
        self.max_level = int(np.ceil(np.log2(T))) + 1  # Max tree depth needed
        self.noisy_partial_sums = {}  # {level: {index: noisy_value}}
        self.size=size

        # Precompute all Laplace noise for power-of-two intervals
        self._precompute_noise()
        self._private_prefix_sum_noise()


    def _precompute_noise(self):
        for level in range(self.max_level):
            self.noisy_partial_sums[level] = {}
            nodes_at_level = (self.T) // (1 << level)
            self.noisy_partial_sums[level] = np.random.laplace(0, np.log2(self.T)/self.epsilon,[self.size,nodes_at_level])

    def _private_prefix_sum_noise(self):

        self.value = []
        # Decompose t into a sum of power-of-two intervals
        for t in range(1,self.T+1):
          total=np.zeros(self.size)
          remaining = t
          level = 0
          while remaining > 0:
              # Check if the current bit is set in the binary representation
              if remaining & 1:
                  node_idx = (remaining - 1) >> level
                  total += self.noisy_partial_sums[level][:,node_idx]
              remaining >>= 1
              level += 1
          self.value.append(total)
        return total
    




class PrefixSum:
  def __init__(self,epsilon,delta,size):
    self.epsilon=epsilon
    self.delta=delta
    self.values=[]
    self.power_series=[]
    self.cnt=0
    self.power_cnt=0
    self.true_value=[]
    self.prefixsum=PrefixSum_fixed_length(1,self.epsilon,self.delta,1)
    self.noise_sum=[]
    self.size=size

  def query(self,num):
    self.cnt+=1
    self.true_value.append(num)
    if math.log2(self.cnt).is_integer():
      self.power_cnt=self.cnt
      self.power_series.append(np.random.laplace(0,1/self.epsilon,self.size)+np.sum(self.true_value,axis=0))
      self.prefixsum=PrefixSum_fixed_length(self.cnt,self.epsilon,self.delta,self.size)
      self.noise_sum.append(self.power_series[-1])
      return self.power_series[-1]
    else:
      self.noise_sum.append(self.power_series[-1]+self.prefixsum.value[self.cnt-self.power_cnt]+np.sum(self.true_value[self.power_cnt-1:self.cnt-1],axis=0))
      return self.power_series[-1]+self.prefixsum.value[self.cnt-self.power_cnt]+np.sum(self.true_value[self.power_cnt-1:self.cnt-1],axis=0)