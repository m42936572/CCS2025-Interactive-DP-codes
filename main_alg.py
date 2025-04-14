import numpy as np
from geometric_noise import geometric_noise
from prefixsum import PrefixSum
from ADMM import MyADMM
from randbin import randbin
from numpy import linalg as LA

def lasso_obj(A, b, x, c1,c2):
    return (LA.norm(A @ x - b))**2/A.shape[0] + c1 * LA.norm(x, ord=1)+c2*(LA.norm(x,ord=2))**2




class main_alg:
  def __init__(self,epsilon,delta,U,dimension,datastream,c1,c2,eta) -> None:
    self.randbin=randbin(epsilon,delta,U)
    self.U=U

    self.scores_users=[]
    self.scores_batches=[]

    self.fix_batch_scores_users=[]
    self.fix_batch_scores_batches=[]

    self.start=3*np.ones(dimension)

    self.admm=MyADMM(eta=eta,epsilon=epsilon,delta=delta,dimension=dimension,start=self.start)
    self.admm.estimate_parameter(c1,c2)

    self.A=datastream[:,0:-1]
    self.b=datastream[:,-1]

    self.c1=c1
    self.c2=c2

    self.datastream=datastream

    self.scores_users.append(lasso_obj(self.A, self.b, self.start, c1,c2))
    self.scores_batches.append(lasso_obj(self.A, self.b, self.start, c1,c2))


    self.fix_batch_scores_users.append(lasso_obj(self.A, self.b, self.start, c1,c2))
    self.fix_batch_scores_batches.append(lasso_obj(self.A, self.b, self.start, c1,c2))



  def simulate(self,iter=1):
    for j in range(iter):
      #self.randbin=randbin(epsilon,delta,U)
      for i in range(np.shape(self.datastream)[0]):
        out,flag=self.randbin.query(self.datastream[i,:])
        if flag==0:
          self.scores_users.append(self.scores_users[-1])
        else:
          sol=self.admm.train(self.c1,self.c2,out,np.shape(self.datastream)[0])
          self.scores_users.append(lasso_obj(self.A, self.b, sol, self.c1,self.c2))
          self.scores_batches.append(lasso_obj(self.A, self.b, sol, self.c1,self.c2))

      #print('finish epoch '+str(j))

  def simulate_fix_size(self,iter=1):
    for j in range(iter):
      cnt=0
      #self.randbin=randbin(epsilon,delta,U)
      while cnt<(np.shape(self.datastream)[0]):
        out=[]
        for k in range(int(self.U*1.5)):
          if cnt>=np.shape(self.datastream)[0]:
            break
          if k <(int(self.U*1.5))-1:
            out.append(self.datastream[cnt,:])
            self.fix_batch_scores_users.append(self.fix_batch_scores_users[-1])
          else:
            sol=self.admm.train(self.c1,self.c2,out,np.shape(self.datastream)[0])
            self.fix_batch_scores_users.append(lasso_obj(self.A, self.b, sol, self.c1,self.c2))
            self.fix_batch_scores_batches.append(lasso_obj(self.A, self.b, sol, self.c1,self.c2))

          cnt+=1
