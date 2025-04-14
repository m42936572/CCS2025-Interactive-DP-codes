import numpy as np



def geometric_noise(U,epsilon,size):
  g1 = np.random.geometric(1 - np.exp(-epsilon), size=size) - 1  # G1 ~ Geom(1-p)
  g2 = np.random.geometric(1 - np.exp(-epsilon), size=size) - 1  # G2 ~ Geom(1-p)
  noise=g1-g2+U*1.5
  noise[np.where(noise<U)]=U
  noise[np.where(noise>2*U)]=2*U
  if size==1:
    noise=noise[0]
  return noise.astype(int)