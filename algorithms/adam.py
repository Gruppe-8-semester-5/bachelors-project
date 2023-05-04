import numpy as np


class Adam:
    def __init__(self, step_size, b1=0.94, b2=0.9879, e=10e-8):
        if b1 > 1 or b1 < 0:
            raise ValueError
        if b2 > 1 or b2 < 0:
            raise ValueError
        self.step_size = step_size
        self.b1 = b1
        self.b2 = b2
        self.m_t = 0
        self.v_t = 0
        self.e = e
        self.t = 0


    def step(self, w, derivation):
        # https://arxiv.org/pdf/1412.6980.pdf
        self.t += 1

        grad_current = derivation(w)
        self.m_t = self.b1 * self.m_t + (1-self.b1) * grad_current
        self.v_t = self.b2 * self.v_t + (1-self.b2) * ( np.square(grad_current) )
        self.mh_t = self.m_t/(1-(self.b1 ** self.t))
        self.vh_t = self.v_t/(1-(self.b2 ** self.t))

        res = w - (self.step_size * (self.mh_t/(safe_sqrt(self.vh_t)+self.e)))
        return res

def safe_sqrt(w):
    if w.dtype != 'object':
        return np.sqrt(w)
    res = []
    # Probably quite inefficient, but works just fine.
    for i in range(w.shape[0]):
        res.append(np.sqrt(w[i]))
    return np.array(res, dtype=object)