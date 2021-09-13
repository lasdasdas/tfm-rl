from collections import deque

import heapq
import numpy as np
import random
import time

class InsRem:
    def __init__(self, data):
        self.data = data
    def __lt__(self, o):
        return np.mean(self.data[0]) < np.mean(o.data[0])
    def get(self):
        return self.data

class PER_priority:
    def __init__(self):
        deq_len = 400000
        # self.memory = deque(maxlen=deq_len)
        # self.prios = deque(maxlen=deq_len)
        self.memory = []

        self.def_val = -1000

    def remember(self, state, action, reward, next_state, done, info):
        t = (state, action, reward, next_state, done, info)
        heapq.heappush(self.memory, (self.def_val, InsRem(t)))

    def remember_weighted(self, h, ins):
        heapq.heappush(self.memory, (h, InsRem(ins)))

    def sample(self, batch_size):
        
        l = [heapq.heappop(self.memory) for _ in range(batch_size)]
        return [x[1].get() for x in l]

    def __len__(self):
        return len(self.memory)

class PER_slow:
    def __init__(self, deq_len=400000):
        self.memory = deque(maxlen=deq_len)
        self.prios = deque(maxlen=deq_len)

        self.def_val = 1000

    # def remember(self, state, action, reward, next_state, done, info):
        # self.memory.append([( state, action, reward, next_state, done, info), self.def_val])

    def remember(self, t):
        self.memory.append([t, self.def_val])


    def sample(self, batch_size):
        val_vec = np.array( [x[1] for x in  self.memory])
        indices = np.array( [x for x in range(len(self.memory))])

        val_vec = val_vec / (np.sum(val_vec))
        
        choices = np.random.choice(indices, batch_size, p=val_vec)
        # choices = np.random.choice(indices, batch_size)
        batch = [self.memory[x][0] for x in choices]


        return batch, choices

    def update_indices(self, indices, new_values):
        assert(len(indices) == len(new_values))
        assert(not any(x < 0 for x in new_values))
        for i, nv in zip(indices, new_values):
            self.memory[i][1] = nv

    def __len__(self):
        return len(self.memory)

class PER_u:
    def __init__(self, size):
        self.memory = deque(maxlen= 2**size)

    def remember(self, obj):
        self.memory.append(obj)

    def sample(self, batch_size):
        batch = random.choices(self.memory, k=batch_size)
        return batch, None

    def update_indices(self, indices, new_values):
        pass

    def __len__(self):
        return len(self.memory)
    def capacity(self):
        return len(self.memory)


class STpoor:
    def __init__(self, depth, smart_unsample=0, file_store=False):
        self.depth = depth

        self.leafs = 2 ** self.depth
        self.nodes = 2 ** (self.depth +1) -1

        self.noleafs = self.nodes - self.leafs
        
        self.probs = np.zeros((self.nodes))
        self.probs_inv = 1000*np.ones((self.nodes))
        self.smart_unsample = smart_unsample
        

        
        ni_def = 1000
        for i in range(0, self.depth +1):
            inv_depth = self.depth - i
            self.probs_inv[2**i -1 : 2**(i + 1)] = ni_def * (2** (inv_depth ) )

        self.cbuff = [None for _ in range(self.leafs)]
        self.samples = np.zeros((self.leafs)).astype(np.int)
        self.ci = 0
        self.rolled_buffer = False
    
    def capacity(self):
        return self.leafs

    def is_leaf(self, idx):
        return idx >= self.noleafs
    def __len__(self):
            if self.rolled_buffer:
                return self.leafs
            else:
                return self.ci

    def is_root(self, idx):
        return idx == 0

    def to_pidx(self, oidx):
        return self.noleafs + oidx 

    def to_oidx(self, pidx):
        return pidx - self.noleafs  

    def insert(self, obj, wp):
        # Replace objects
        ni=0
        if self.smart_unsample != 0:
            ni = self.single_sample_inverse()
        else:
            ni = self.ci

        self.cbuff[ni] = obj
        self.samples[ni] = 0

        # Update probs
        
        pidx = self.to_pidx(ni)

        self.update_probs(pidx, wp)
        
        # Roll buffer
        self.ci += 1
        if self.ci == self.leafs:
            self.ci = 0
            self.rolled_buffer = True

    def update_indices(self, indices, nw):

        for i in range(len(indices)):
            pidx = self.to_pidx(indices[i])
            self.update_probs(pidx, nw[i])

    def assert_integrity(self, ):
        for idx in range(self.nodes):
            eps = 0.1
            nl = 2*idx + 1 
            nr = 2*idx + 2
            if nl >= self.nodes or nr >= self.nodes:
                continue
            assert(self.probs[idx] >= 0)
            assert(np.abs(self.probs[nl] + self.probs[nr]  - self.probs[idx]) < eps)

            assert(self.probs_inv[idx] >= 0)
            assert(np.abs(self.probs_inv[nl] + self.probs_inv[nr]  - self.probs_inv[idx]) < eps)

        


    def update_probs(self, pidx, nprob):
        i = pidx
        l = []
        while 1: 
            l.append(i)
            if not self.is_root(i):
                i = (i-1) // 2
            else:
                break
        larr = np.array(l)
        self.probs[larr] -= self.probs[pidx] - nprob

        if self.smart_unsample == 1:
            self.probs_inv[larr] -= self.probs_inv[pidx] -   1/ (nprob*nprob + 0.001)
        elif self.smart_unsample == 2:
            self.probs_inv[larr] -= self.probs_inv[pidx] -   1/ (np.sqrt(nprob+ 0.001))
        elif self.smart_unsample == 3:
            self.probs_inv[larr] -= self.probs_inv[pidx] -   1/ (nprob + 0.001)
        elif self.smart_unsample == 0:
            self.probs_inv[larr] -= self.probs_inv[pidx] -   1/ (nprob*nprob + 0.001)
        else:
            print("not possible")
            assert(0)


    def print(self):
        aux = 0
        aux2 = 0
        for i in range(self.nodes):
            print(self.probs[i], end =" | ") 
            if i == aux2:
                aux += 1
                aux2 += 2 ** aux
                print("")
        print([x is not None for x in self.cbuff])
        print("")

        aux = 0
        aux2 = 0
        for i in range(self.nodes):
            print(self.probs_inv[i], end =" | ") 
            if i == aux2:
                aux += 1
                aux2 += 2 ** aux
                print("")
        print([x is not None for x in self.cbuff])
        print("")

    
    def sample(self, size):
        l = [self.single_sample() for x in range(size)]
        return [self.cbuff[x] for x in l], np.array(l)



    def single_sample_inverse(self, idx=0):
        p = random.uniform(0, self.probs_inv[idx])
        while 1:
            if self.is_leaf(idx):
                oidx = self.to_oidx(idx)
                # assert(self.cbuff[oidx] is not None)
                return  oidx

            nl = 2*idx + 1 
            nr = 2*idx + 2

            p = random.uniform(0, self.probs_inv[idx])
            if p > self.probs_inv[nl]:
                idx = nr
            else:
                idx = nl

    def single_sample(self, idx=0):
        p = random.uniform(0, self.probs[idx])
        while 1:
            if self.is_leaf(idx):
                oidx = self.to_oidx(idx)
                self.samples[oidx] += 1
                return oidx

            nl = 2*idx + 1 
            nr = 2*idx + 2

            p = random.uniform(0, self.probs[idx])
            # p = p %self.probs[idx]
            if p > self.probs[nl]:
                idx = nr
            else:
                idx = nl


class PER_st:
    def __init__(self, size=19, smart_unsample=0, file_store=False):
        self.st = STpoor(size, smart_unsample)
        self.def_mult = 1
        self.eps= 0.01
        self.insertions = 0

    def remember(self, obj):
        self.insertions += 1
        def_val = (self.st.probs[0] )/ (len(self.st) + 1) * self.def_mult + self.eps
        def_val  = 1
        self.st.insert(obj, def_val)
        

    def update_indices(self, indices, new_values):
        new_values += self.eps
        # PI = new_values / self.st.probs[0]
        # wi = 1/self.st

        self.st.update_indices(indices, new_values )

    def sample(self, batch_size):
        return self.st.sample(batch_size)
    
    def debug(self):
        print(batch)
        print([self.st.probs[self.st.to_pidx(i)] for i in range(len(self.st.cbuff))] )

    def __len__(self):
        return min(len(self.st), self.insertions)
    def capacity(self):
        return self.st.capacity()




if __name__ == "__main__":
    # Build st

    st = STpoor(4)

    st.print()

    print("--------------------------------------------------")
    st.insert(1, 0.1)
    st.print()

    print("--------------------------------------------------")
    st.insert(2, 0.1)
    st.print()

    st.insert(3, 0.2)
    st.insert(4, 0.3)
    st.insert(5, 0.3)
    st.insert(6, 0.1)
    st.insert(7, 0.9)

    st.update_indices([0], [80] )
    print("--------------------------------------------------")
    st.print()

    for i in range(len(st)):
        st.insert(i % 8, 0.1)

    for x in range(500):
        st.single_sample()

    print(len(st))
    # Build deque

    us= PER_slow(len(st))

    for i in range(len(st)):
        us.remember(i % 8)


    print(len(st))
    t = time.time()
    values, idx = st.sample(64)
    print("time is st: ", time.time() - t)

    t = time.time()
    values, idx = us.sample(64)
    print("time is: us", time.time() - t)


















# p, v = np.histogram(values, bins=np.arange(11), density=True)
# for i in range(len(p)):
    # print(p[i], v[i])
