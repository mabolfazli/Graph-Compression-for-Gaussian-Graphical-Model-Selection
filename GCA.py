import sys
from CTWbMain import CTWb
import math as m
import numpy as np
from numpy import array as arr
from numba import jit
from scipy import stats
from scipy import special
import networkx as nx
from timeit import default_timer as timer

class LinkList:
    def __init__(self,total=0):
        self.total = total  # Total number of neighbors
        self.nlinks = []    # List of groups: [size of group, number of links]
        self.p = []         # P(1) that should be used to encode each group in class-1
        self.pc = []        # P(1|#common neighbors) for common neighbors in class-1
        self.p4 = []        # P(1|4-node motifs) that works based on 4-cycle, double-triangle and 4-clique in class-1
        self._p = []         # P(1) that should be used to encode each group in class-2
        self._pc = []        # P(1|#common neighbors) for common neighbors in class-2
        self._p4 = []        # P(1|4-node motifs) that works based on 4-cycle, double-triangle and 4-clique in class-2



class CS12s:
    """CS12 algorithm, slow version"""

    def __init__(self):
        # Notice: by mistake, the meaning of B1 and B2 has been swapped compared to CS12
        self.B1 = []
        self.B2n = []
        self.B2 = []

        # Alternative code distribution
        self.link_list = [] # list of links to be coded, class LinkList
        self.degree_dist = None # pmf
        self.icdf = None # 1-CDF of degree distribution
        self.n = None

    def parse(self, n, al, p=0.5, _p=0.5, pt=None, _pt=None, pc=None, _pc=None, p4=None, _p4=None, update=False):
        """Parses a graph into B1 and B2
        n is graph size, al is an adjacency list as give by NetworkX.
        
        statistics for class-1 coder, If update is true, statistics are updated after coding each partition:
        p is p(1), 
        pt is triangle probability
        pc is an array of size n for number of common neighbors
        p4 is an array of size 4 for coding with 4-node motifs
        
        statistics for class-2 coder, If update is true, statistics are updated after coding each node:
        _p is p(1), 
        _pt is triangle probability
        _pc is an array of size n for number of common neighbors
        _p4 is an array of size 4 for coding with 4-node motifs
        """

        self.n = n
        self.B1 = []
        self.B2n = []
        Pk = [set(range(0,n))]
        v = 0
        prev_list = set()
        if update:
            """class-1 statistics"""
            ab = arr([0,0]) # keep track of KTe for p
            abt = arr([0,0]) # keep track of KTe for pt
            abc = np.zeros([n,2]) # keep track of KTe for common neighbors coder
            ab4 = np.zeros([4,2]) # keep track of KTe for 4-node motifs
            p = 0.5
            pt = 0.5
            pc = 0.5*np.ones(n)
            p4 = 0.5*np.ones(4)
            """"class-2 statistics"""
            _ab = arr([0,0]) # keep track of KTe for p
            _abt = arr([0,0]) # keep track of KTe for pt
            _abc = np.zeros([n,2]) # keep track of KTe for common neighbors coder
            _ab4 = np.zeros([4,2]) # keep track of KTe for 4-node motifs
            _p = 0.5
            _pt = 0.5
            _pc = 0.5*np.ones(n)
            _p4 = 0.5*np.ones(4)
            
        for k in range(0,n-1):
            #if not k%50: print('node:{0}, time:{1}'.format(k, timer()))
            #print()
            #print('selected node: ', v)
            #print('previous nodes: ', prev_list)
            if len(Pk[0]) == 1:
                del Pk[0]
            else:
                Pk[0] -= {v}

            # Calculate number of neighbors for each element in Pk, with that you can also update KT estimator
            neigbors = set(al[v])
            self.link_list.append(LinkList(len(neigbors)))
            for s in Pk:
                nn = len(neigbors & s)
                #print('nn: {0}, partition {1} '.format(nn, s) )
                """coding with B1 and B2"""
                if len(s) <= 1:
                    self.B1.append(nn)
                else:
                    self.B2n.append([len(s), nn])
                self.link_list[-1].nlinks.append([len(s),nn])
                
                """coding with p and pt"""
                if pt is None or len(set(al[min(s)]) & prev_list & neigbors) == 0:
                    self.link_list[-1].p.append(p)
                    self.link_list[-1]._p.append(_p)
                    #print('coding with non-tri prob: {0}'.format(p) )
                    if update:
                        p, ab = KTe(ab,nn,len(s))
                        _ab[0]+=len(s)-nn
                        _ab[1]+=nn
                else:
                    self.link_list[-1].p.append(pt)
                    self.link_list[-1]._p.append(_pt)
                    #print('coding with tri prob: {0}'.format(pt) )
                    if update:
                        pt, abt = KTe(abt,nn,len(s))
                        _abt[0]+=len(s)-nn
                        _abt[1]+=nn
                
                """coding with common neighbors"""
                if pc is not None:
                    idc = len(set(al[min(s)]) & prev_list & neigbors)
                    self.link_list[-1].pc.append(pc[idc])
                    self.link_list[-1]._pc.append(_pc[idc])
                    #print('coding with com neigh prob: {0}, number: {1}'.format(pc[idc], idc) )
                    if update:
                        pc[idc], abc[idc] = KTe(abc[idc],nn,len(s))
                        _abc[idc][0]+=len(s)-nn
                        _abc[idc][1]+=nn
                
                """coding with 4-node motifs"""
                if p4 is not None:
                    n_com  = list(set(al[min(s)]) & prev_list & neigbors)  # number of common nodes
                    pre_ns  = set(al[min(s)]) & prev_list # for cycle
                    pre_nv = neigbors & prev_list # for cycle
                    status = True # it guarantee that we encode with one of them
                    #4-clique
                    if len(n_com)>=2:
                        i=0
                        while i<len(n_com) and status:
                            for j in range(1,len(n_com)):
                                if len(set(al[n_com[i]]) & prev_list & set(al[n_com[j]])) >=1:
                                    self.link_list[-1].p4.append(p4[3])
                                    self.link_list[-1]._p4.append(_p4[3])
                                    #print('coding with 4-clique: {0}'.format(p4[3]) )
                                    status = False
                                    if update:
                                        p4[3], ab4[3] = KTe(ab4[3],nn,len(s))
                                        _ab4[3][0]+=len(s)-nn
                                        _ab4[3][1]+=nn
                                    break
                            i+=1
                        if status: # this is one realization of double-triangle
                            self.link_list[-1].p4.append(p4[2])
                            self.link_list[-1]._p4.append(_p4[2])
                            status = False
                            #print('coding with doub-tri: {0}'.format(p4[2]) )
                            if update:
                                p4[2], ab4[2] = KTe(ab4[2],nn,len(s))
                                _ab4[2][0]+=len(s)-nn
                                _ab4[2][1]+=nn            
                        
                    #double-triangle
                    if status and len(n_com)==1 and len(prev_list)>=2:
                        temp = set(al[n_com[0]]) & prev_list
                        if len(temp & set(al[min(s)]))>=1 or len(temp & neigbors)>=1:
                            self.link_list[-1].p4.append(p4[2])
                            self.link_list[-1]._p4.append(_p4[2])
                            status = False
                            #print('coding with doub-tri: {0}'.format(p4[2]) )
                            if update:
                                p4[2], ab4[2] = KTe(ab4[2],nn,len(s))
                                _ab4[2][0]+=len(s)-nn
                                _ab4[2][1]+=nn
                    
                    #cycle
                    if status and len(pre_ns) and len(pre_nv) and len(prev_list)>=2:
                        for ns in pre_ns:
                            if len(set(al[ns]) & pre_nv)>=1:
                                self.link_list[-1].p4.append(p4[1])
                                self.link_list[-1]._p4.append(_p4[1])
                                status = False
                                #print('coding with 4-cycle: {0}'.format(p4[1]) )
                                if update:
                                    p4[1], ab4[1] = KTe(ab4[1],nn,len(s))
                                    _ab4[1][0]+=len(s)-nn
                                    _ab4[1][1]+=nn
                                break
                        
                    if status: #simple edge
                        self.link_list[-1].p4.append(p4[0])
                        self.link_list[-1]._p4.append(_p4[0])
                        #print('coding with not-cycle: {0}'.format(p4[0]) )
                        if update:
                            p4[0], ab4[0] = KTe(ab4[0],nn,len(s))
                            _ab4[0][0]+=len(s)-nn
                            _ab4[0][1]+=nn

            # Update statistics for class-2
            if update:
                _p, _ab = KTe(_ab,0,0)
                _pt, _abt = KTe(_abt,0,0)
                for idc in range(len(_abc)):
                    _pc[idc], _abc[idc] = KTe(_abc[idc],0,0)
                for idc in range(len(_ab4)):
                    _p4[idc], _ab4[idc] = KTe(_ab4[idc],0,0)
                    
            # Update Pk
            Pkn = []
            for s in Pk:
                Pkn.append(neigbors & s)
                Pkn.append(s - neigbors)
            Pk = Pkn
            for j in reversed(range(len(Pk))): # remove empty sets
                if len(Pk[j]) == 0:
                    del Pk[j]

            prev_list |= {v}
            v = min(Pk[0])            

        # Convert B2n to binary
        self.B2 = []
        for nx in self.B2n:
            for j in reversed(range(m.ceil(m.log2(nx[0]+1)))):
                self.B2.append((nx[1] >> j ) & 0x1)

        # Calculate empirical degree distribution, if not already set

        if self.degree_dist is None:
            max_links = 0
            for node in range(n): # find maximum degrees
                max_links = max(max_links, len(al[node]))
            self.degree_dist = [0] * (max_links+1)
            for node in range(n):
                self.degree_dist[len(al[node])] += 1 / n

        if update:
            # final statistics of class1 and class2 are the same (i.e., p =_p, pc = _pc). So, we return just one of them
            return (p, pt), pc, p4


    def set_degree_dist(self,d):
        """Sets the degree distribution"""
        self.degree_dist = d


    def codelength(self):
        """Calculates codelength for B1 and B2 using CTW"""
        c = CTWb(2)
        for b in self.B1:
            c.update(b)
        tc = c.codelength()

        c = CTWb(2)
        for b in self.B2:
            c.update(b)
        tc += c.codelength()
        return tc

    def codelength_bin(self, ph=None):
        """Calculates codelength for B1 using CTW and for B2 using Binomial learned
        from B1. If ph is given, that is used"""
        if ph is None:
            c = CTWb(2)
            for b in self.B1:
                c.update(b)
            tc = c.codelength()

            ph = (sum(self.B1)+1/2)/(len(self.B1)+1) #KT estimate
        else:
            tc = -sum(self.B1) * np.log2(ph) - (len(self.B1)-sum(self.B1)) * np.log2(1-ph)

        for b in self.B2n:
            tc += -(stats.binom.logpmf(b[1], b[0], ph)* m.log2(m.exp(1)))

        return tc


    def codelength_degree(self, known_degree = False):
        """Calculates codelength using empirical degree distribution"""

        if self.icdf is None:
            self.icdf = 1 - np.cumsum(self.degree_dist)
            self.icdf[0] = 1
            self.icdf[1:] = self.icdf[0:-1]

        tc = 0
        for links in self.link_list:
            slinks = sum([x[1] for x in links.nlinks])
            p = self.degree_dist[links.total]/self.icdf[links.total - slinks]
            cl = -m.log2(p)
            if known_degree:
                cl = 0

            n = 0 # number of rows below diagonal
            for link in links.nlinks:
                n += link[0]
                cl += -_log_binom(link[0],link[1],True)
            cl -= -_log_binom(n,slinks,False)

            tc += cl

        return tc

    
    def codelength_seq(self):
        """Calculates codelength with conditional distribution"""
        tc = 0
        for links in self.link_list:
            for b,p in zip(links.nlinks,links.p):
                # need to use logpmf otherwise precision error happens
                tc += -(stats.binom.logpmf(b[1], b[0], p)* m.log2(m.exp(1)))

        return tc

    def codelength_class1(self, ctype='tri'):
        """Calculates codelength for coders in class1
        'tri': tirnagle coder
        'com': number of common neighbors coder
        '4nod': 4-node motifs coder
        """
        tc = 0
        if ctype=='tri':
            for links in self.link_list:
                for b,p in zip(links.nlinks,links.p):
                    # need to use logpmf otherwise precision error happens
                    tc += -(stats.binom.logpmf(b[1], b[0], p)* m.log2(m.exp(1)))
        
        if ctype=='com':
            for links in self.link_list:
                for b,pc in zip(links.nlinks,links.pc):
                    # need to use logpmf otherwise precision error happens
                    tc += -(stats.binom.logpmf(b[1], b[0], pc)* m.log2(m.exp(1)))
                    
        if ctype=='4nod':
            for links in self.link_list:
                for b,p4 in zip(links.nlinks,links.p4):
                    # need to use logpmf otherwise precision error happens
                    tc += -(stats.binom.logpmf(b[1], b[0], p4)* m.log2(m.exp(1)))
        return tc
    
    def codelength_class2(self, ctype='tri', known_degree = False):
        """Calculates codelength for coders in class2 using PBD
        'tri': tirnagle coder
        'com': number of common neighbors coder
        '4nod': 4-node motifs coder
        """
        #compute the degree distribution
        if self.icdf is None:
            self.icdf = 1 - np.cumsum(self.degree_dist)
            self.icdf[0] = 1
            self.icdf[1:] = self.icdf[0:-1]

        tc = 0
        count=False            
        if ctype=='tri':
            for links in self.link_list:
                slinks = sum([x[1] for x in links.nlinks])
                p = self.degree_dist[links.total]/self.icdf[links.total - slinks]
                cl = -m.log2(p) # sending the number of new connections
                if known_degree:
                    cl = 0

                # Now calculate probability of specific configuration
                n = 0 # number of rows below diagonal
                pp = [] # probability realization for poisson binomial distribution
                
                if not count: #for first node is same as degree distribution
                    count=True
                    for link in links.nlinks:
                        n += link[0]
                        cl += -_log_binom(link[0],link[1],True)
                    cl -= -_log_binom(n,slinks,False)
                    tc += cl
                    
                else: #for later nodes
                    for b,p in zip(links.nlinks, links._p):
                        n += b[0]
                        for i in range(b[0]):
                            pp.append(p)
                        cl += -(stats.binom.logpmf(b[1], b[0], p)* m.log2(m.exp(1)))
                    cl -= -m.log2(pbd_calc(pp, slinks))
                    tc += cl
        
        if ctype=='com':
            for links in self.link_list:
                slinks = sum([x[1] for x in links.nlinks])
                p = self.degree_dist[links.total]/self.icdf[links.total - slinks]
                cl = -m.log2(p)
                if known_degree:
                    cl = 0

                n = 0
                pp = [] 
                if not count: #for first node is same as degree distribution
                    count=True
                    for link in links.nlinks:
                        n += link[0]
                        cl += -_log_binom(link[0],link[1],True)
                    cl -= -_log_binom(n,slinks,False)
                    tc += cl
                    
                else: #for later nodes
                    for b,p in zip(links.nlinks, links._pc):
                        n += b[0]
                        for i in range(b[0]):
                            pp.append(p)
                        cl += -(stats.binom.logpmf(b[1], b[0], p)* m.log2(m.exp(1)))
                    cl -= -m.log2(pbd_calc(pp, slinks))
                    tc += cl
            
        if ctype=='4nod':
            for links in self.link_list:
                slinks = sum([x[1] for x in links.nlinks])
                p = self.degree_dist[links.total]/self.icdf[links.total - slinks]
                cl = -m.log2(p)
                if known_degree:
                    cl = 0

                n = 0 
                pp = [] 
                if not count: #for first node is same as degree distribution
                    count=True
                    for link in links.nlinks:
                        n += link[0]
                        cl += -_log_binom(link[0],link[1],True)
                    cl -= -_log_binom(n,slinks,False)
                    tc += cl
                    
                else: #for later nodes
                    for b,p in zip(links.nlinks, links._p4):
                        n += b[0]
                        for i in range(b[0]):
                            pp.append(p)
                        cl += -(stats.binom.logpmf(b[1], b[0], p)* m.log2(m.exp(1)))
                    cl -= -m.log2(pbd_calc(pp, slinks))
                    tc += cl
        return tc
    
    def opt_codelength(self): #needs to be updated
        """Calculates optimum universal codelength by taking the minimum of various methods"""
        bincl = self.codelength_bin()
        degreecl = self.codelength_degree() + self.n - 0.5 * np.log2(self.n)
        tricl = self.codelength_seq()
        return min((bincl, degreecl, tricl)) + 2

def codelength(G):
    """Calculates the codelength of the graph G"""
    pg = CS12s()
    pg.parse(G.number_of_nodes(), G, update=True)
    return pg.opt_codelength()


def subgraph(G, nodes):
    """Gives a subgraph from the nodes, with nodes renumbered 0-len(nodes)-1"""

    # Probably not efficient for large graphs
    A = nx.convert_matrix.to_numpy_array(G, dtype=bool)
    A = A[nodes, :][:, nodes]
    return nx.convert_matrix.from_numpy_array(A)


def subgraph_codelength(G, nodes):
    """Calculates the codelength of a graph split into two subgraphs"""
    nodes = np.asarray(nodes).flat
    complement = list(set(range(G.number_of_nodes())) - set(nodes))
    G1 = subgraph(G, nodes)
    G2 = subgraph(G, complement)

    # Find connection graph
    Gc = np.zeros((len(nodes), len(complement)), dtype=bool)
    for i in range(len(nodes)):
        neighbors = set(G[nodes[i]])
        ll = [complement[j] in neighbors for j in range(len(complement))]
        Gc[i, ll] = True

    c = CTWb(4)
    c.updates(Gc.flat)

    return codelength(G1), codelength(G2), c.codelength()


#numba.jit translates a subset of Python and NumPy into fast machine code using LLVM, via the llvmlite Python package.
#@jit(nopython=True)
def _log_binom(n,k,upper):
    """Calculates bound on logarithm of binomial.
    If upper = true, an upper bound is calculated, otherwise lower bound"""
    if n < 100 or k < 10 or n-k < 10:
        return m.log2(special.binom(n,k))

    h = -(k/n) * m.log2(k/n) - (1 - k/n) * m.log2(1 - k/n)

    if upper:
        return n * h + 1/2 * m.log2(n/(m.pi * k * (n-k)))
    else:
        return n * h + 1/2 * m.log2(n/(8 * k * (n-k)))

def ERsuper(n,p):
    """ER graph plus super node connected to all nodes"""
    G = nx.gnp_random_graph(n,p)
    G.add_star(G.nodes())
    return G

def KTe(ab,n1,nt):
    """KT estimator update: n1 is number of ones, nt total number of bits"""
    ab[0] += nt - n1
    ab[1] += n1
    p = (ab[1] + 0.5)/(sum(ab) + 1)
    return p, ab

def pbd_calc(p, s):
    """computing the denominator probability in coding with PBD"""
    mat = np.ones([s+1, len(p)+1])
    for j in range(1,len(p)+1):
        for k in range(s+1):
            if k == 0:
                mat[k][j] = (1-p[j-1])*mat[k][j-1]
            elif k == j:
                mat[k][j] = p[j-1]*mat[k-1][j-1]
            else:
                mat[k][j] = (1-p[j-1])*mat[k][j-1] + p[j-1]*mat[k-1][j-1]
    return mat[s][len(p)]
