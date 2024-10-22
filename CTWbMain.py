import math as m
from abc import ABC, abstractmethod

class Coder(ABC):
    @abstractmethod
    def update(self,b):
        """Updates codelength when b is added to the sequence"""
        pass

    def updates(self,bits):
        """updates with multiple bits"""
        for b in bits:
            self.update(b)

    @abstractmethod
    def codelength(self):
        """returns codelength for the string seen so far"""
        pass

    @abstractmethod
    def restart(self,context=None):
        """starts coding a new sequence"""
        pass

def KTe(P,ab,b):
    """Calculates KT Estimator

    KTeKTe(P,ab,b): P is prior probability, ab is count of 0 and 1, b is current bit"""
    return P+m.log2((ab[b]+1/2)/(sum(ab)+1))

ln2i=1/m.log(2) # constant to avoid recalculation

def logsp(x,y):
    """calculates log2(2^x+2^y) so as to avoid overflow"""
    a = min(x,y)
    b = max(x,y)
    return b+m.log1p(2**(a-b))*ln2i

class ContextTree:
    """Handles contexts"""
    def __init__(self,D):
        self.D = D
        self.context = []

    def set_context(self,context):
        """Sets context"""
        if len(context) > self.D - 1:
            self.context = list(context[-self.D + 1:-1])
        else:
            self.context = list(context[:])

    def update_context(self,b):
        if len(self.context) < self.D-1:
            self.context.append(b)
        else:
            self.context = self.context[1:] + [b]

class ContextNode:
    """Basic tree structure without values"""
    def __init__(self):
        self.child = [None,None]    # Children


class CTWNode(ContextNode):
    def __init__(self):
        ContextNode.__init__(self)
        self.count = [0,0]          # Count of zeros and ones
        self.Pe = 0.0               # Pe (log)
        self.Pw = 0.0               # Pw (log)
        self.tail = 0               # if there is a tail. See Willem's 98 paper

class CTWb(ContextTree, Coder):
    def __init__(self,D):
        ContextTree.__init__(self,D)
        self.root = CTWNode()

    def codelength(self):
        """Returns total codelength"""
        if self.D == 0:
            return self.root.count[1] + self.root.count[2]
        else:
            return -self.root.Pw

    def update(self,b):
        """Updates the CTW tree with the bit b"""
        self.__update(b,self.root,len(self.context)-1)
        self.update_context(b)

    def __update(self,b,node,ci): #recursive update function
        node.Pe = KTe(node.Pe,node.count,b) # Update Pe
        node.count[b] += 1 # Update bit count
        if ci < 0: # Empty context. End of recursion
            node.Pw = node.Pe
            if len(self.context) < self.D - 1:
                node.tail = 1
        else:
            # First update outer leaves
            contextb=self.context[ci]
            if node.child[contextb] == None: #context not seen yet
                node.child = [CTWNode(),CTWNode()]
            self.__update(b,node.child[contextb],ci-1)

            # Now update Pw
            node.Pw = logsp(node.Pe,node.child[0].Pw
                            + node.child[1].Pw - node.tail)-1

    def recalc_codelength(self,rep):
        """Recalculates the codelength with rep repetitions (rep=1: no repetition)"""

        if self.D == 0:
            return rep * sum(self.root.count)

        add = 1 / 2 * m.log2(rep)

        # implementation is in terms of Python dictionary
        # the key is the address of a node
        # This implementation cannot be done directly in C
        Pe = {}
        Pw = {}

        def recalcPe(node): # recursive function for calculating Pe
            if node.Pe:
                Pe[id(node)] = rep * node.Pe - add \
                    + (rep - 1)/2 * m.log2(sum(node.count))
                if node.child[0] != None:
                    recalcPe(node.child[0])
                    recalcPe(node.child[1])
            else:
                Pe[id(node)] = 0

        def recalcPW(node): # recursive function for calculation Pw
            # first calculate outer leaves
            if node.child[0] != None:
                recalcPW(node.child[0])
                recalcPW(node.child[1])

            # Update Pw
            idn = id(node)
            if (node.child[0] == None) or node.Pe > node.Pw:
                # The second condition ensures that the model does not
                # change with repetition: if the iid model is better for the
                # original data, then the non-iid model will be discarded
                # in repeat model.

                Pw[idn] = Pe[idn]
            else:
                Pw[idn] = logsp(Pe[idn],Pw[id(node.child[0])]
                                + Pw[id(node.child[1])] - node.tail) - 1

        recalcPe(self.root)
        recalcPW(self.root)
        return -Pw[id(self.root)]

    def restart(self,context=None):
        self.__init__(self.D)
        if context is not None:
            self.set_context(context)




