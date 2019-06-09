# Python modul vo koj se implementirani algoritmite za informirano prebaruvanje

# ______________________________________________________________________________________________
# Improtiranje na dopolnitelno potrebni paketi za funkcioniranje na kodovite

import sys
import bisect

infinity = float('inf')  # sistemski definirana vrednost za beskonecnost


# ______________________________________________________________________________________________
# Definiranje na pomosni strukturi za cuvanje na listata na generirani, no neprovereni jazli

class Queue:
    """Queue is an abstract class/interface. There are three types:
       Stack(): A Last In First Out Queue.
       FIFOQueue(): A First In First Out Queue.
       PriorityQueue(order, f): Queue in sorted order (default min-first).
   Each type supports the following methods and functions:
       q.append(item)  -- add an item to the queue
       q.extend(items) -- equivalent to: for item in items: q.append(item)
       q.pop()         -- return the top item from the queue
       len(q)          -- number of items in q (also q.__len())
       item in q       -- does q contain item?
   Note that isinstance(Stack(), Queue) is false, because we implement stacks
   as lists.  If Python ever gets interfaces, Queue will be an interface."""

    def __init__(self):
        raise NotImplementedError

    def extend(self, items):
        for item in items:
            self.append(item)


def Stack():
    """A Last-In-First-Out Queue."""
    return []


class FIFOQueue(Queue):
    """A First-In-First-Out Queue."""

    def __init__(self):
        self.A = []
        self.start = 0

    def append(self, item):
        self.A.append(item)

    def __len__(self):
        return len(self.A) - self.start

    def extend(self, items):
        self.A.extend(items)

    def pop(self):
        e = self.A[self.start]
        self.start += 1
        if self.start > 5 and self.start > len(self.A) / 2:
            self.A = self.A[self.start:]
            self.start = 0
        return e

    def __contains__(self, item):
        return item in self.A[self.start:]


class PriorityQueue(Queue):
    """A queue in which the minimum (or maximum) element (as determined by f and
   order) is returned first. If order is min, the item with minimum f(x) is
   returned first; if order is max, then it is the item with maximum f(x).
   Also supports dict-like lookup. This structure will be most useful in informed searches"""

    def __init__(self, order=min, f=lambda x: x):
        self.A = []
        self.order = order
        self.f = f

    def append(self, item):
        bisect.insort(self.A, (self.f(item), item))

    def __len__(self):
        return len(self.A)

    def pop(self):
        if self.order == min:
            return self.A.pop(0)[1]
        else:
            return self.A.pop()[1]

    def __contains__(self, item):
        return any(item == pair[1] for pair in self.A)

    def __getitem__(self, key):
        for _, item in self.A:
            if item == key:
                return item

    def __delitem__(self, key):
        for i, (value, item) in enumerate(self.A):
            if item == key:
                self.A.pop(i)


# ______________________________________________________________________________________________
# Definiranje na klasa za strukturata na problemot koj ke go resavame so prebaruvanje
# Klasata Problem e apstraktna klasa od koja pravime nasleduvanje za definiranje na osnovnite karakteristiki
# na sekoj eden problem sto sakame da go resime


class Problem:
    """The abstract class for a formal problem.  You should subclass this and
   implement the method successor, and possibly __init__, goal_test, and
   path_cost. Then you will create instances of your subclass and solve them
   with the various search functions."""

    def __init__(self, initial, goal=None):
        """The constructor specifies the initial state, and possibly a goal
       state, if there is a unique goal.  Your subclass's constructor can add
       other arguments."""
        self.initial = initial
        self.goal = goal

    def successor(self, state):
        """Given a state, return a dictionary of {action : state} pairs reachable
       from this state. If there are many successors, consider an iterator
       that yields the successors one at a time, rather than building them
       all at once. Iterators will work fine within the framework. Yielding is not supported in Python 2.7"""
        raise NotImplementedError

    def actions(self, state):
        """Given a state, return a list of all actions possible from that state"""
        raise NotImplementedError

    def result(self, state, action):
        """Given a state and action, return the resulting state"""
        raise NotImplementedError

    def goal_test(self, state):
        """Return True if the state is a goal. The default method compares the
       state to self.goal, as specified in the constructor. Implement this
       method if checking against a single self.goal is not enough."""
        return state == self.goal

    def path_cost(self, c, state1, action, state2):
        """Return the cost of a solution path that arrives at state2 from
       state1 via action, assuming cost c to get up to state1. If the problem
       is such that the path doesn't matter, this function will only look at
       state2.  If the path does matter, it will consider c and maybe state1
       and action. The default method costs 1 for every step in the path."""
        return c + 1

    def value(self):
        """For optimization problems, each state has a value.  Hill-climbing
       and related algorithms try to maximize this value."""
        raise NotImplementedError


# ______________________________________________________________________________
# Definiranje na klasa za strukturata na jazel od prebaruvanje
# Klasata Node ne se nasleduva

class Node:
    """A node in a search tree. Contains a pointer to the parent (the node
   that this is a successor of) and to the actual state for this node. Note
   that if a state is arrived at by two paths, then there are two nodes with
   the same state.  Also includes the action that got us to this state, and
   the total path_cost (also known as g) to reach the node.  Other functions
   may add an f and h value; see best_first_graph_search and astar_search for
   an explanation of how the f and h values are handled. You will not need to
   subclass this class."""

    def __init__(self, state, parent=None, action=None, path_cost=0):
        "Create a search tree Node, derived from a parent by an action."
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.depth = 0
        if parent:
            self.depth = parent.depth + 1

    def __repr__(self):
        return "<Node %s>" % (self.state,)

    def __lt__(self, node):
        return self.state < node.state

    def expand(self, problem):
        "List the nodes reachable in one step from this node."
        return [self.child_node(problem, action)
                for action in problem.actions(self.state)]

    def child_node(self, problem, action):
        "Return a child node from this node"
        next = problem.result(self.state, action)
        return Node(next, self, action,
                    problem.path_cost(self.path_cost, self.state,
                                      action, next))

    def solution(self):
        "Return the sequence of actions to go from the root to this node."
        return [node.action for node in self.path()[1:]]

    def solve(self):
        "Return the sequence of states to go from the root to this node."
        return [node.state for node in self.path()[0:]]

    def path(self):
        "Return a list of nodes forming the path from the root to this node."
        x, result = self, []
        while x:
            result.append(x)
            x = x.parent
        return list(reversed(result))

    # We want for a queue of nodes in breadth_first_search or
    # astar_search to have no duplicated states, so we treat nodes
    # with the same state as equal. [Problem: this may not be what you
    # want in other contexts.]

    def __eq__(self, other):
        return isinstance(other, Node) and self.state == other.state

    def __hash__(self):
        return hash(self.state)


# ________________________________________________________________________________________________________
# Pomosna funkcija za informirani prebaruvanja
# So pomos na ovaa funkcija gi keshirame rezultatite od funkcijata na evaluacija

def memoize(fn, slot=None):
    """Memoize fn: make it remember the computed value for any argument list.
   If slot is specified, store result in that slot of first argument.
   If slot is false, store results in a dictionary."""
    if slot:
        def memoized_fn(obj, *args):
            if hasattr(obj, slot):
                return getattr(obj, slot)
            else:
                val = fn(obj, *args)
                setattr(obj, slot, val)
                return val
    else:
        def memoized_fn(*args):
            if not memoized_fn.cache.has_key(args):
                memoized_fn.cache[args] = fn(*args)
            return memoized_fn.cache[args]

        memoized_fn.cache = {}
    return memoized_fn


# ________________________________________________________________________________________________________
# Informirano prebaruvanje vo ramki na graf

def best_first_graph_search(problem, f):
    """Search the nodes with the lowest f scores first.
   You specify the function f(node) that you want to minimize; for example,
   if f is a heuristic estimate to the goal, then we have greedy best
   first search; if f is node.depth then we have breadth-first search.
   There is a subtlety: the line "f = memoize(f, 'f')" means that the f
   values will be cached on the nodes as they are computed. So after doing
   a best first search you can examine the f values of the path returned."""

    f = memoize(f, 'f')
    node = Node(problem.initial)
    if problem.goal_test(node.state):
        return node
    frontier = PriorityQueue(min, f)
    frontier.append(node)
    explored = set()
    while frontier:
        node = frontier.pop()
        if problem.goal_test(node.state):
            return node
        explored.add(node.state)
        for child in node.expand(problem):
            if child.state not in explored and child not in frontier:
                frontier.append(child)
            elif child in frontier:
                incumbent = frontier[child]
                if f(child) < f(incumbent):
                    del frontier[incumbent]
                    frontier.append(child)
    return None


def greedy_best_first_graph_search(problem, h=None):
    "Greedy best-first search is accomplished by specifying f(n) = h(n)"
    h = memoize(h or problem.h, 'h')
    return best_first_graph_search(problem, h)


def astar_search(problem, h=None):
    "A* search is best-first graph search with f(n) = g(n)+h(n)."
    h = memoize(h or problem.h, 'h')
    return best_first_graph_search(problem, lambda n: n.path_cost + h(n))


# ________________________________________________________________________________________________________
# Dopolnitelni prebaruvanja
# Recursive_best_first_search e implementiran
# Kako zadaca za studentite da gi implementiraat SMA* i IDA*

def recursive_best_first_search(problem, h=None):
    h = memoize(h or problem.h, 'h')

    def RBFS(problem, node, flimit):
        if problem.goal_test(node.state):
            return node, 0  # (The second value is immaterial)
        successors = node.expand(problem)
        if len(successors) == 0:
            return None, infinity
        for s in successors:
            s.f = max(s.path_cost + h(s), node.f)
        while True:
            # Order by lowest f value
            successors.sort(key=lambda x: x.f)
            best = successors[0]
            if best.f > flimit:
                return None, best.f
            if len(successors) > 1:
                alternative = successors[1].f
            else:
                alternative = infinity
            result, best.f = RBFS(problem, best, min(flimit, alternative))
            if result is not None:
                return result, best.f

    node = Node(problem.initial)
    node.f = h(node)
    result, bestf = RBFS(problem, node, infinity)
    return result


# _________________________________________________________________________________________________________
# PRIMER 2 : PROBLEM NA SLOZUVALKA
# OPIS: Dadena e slozuvalka 3x3 na koja ima polinja so brojki od 1 do 8 i edno prazno pole
# Praznoto pole e obelezano so *. Eden primer na slozuvalka e daden vo prodolzenie:
#  -------------
#  | * | 3 | 2 |
#  |---|---|---|
#  | 4 | 1 | 5 |
#  |---|---|---|
#  | 6 | 7 | 8 |
#  -------------
# Problemot e kako da se stigne do nekoja pocetna raspredelba na polinjata do nekoja posakuvana, na primer do:
#  -------------
#  | * | 1 | 2 |
#  |---|---|---|
#  | 3 | 4 | 5 |
#  |---|---|---|
#  | 6 | 7 | 8 |
#  -------------
# AKCII: Akciite ke gi gledame kako dvizenje na praznoto pole, pa mozni akcii se : gore, dole, levo i desno.
# Pri definiranjeto na akciite mora da se vnimava dali akciite voopsto mozat da se prevzemat vo dadenata slozuvalka
# STATE: Sostojbata ke ja definirame kako string koj ke ima 9 karakteri (po eden za sekoe brojce plus za *)
# pri sto stringot ke se popolnuva so izminuvanje na slozuvalkata od prviot kon tretiot red, od levo kon desno.
# Na primer sostojbata za pocetnata slozuvalka e: '*32415678'
# Sostojbata za finalnata slozuvalka e: '*12345678'
# ________________________________________________________________________________________________________
#

default_goal = '*12345678'  # predefinirana cel


# Ke definirame 3 klasi za problemot
# Prvata klasa nema implementirano nikakva hevristika

class P8(Problem):
    name = 'No Heuristic'

    def __init__(self, goal=default_goal, initial=None, N=20):
        self.goal = goal
        self.initial = initial

    def successor(self, state):
        return successor8(state)

    def actions(self, state):
        return self.successor(state).keys()

    def h(self, node):
        """Heuristic for 8 puzzle: returns 0"""
        return 0

    def result(self, state, action):
        possible = self.successor(state)
        return possible[action]


# Slednite klasi ke nasleduvaat od prethodnata bez hevristika so toa sto ovde ke ja definirame i hevristikata

class P8_h1(P8):
    """ Slozuvalka so hevristika
   HEVRISTIKA: Brojot na polinja koi ne se na vistinskoto mesto"""

    name = 'Out of Place Heuristic'

    def h(self, node):
        """Funkcija koja ja presmetuva hevristikata,
       t.e. razlikata pomegu nekoj tekoven jazel od prebaruvanjeto i celniot jazel"""
        matches = 0
        for (t1, t2) in zip(node.state, self.goal):
            # zip funkcijata od dve listi na vlez pravi edna lista od parovi (torki)
            # primer: zip(['a','b','c'],[1,2,3]) == [('a',1),('b',2),('c',3)]
            # zip('abc','123') == [('a','1'),('b','2'),('c','3')]
            if t1 != t2:
                matches = + 1
        return matches


class P8_h2(P8):
    """ Slozuvalka so hevristika
   HEVRISTIKA: Menheten rastojanie do celna sostojba"""

    name = 'Manhattan Distance Heuristic (MHD)'

    def h(self, node):
        """Funkcija koja ja presmetuva hevristikata,
       t.e. Menheten rastojanieto pomegu nekoj tekoven jazel od prebaruvanjeto i celniot jazel, pri sto
       Menheten rastojanieto megu jazlite e zbir od Menheten rastojanijata pomegu istite broevi vo dvata jazli"""
        sum = 0
        for c in '12345678':
            sum = + mhd(node.state.index(c), self.goal.index(c))  # pomosna funkcija definirana vo prodolzenie
        return sum


# Za da mozeme da go definirame rastojanieto treba da definirame koordinaten sistem
# Pozetokot na koordinatniot sistem e postaven vo gorniot lev agol na slozuvalkata
# Definirame recnik za koordinatite na sekoe pole od slozuvalkata
coordinates = {0: (0, 0), 1: (1, 0), 2: (2, 0),
               3: (0, 1), 4: (1, 1), 5: (2, 1),
               6: (0, 2), 7: (1, 2), 8: (2, 2)}


# Funkcija koja presmetuva Menheten rastojanie za slozuvalkata
# Na vlez dobiva dva celi broja koi odgovaraat na dve polinja na koi se naogaat broevite za koi treba da presmetame rastojanie
def mhd(n, m):
    x1, y1 = coordinates[n]
    x2, y2 = coordinates[m]
    return abs(x1 - x2) + abs(y1 - y2)


def successor8(S):
    """Pomosna funkcija koja generira recnik za sledbenicite na daden jazel"""

    blank = S.index('*')  # kade se naoga praznoto pole

    succs = {}

    # GORE: Ako praznoto pole ne e vo prviot red, togas vo sostojbata napravi swap
    # na praznoto pole so brojceto koe se naoga na poleto nad nego
    if blank > 2:
        swap = blank - 3
        succs['GORE'] = S[0:swap] + '*' + S[swap + 1:blank] + S[swap] + S[blank + 1:]

    # DOLE: Ako praznoto pole ne e vo posledniot red, togas vo sostojbata napravi swap
    # na praznoto pole so brojceto koe se naoga na poleto pod nego
    if blank < 6:
        swap = blank + 3
        succs['DOLE'] = S[0:blank] + S[swap] + S[blank + 1:swap] + '*' + S[swap + 1:]

    # LEVO: Ako praznoto pole ne e vo prvata kolona, togas vo sostojbata napravi swap
    # na praznoto pole so brojceto koe se naoga na poleto levo od nego
    if blank % 3 > 0:
        swap = blank - 1
        succs['LEVO'] = S[0:swap] + '*' + S[swap] + S[blank + 1:]

    # DESNO: Ako praznoto pole ne e vo poslednata kolona, togas vo sostojbata napravi swap
    # na praznoto pole so brojceto koe se naoga na poleto desno od nego
    if blank % 3 < 2:
        swap = blank + 1
        succs['DESNO'] = S[0:blank] + S[swap] + '*' + S[swap + 1:]

    return succs


# Vcituvanje na vleznite argumenti za test primerite

N = input()
K = input()


# Vasiot kod pisuvajte go pod ovoj komentar


class Misioneri(Problem):
    def __init__(self, initial, goal):
        self.initial = initial
        self.goal = goal

    def actions(self, state):
        return self.successor(state).keys()

    def result(self, state, action):
        possible = self.successor(state)
        return possible[action]

    def goal_test(self, state):
        return (state[3] == self.goal[3] and state[4] == self.goal[4])

    def isValid(self, state):
        if (state[1] > state[0] > 0) or (state[4] > state[3] > 0):
            return False
        if state[0] < 0 or state[1] < 0 or state[3] < 0 or state[4] < 0:
            return False
        return True

    def h(self, node):
        k = node.state[4]
        m = node.state[3]
        return 2 * N - (k + m)

    def successor(self, state):
        misioneriL = state[0]
        misioneriD = state[3]
        kanibaliL = state[1]
        kanibaliD = state[4]
        rez = dict()
        for i in range(1, K + 1):
            for j in range(i + 1):
                k = i - j
                if state[2] == 'L':
                    newstate = (misioneriL - j, kanibaliL - k, 'D', misioneriD + j, kanibaliD + k)
                    if self.isValid(newstate):
                        a = str(j) + str(k)
                        rez[a] = newstate
                else:
                    newstate = (misioneriL + j, kanibaliL + k, 'L', misioneriD - j, kanibaliD - k)
                    if self.isValid(newstate):
                        a = str(j) + str(k)
                        rez[a] = newstate
        return rez


kl = Misioneri((N, N, 'L', 0, 0), (0, 0, 'D', N, N))
rez = astar_search(kl)
print rez.solve()