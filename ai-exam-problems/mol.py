infinity = float('inf')  # sistemski definirana vrednost za beskonecnost


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


def graph_search(problem, fringe):
    """Search through the successors of a problem to find a goal.
   The argument fringe should be an empty queue.
   If two paths reach a state, only use the best one."""
    closed = {}
    fringe.append(Node(problem.initial))
    while fringe:
        node = fringe.pop()
        if problem.goal_test(node.state):
            return node
        if node.state not in closed:
            closed[node.state] = True
            fringe.extend(node.expand(problem))
    return None


def breadth_first_graph_search(problem):
    "Search the shallowest nodes in the search tree first."
    return graph_search(problem, FIFOQueue())


# matrica

obstacles = [(1,4), (1,6), (1,8), (2,3), (2,9), (4,2), (4,7), (4,8), (5,5), (5,7), (6,1), (6,2), (6,4), (6,7)]

def moveAtomUp(a1, a2, a3):
    x = a1[0]
    y = a1[1]
    if ( 0 < x < 8 and 0 < y < 10 ) and (a1 not in obstacles) and (a1 not in [a2, a3]):
        y = y + 1
    a1 = x, y
    return a1

def moveAtomDown(a1, a2, a3):
    x = a1[0]
    y = a1[1]
    if (0 < x < 8 and 0 < y < 10) and (a1 not in obstacles) and (a1 not in [a2, a3]):
        y = y - 1
    a1 = x, y
    return a1

def moveAtomLeft(a1,a2,a3):
    x = a1[0]
    y = a1[1]
    if (0 < x < 8 and 0 < y < 10) and (a1 not in obstacles) and (a1 not in [a2, a3]):
        x = x - 1
    a1 = x, y
    return a1

def moveAtomRight(a1,a2,a3):
    x = a1[0]
    y = a1[1]
    if (0 < x < 8 and 0 < y < 10) and (a1 not in obstacles) and (a1 not in [a2, a3]):
        x = x + 1
    a1 = x, y
    return a1

class Molecule(Problem):

    def __init__(self, initial):
        self.initial = initial

    def goal_test(self, state):
        h1_x = state[0]
        h1_y = state[1]
        o_x = state[2]
        o_y = state[3]
        h2_x = state[4]
        h2_y = state[5]
        return h1_x == o_x and o_x == h2_x and \
               h1_y == o_y - 1 and o_y == h2_y - 1

    def successor(self, state):
        suc = dict()
        h1 = state[0], state[1]
        o = state[2], state[3]
        h2 = state[4], state[5]

        #up h1
        h1_new = moveAtomUp(h1, o, h2)
        state_new = h1_new + o + h2
        if state != state_new:
            suc['UpH1'] = state_new

        #down h1
        h1_new = moveAtomDown(h1, o, h2)
        state_new = h1_new + o + h2
        if state != state_new:
            suc['DownH1'] = state_new

        #left h1
        h1_new = moveAtomLeft(h1, o, h2)
        state_new = h1_new + o + h2
        if state != state_new:
            suc['LeftH1'] = state_new
        #right h1
        h1_new = moveAtomRight(h1, o, h2)
        state_new = h1_new + o + h2
        if state != state_new:
            suc['RightH1'] = state_new

        #o up
        o_new = moveAtomUp(o, h1, h2)
        state_new = h1 + o_new + h2
        if state_new != state:
            suc['UpO'] = state_new
        # o down
        o_new = moveAtomDown(o, h1, h2)
        state_new = h1 + o_new + h2
        if state_new != state:
            suc['DownO'] = state_new
        # o left
        o_new = moveAtomLeft(o, h1, h2)
        state_new = h1 + o_new + h2
        if state_new != state:
            suc['LeftO'] = state_new
        #o right
        o_new = moveAtomRight(o, h1, h2)
        state_new = h1 + o_new + h2
        if state_new != state:
            suc['RightO'] = state_new

        #h2 up
        h2_new = moveAtomUp(h2, o, h1)
        state_new = h1 + o + h2_new
        if state != state_new:
            suc['UpH2'] = state_new
        #h2 down
        h2_new = moveAtomDown(h2, o, h1)
        state_new = h1 + o + h2_new
        if state != state_new:
            suc['DownH2'] = state_new
        #h2 left
        h2_new = moveAtomLeft(h2, o, h1)
        state_new = h1 + o + h2_new
        if state != state_new:
            suc['LeftH2'] = state_new
        #h2 right
        h2_new = moveAtomRight(h2, o, h1)
        state_new = h1 + o + h2_new
        if state != state_new:
            suc['RightH2'] = state_new

        return suc

    def actions(self, state):
        return self.successor(state).keys()

    def result(self, state, action):
        possible = self.successor(state)
        return possible[action]

if __name__ == '__main__':
    h1_atom_row = int(input())
    h1_atom_column = int(input())
    o_atom_row = int(input())
    o_atom_column = int(input())
    h2_atom_row = int(input())
    h2_atom_column = int(input())

    molecule = Molecule((h1_atom_row, h1_atom_column, o_atom_row,
                         o_atom_column, h2_atom_row, h2_atom_column))

    answer = breadth_first_graph_search(molecule)
    print(answer.solution())