import sys
import bisect

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

def update_obstacle1(obstacle1):
    x = obstacle1[0]
    y = obstacle1[1]
    direction = obstacle1[2]
    if ( x == 1 and direction == 1) or ( x == 5 and direction == -1):
        direction*=(-1)
    x_new = x + direction
    position_new = x_new, y, direction
    return position_new

def update_obstacle2(obstacle2):
    x = obstacle2[0]
    y = obstacle2[1]
    direction = obstacle2[2]
    if ( x == 1 and y == 1 and direction == 1) or (x == 5 and y == 5 and direction == -1):
        direction*=(-1)
    x_new = x - direction
    y_new = y - direction
    position_new = x_new, y_new, direction
    return position_new
def update_obstacle3(obstacle3):
    x = obstacle3[0]
    y = obstacle3[1]
    direction = obstacle3[2]
    if(y == 1 and direction == 1) or ( y == 5 and direction == -1):
        direction*=(-1)
    y_new = y - direction
    position_new = x, y_new, direction
    return position_new

def isValid(x,y,obstacle1, obstacle2, obstacle3):
    flag = 1;
    if(x == obstacle1[0] and y == obstacle1[1]) or (x == obstacle1[0] and y == obstacle1[1] + 1):
        flag = 0
    if (x == obstacle2[0] and y == obstacle2[1] ) or (x == obstacle2[0] and y == obstacle2[1] + 1) or (x == obstacle2[0] + 1 and y == obstacle2[1]) or (x == obstacle2[0] + 1 and y == obstacle2[1] + 1):
        flag = 0
    if (x == obstacle3[0] and y == obstacle3[1]) or (x == obstacle3[0] + 1 and y == obstacle3[1]):
        flag=0
    return flag

class Istrazuvac(Problem):
    def __init__(self, initial, goal):
        self.initial = initial
        self.goal = goal

    def successor(self, state):
        suc = dict()
        x = state[0]
        y = state[1]
        obstacle1 = state[2]
        obstacle2 = state[3]
        obstacle3 = state[4]

        #move man left
        if x < 6:
            x_new = x - 1
            y_new = y
            obstacle1_new = update_obstacle1(obstacle1)
            obstacle2_new = update_obstacle2(obstacle2)
            obstacle3_new = update_obstacle3(obstacle3)
            if(isValid(x_new, y_new, obstacle1_new, obstacle2_new, obstacle3_new)):
                state_new = x_new, y_new, obstacle1_new, obstacle2_new, obstacle3_new
                suc['Left'] = state_new
        if x > 1:
            x_new = x + 1
            y_new = y
            obstacle1_new = update_obstacle1(obstacle1)
            obstacle2_new = update_obstacle2(obstacle2)
            obstacle3_new = update_obstacle3(obstacle3)
            if(isValid(x_new, y_new, obstacle1_new, obstacle2_new, obstacle3_new)):
                state_new = x_new, y_new, obstacle1_new, obstacle2_new, obstacle3_new
                suc['Right'] = state_new
        if y > 1:
            x_new = x
            y_new = y + 1
            obstacle1_new = update_obstacle1(obstacle1)
            obstacle2_new = update_obstacle2(obstacle2)
            obstacle3_new = update_obstacle3(obstacle3)
            if (isValid(x_new, y_new, obstacle1_new, obstacle2_new, obstacle3_new)):
                state_new = x_new, y_new, obstacle1_new, obstacle2_new, obstacle3_new
                suc['Up'] = state_new

        if y < 11:
            x_new = x
            y_new = y -1
            obstacle1_new = update_obstacle1(obstacle1)
            obstacle2_new = update_obstacle2(obstacle2)
            obstacle3_new = update_obstacle3(obstacle3)
            if (isValid(x_new, y_new, obstacle1_new, obstacle2_new, obstacle3_new)):
                state_new = x_new, y_new, obstacle1_new, obstacle2_new, obstacle3_new
                suc['Down'] = state_new

        return state_new

    def actions(self, state):
        return self.successor(state).keys()

    def result(self, state, action):
        possible = self.successor(state)
        return possible[action]

    def goal_test(self, state):
        g = self.goal
        return (state[0] == g[0] and state[1] == g[1])


CoveceRedica = int(input())
CoveceKolona = int(input())
KukaRedica = int(input())
KukaKolona = int(input())

IstrazuvacInstance = Istrazuvac((CoveceRedica, CoveceKolona, (2, 2, -1), (7, 2, 1), (7, 8, 1)),(KukaRedica, KukaKolona))

answer = breadth_first_graph_search(IstrazuvacInstance)
print(answer.solution())