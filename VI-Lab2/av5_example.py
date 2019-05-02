from utils import Problem
from uninformed_search import breadth_first_tree_search, depth_first_tree_search, \
    breadth_first_graph_search, depth_first_graph_search, depth_limited_search, \
    iterative_deepening_search

"""
--------------------------------------------------------------------------------------------------
PRIMER 1 : PROBLEM SO DVA SADA SO VODA
OPIS: 
Dadeni se dva sada J0 i J1 so kapaciteti C0 i C1. Na pocetok dvata sada se polni. Inicijalnata 
sostojba moze da se prosledi na pocetok. Problemot e kako da se stigne do sostojba vo koja J0 
ke ima G0 litri, a J1 ke ima G1 litri. 
AKCII:
1. Da se isprazni bilo koj od sadovite
2. Da se prefrli tecnosta od eden sad vo drug so toa sto ne moze da se nadmine kapacitetot na 
sadovite
Moze da ima i opcionalen tret vid na akcii 3. Napolni bilo koj od sadovite (studentite da ja 
implementiraat ovaa varijanta)
--------------------------------------------------------------------------------------------------
"""


class WJ(Problem):

    def __init__(self, capacities=(5, 2), initial=(5, 0), goal=(0, 1)):
        super().__init__(initial, goal)
        self.capacities = capacities

    def goal_test(self, state):
        """Враќа True ако состојбата state е целна состојба.

        :param state: дадена состојба
        :type state: tuple
        :return: дали состојбата е целна состојба
        :rtype: bool
        """
        g = self.goal
        return (state[0] == g[0] or g[0] == '*') and \
               (state[1] == g[1] or g[1] == '*')

    def successor(self, j):
        """Враќа речник од следбеници на состојбата

        :param j: дадена состојба
        :type j: tuple
        :return: речник од следни состојби
        :rtype: dict
        """
        successors = dict()
        j0, j1 = j
        (C0, C1) = self.capacities
        if j0 > 0:
            j_new = 0, j1
            successors['isprazni go sadot J0'] = j_new
        if j1 > 0:
            j_new = j0, 0
            successors['isprazni go sadot J1'] = j_new
        if j1 < C1 and j0 > 0:
            delta = min(j0, C1 - j1)
            j_new = j0 - delta, j1 + delta
            successors['preturi od sadot J0 vo sadot J1'] = j_new
        if j0 < C0 and j1 > 0:
            delta = min(j1, C0 - j0)
            j_new = j0 + delta, j1 - delta
            successors['preturi od sadot J1 vo sadot J0'] = j_new
        return successors

    def actions(self, state):
        return self.successor(state).keys()

    def result(self, state, action):
        possible = self.successor(state)
        return possible[action]


# So vaka definiraniot problem mozeme da gi koristime site neinformirani prebaruvanja
# Vo prodolzenie se dadeni mozni povici (vnimavajte za da moze da napravite povik mora da definirate problem)

wj_instance = WJ((5, 2), (5, 2), ('*', 1))
print(wj_instance)

answer1 = breadth_first_tree_search(wj_instance)
print(answer1.solve())

# внимавајте на овој повик, може да влезе во бесконечен циклус
answer2 = depth_first_tree_search(wj_instance)
print(answer2.solve())

answer3 = breadth_first_graph_search(wj_instance)
print(answer3.solve())

answer4 = depth_first_graph_search(wj_instance)
print(answer4.solve())

answer5 = depth_limited_search(wj_instance)
print(answer5.solve())

answer6 = iterative_deepening_search(wj_instance)
print(answer6.solve())
