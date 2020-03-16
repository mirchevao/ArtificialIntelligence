import bisect
import sys
from sys import maxsize as infinity

"""
Дефинирање на класа за структурата на проблемот кој ќе го решаваме со пребарување.
Класата Problem е апстрактна класа од која правиме наследување за дефинирање на основните
карактеристики на секој проблем што сакаме да го решиме
"""


class Problem:
    def __init__(self, initial, goal=None):
        self.initial = initial
        self.goal = goal

    def successor(self, state):
        """За дадена состојба, врати речник од парови {акција : состојба}
       достапни од оваа состојба. Ако има многу следбеници, употребете
       итератор кој би ги генерирал следбениците еден по еден, наместо да
       ги генерирате сите одеднаш.

       :param state: дадена состојба
       :return:  речник од парови {акција : состојба} достапни од оваа
                 состојба
       :rtype: dict
       """
        raise NotImplementedError

    def actions(self, state):
        """За дадена состојба state, врати листа од сите акции што може да
       се применат над таа состојба

       :param state: дадена состојба
       :return: листа на акции
       :rtype: list
       """
        return self.successor(state).keys()

    def result(self, state, action):
        """За дадена состојба state и акција action, врати ја состојбата
       што се добива со примена на акцијата над состојбата

       :param state: дадена состојба
       :param action: дадена акција
       :return: резултантна состојба
       """
        possible = self.successor(state)
        return possible[action]

    def goal_test(self, state):
        """Врати True ако state е целна состојба. Даденава имплементација
       на методот директно ја споредува state со self.goal, како што е
       специфицирана во конструкторот. Имплементирајте го овој метод ако
       проверката со една целна состојба self.goal не е доволна.

       :param state: дадена состојба
       :return: дали дадената состојба е целна состојба
       :rtype: bool
       """
        return state == self.goal

    def path_cost(self, c, state1, action, state2):
        """Врати ја цената на решавачкиот пат кој пристигнува во состојбата
       state2 од состојбата state1 преку акцијата action, претпоставувајќи
       дека цената на патот до состојбата state1 е c. Ако проблемот е таков
       што патот не е важен, оваа функција ќе ја разгледува само состојбата
       state2. Ако патот е важен, ќе ја разгледува цената c и можеби и
       state1 и action. Даденава имплементација му доделува цена 1 на секој
       чекор од патот.

       :param c: цена на патот до состојбата state1
       :param state1: дадена моментална состојба
       :param action: акција која треба да се изврши
       :param state2: состојба во која треба да се стигне
       :return: цена на патот по извршување на акцијата
       :rtype: float
       """
        return c + 1

    def value(self):
        """За проблеми на оптимизација, секоја состојба си има вредност.
       Hill-climbing и сличните алгоритми се обидуваат да ја максимизираат
       оваа вредност.

       :return: вредност на состојба
       :rtype: float
       """
        raise NotImplementedError


"""
Дефинирање на класата за структурата на јазел од пребарување.
Класата Node не се наследува
"""


class Node:
    def __init__(self, state, parent=None, action=None, path_cost=0):
        """Креирај јазол од пребарувачкото дрво, добиен од parent со примена
       на акцијата action

       :param state: моментална состојба (current state)
       :param parent: родителска состојба (parent state)
       :param action: акција (action)
       :param path_cost: цена на патот (path cost)
       """
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.depth = 0  # search depth
        if parent:
            self.depth = parent.depth + 1

    def __repr__(self):
        return "<Node %s>" % (self.state,)

    def __lt__(self, node):
        return self.state < node.state

    def expand(self, problem):
        """Излистај ги јазлите достапни во еден чекор од овој јазол.

       :param problem: даден проблем
       :return: листа на достапни јазли во еден чекор
       :rtype: list(Node)
       """
        return [self.child_node(problem, action)
                for action in problem.actions(self.state)]

    def child_node(self, problem, action):
        """Дете јазел

       :param problem: даден проблем
       :param action: дадена акција
       :return: достапен јазел според дадената акција
       :rtype: Node
       """
        next_state = problem.result(self.state, action)
        return Node(next_state, self, action,
                    problem.path_cost(self.path_cost, self.state,
                                      action, next_state))

    def solution(self):
        """Врати ја секвенцата од акции за да се стигне од коренот до овој јазол.

       :return: секвенцата од акции
       :rtype: list
       """
        return [node.action for node in self.path()[1:]]

    def solve(self):
        """Врати ја секвенцата од состојби за да се стигне од коренот до овој јазол.

       :return: листа од состојби
       :rtype: list
       """
        return [node.state for node in self.path()[0:]]

    def path(self):
        """Врати ја листата од јазли што го формираат патот од коренот до овој јазол.

       :return: листа од јазли од патот
       :rtype: list(Node)
       """
        x, result = self, []
        while x:
            result.append(x)
            x = x.parent
        result.reverse()
        return result

    """Сакаме редицата од јазли кај breadth_first_search или
   astar_search да не содржи состојби - дупликати, па јазлите што
   содржат иста состојба ги третираме како исти. [Проблем: ова може
   да не биде пожелно во други ситуации.]"""

    def __eq__(self, other):
        return isinstance(other, Node) and self.state == other.state

    def __hash__(self):
        return hash(self.state)


"""
Дефинирање на помошни структури за чување на листата на генерирани, но непроверени јазли
"""


class Queue:
    """Queue е апстрактна класа / интерфејс. Постојат 3 типа:
       Stack(): Last In First Out Queue (стек).
       FIFOQueue(): First In First Out Queue (редица).
       PriorityQueue(order, f): Queue во сортиран редослед (подразбирливо,од најмалиот кон
                                најголемиот јазол).
   """

    def __init__(self):
        raise NotImplementedError

    def append(self, item):
        """Додади го елементот item во редицата

       :param item: даден елемент
       :return: None
       """
        raise NotImplementedError

    def extend(self, items):
        """Додади ги елементите items во редицата

       :param items: дадени елементи
       :return: None
       """
        raise NotImplementedError

    def pop(self):
        """Врати го првиот елемент од редицата

       :return: прв елемент
       """
        raise NotImplementedError

    def __len__(self):
        """Врати го бројот на елементи во редицата

       :return: број на елементи во редицата
       :rtype: int
       """
        raise NotImplementedError

    def __contains__(self, item):
        """Проверка дали редицата го содржи елементот item

       :param item: даден елемент
       :return: дали queue го содржи item
       :rtype: bool
       """
        raise NotImplementedError


class Stack(Queue):
    """Last-In-First-Out Queue."""

    def __init__(self):
        self.data = []

    def append(self, item):
        self.data.append(item)

    def extend(self, items):
        self.data.extend(items)

    def pop(self):
        return self.data.pop()

    def __len__(self):
        return len(self.data)

    def __contains__(self, item):
        return item in self.data


class FIFOQueue(Queue):
    """First-In-First-Out Queue."""

    def __init__(self):
        self.data = []

    def append(self, item):
        self.data.append(item)

    def extend(self, items):
        self.data.extend(items)

    def pop(self):
        return self.data.pop(0)

    def __len__(self):
        return len(self.data)

    def __contains__(self, item):
        return item in self.data


class PriorityQueue(Queue):
    """Редица во која прво се враќа минималниот (или максималниот) елемент
   (како што е определено со f и order). Оваа структура се користи кај
   информирано пребарување"""
    """"""

    def __init__(self, order=min, f=lambda x: x):
        """
       :param order: функција за подредување, ако order е min, се враќа елементот
                     со минимална f(x); ако order е max, тогаш се враќа елементот
                     со максимална f(x).
       :param f: функција f(x)
       """
        assert order in [min, max]
        self.data = []
        self.order = order
        self.f = f

    def append(self, item):
        bisect.insort_right(self.data, (self.f(item), item))

    def extend(self, items):
        for item in items:
            bisect.insort_right(self.data, (self.f(item), item))

    def pop(self):
        if self.order == min:
            return self.data.pop(0)[1]
        return self.data.pop()[1]

    def __len__(self):
        return len(self.data)

    def __contains__(self, item):
        return any(item == pair[1] for pair in self.data)

    def __getitem__(self, key):
        for _, item in self.data:
            if item == key:
                return item

    def __delitem__(self, key):
        for i, (value, item) in enumerate(self.data):
            if item == key:
                self.data.pop(i)


"""
Информирано пребарување во рамки на граф
"""


def memoize(fn, slot=None):
    """ Запамети ја пресметаната вредност за која била листа од
   аргументи. Ако е специфициран slot, зачувај го резултатот во
   тој slot на првиот аргумент. Ако slot е None, зачувај ги
   резултатите во речник.

   :param fn: зададена функција
   :param slot: име на атрибут во кој се чуваат резултатите од функцијата
   :return: функција со модификација за зачувување на резултатите
   """
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
            if args not in memoized_fn.cache:
                memoized_fn.cache[args] = fn(*args)
            return memoized_fn.cache[args]

        memoized_fn.cache = {}
    return memoized_fn


def best_first_graph_search(problem, f):
    """Пребарувај низ следбениците на даден проблем за да најдеш цел. Користи
    функција за евалуација за да се одлучи кој е сосед најмногу ветува и
    потоа да се истражи. Ако до дадена состојба стигнат два пата, употреби
    го најдобриот пат.

   :param problem: даден проблем
   :param f: дадена функција за евристика
   :return: Node or None
   """
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
    """ Greedy best-first пребарување се остварува ако се специфицира дека f(n) = h(n).

   :param problem: даден проблем
   :param h: дадена функција за евристика
   :return: Node or None
   """
    h = memoize(h or problem.h, 'h')
    return best_first_graph_search(problem, h)


def astar_search(problem, h=None):
    """ A* пребарување е best-first graph пребарување каде f(n) = g(n) + h(n).

   :param problem: даден проблем
   :param h: дадена функција за евристика
   :return: Node or None
   """
    h = memoize(h or problem.h, 'h')
    return best_first_graph_search(problem, lambda n: n.path_cost + h(n))


def recursive_best_first_search(problem, h=None):
    """Recursive best first search - ја ограничува рекурзијата
    преку следење на f-вредноста на најдобриот алтернативен пат
    од било кој јазел предок (еден чекор гледање нанапред).

   :param problem: даден проблем
   :param h: дадена функција за евристика
   :return: Node or None
   """
    h = memoize(h or problem.h, 'h')

    def RBFS(problem, node, flimit):
        if problem.goal_test(node.state):
            return node, 0  # (втората вредност е неважна)
        successors = node.expand(problem)
        if len(successors) == 0:
            return None, infinity
        for s in successors:
            s.f = max(s.path_cost + h(s), node.f)
        while True:
            # Подреди ги според најниската f вредност
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


"""
Неинформирано пребарување во рамки на дрво.
Во рамки на дрвото не разрешуваме јамки.
"""


def tree_search(problem, fringe):
    """ Пребарувај низ следбениците на даден проблем за да најдеш цел.

   :param problem: даден проблем
   :param fringe:  празна редица (queue)
   :return: Node
   """
    fringe.append(Node(problem.initial))
    while fringe:
        node = fringe.pop()
        print(node.state)
        if problem.goal_test(node.state):
            return node
        fringe.extend(node.expand(problem))
    return None


def breadth_first_tree_search(problem):
    """Експандирај го прво најплиткиот јазол во пребарувачкото дрво.

   :param problem: даден проблем
   :return: Node
   """
    return tree_search(problem, FIFOQueue())


def depth_first_tree_search(problem):
    """Експандирај го прво најдлабокиот јазол во пребарувачкото дрво.

   :param problem:даден проблем
   :return: Node
   """
    return tree_search(problem, Stack())


"""
Неинформирано пребарување во рамки на граф
Основната разлика е во тоа што овде не дозволуваме јамки,
т.е. повторување на состојби
"""


def graph_search(problem, fringe):
    """Пребарувај низ следбениците на даден проблем за да најдеш цел.
    Ако до дадена состојба стигнат два пата, употреби го најдобриот пат.

   :param problem: даден проблем
   :param fringe: празна редица (queue)
   :return: Node
   """
    closed = set()
    fringe.append(Node(problem.initial))
    while fringe:
        node = fringe.pop()
        if problem.goal_test(node.state):
            return node
        if node.state not in closed:
            closed.add(node.state)
            fringe.extend(node.expand(problem))
    return None


def breadth_first_graph_search(problem):
    """Експандирај го прво најплиткиот јазол во пребарувачкиот граф.

   :param problem: даден проблем
   :return: Node
   """
    return graph_search(problem, FIFOQueue())


def depth_first_graph_search(problem):
    """Експандирај го прво најдлабокиот јазол во пребарувачкиот граф.

   :param problem: даден проблем
   :return: Node
   """
    return graph_search(problem, Stack())


def depth_limited_search(problem, limit=50):
    def recursive_dls(node, problem, limit):
        """Помошна функција за depth limited"""
        cutoff_occurred = False
        if problem.goal_test(node.state):
            return node
        elif node.depth == limit:
            return 'cutoff'
        else:
            for successor in node.expand(problem):
                result = recursive_dls(successor, problem, limit)
                if result == 'cutoff':
                    cutoff_occurred = True
                elif result is not None:
                    return result
        if cutoff_occurred:
            return 'cutoff'
        return None

    return recursive_dls(Node(problem.initial), problem, limit)


def iterative_deepening_search(problem):
    for depth in range(sys.maxsize):
        result = depth_limited_search(problem, depth)
        if result is not 'cutoff':
            return result


def uniform_cost_search(problem):
    """Експандирај го прво јазолот со најниска цена во пребарувачкиот граф."""
    return graph_search(problem, PriorityQueue(lambda a, b:
                                               a.path_cost < b.path_cost))

def isValid(position):
    x, y = position
    if x < 0 or x > 9 or y < 0 or y > 9:
        return False
    return True

def prodolziPravo(state):
    direction = state[0]
    snake = list(state[1])
    green_apples = list(state[2])

    if direction == 'D':
        snake.append(snake[-1][0] + 1, snake[-1][1])
    elif direction == 'U':
        snake.append(snake[-1][0] - 1, snake[-1][1])
    elif direction == 'L':
        snake.append(snake[-1][0], snake[-1][1] -1)
    elif direction == 'R':
        snake.append(snake[-1][0], snake[-1][1] + 1)

    new_snake = snake
    if snake[-1] not in green_apples:
        new_snake = new_snake[1:]
        green_apples = tuple(green_apples)
    else:
        green_apples = tuple([s for s in green_apples if s != new_snake[-1]])
    if(not isValid(new_snake[-1]) or new_snake[-1] in new_snake[:-1]):
        return None
    else:
        return (direction, tuple(new_snake, green_apples))

def svrtiLevo(state):
    direction = state[0]
    snake = list(state[1])
    green_apples = list(state[2])

    if direction == 'D':
        snake.append(snake[-1][0], snake[-1][1] + 1)
        new_direction = 'R'
    elif direction == 'U':
        snake.append(snake[-1][0], snake[-1][1] -1)
        new_direction = 'L'
    elif direction == 'L':
        snake.append(snake[-1][0] + 1, snake[-1][1])
        new_direction = 'D'
    elif direction == 'R':
        snake.append(snake[-1][0]-1, snake[-1][1])
        new_direction = 'U'
    new_snake=snake
    if new_snake[-1] not in green_apples:
        new_snake=new_snake[1:]
        green_apples=tuple(green_apples)
    else:
        green_apples = tuple([s for s in green_apples if s!= new_snake[-1]])

    if( not isValid(new_snake[-1]) or new_snake[-1] in new_snake[:-1]):
        return None
    else:
        return (direction, tuple(new_snake,green_apples))
def svrtiDesno(state):
    direction = state[0]
    snake = list(state[1])
    green_apples = state[2]

    if direction == "D":
        snake.append((snake[-1][0], snake[-1][1] - 1))
        new_direction = "L"
    elif direction == "U":
        snake.append((snake[-1][0], snake[-1][1] + 1))
        new_direction = "R"
    elif direction == "L":
        snake.append((snake[-1][0] - 1, snake[-1][1]))
        new_direction = "U"
    elif direction == "R":
        snake.append((snake[-1][0] + 1, snake[-1][1]))
        new_direction = "D"

    new_snake = snake
    if new_snake[-1] not in green_apples:
        new_snake = new_snake[1:]
        green_apples = tuple(green_apples)
    else:
        green_apples = tuple([s for s in green_apples if s != new_snake[-1]])

    if (checkNotValid(new_snake[-1])) or new_snake[-1] in new_snake[:-1]:
        return None
    else:
        return (new_direction, tuple(new_snake), green_apples)
class Zmija2(Problem):
    def __init__(self, initial):
        self.initial = initial
    def goal_test(self, state):
        return len(state[-1]) == 0

    def successor(self, state):
        suc=dict()
        new_state = prodolziPravo(state)
        if new_state != None:
            suc['ProdolziPravo']=new_state
        new_state= svrtiDesno(state)
        if new_state != Node:
            suc['SvrtiDesno'] = new_state
        new_state = svrtiLevo(state)
        if new_state != None:
            suc['SvrtiLevo']= new_state
        return suc

    def actions(self, state):
        return self.successor(state).keys()

    def result(self, state, action):
        possible = self.successor(state)
        return possible[action]

    def h(self, node):
        state = node.state
        snake = state[1]
        green_apples = state[2]
        value = 0
        for apple in green_apples:
            value+= abs(snake[-1][0] - apple[0]) + abs(snake[-1][1] - apple[1])
        return value
    