from math import log
from math import floor


def unique_counts(rows):
    """Креирај броење на можни резултате (последната колона
    во секоја редица е класата)

    :param rows: dataset
    :type rows: list
    :return: dictionary of possible classes as keys and count
             as values
    :rtype: dict
    """
    results = {}
    for row in rows:
        # Клацата е последната колона
        r = row[len(row) - 1]
        if r not in results:
            results[r] = 0
        results[r] += 1
    return results


def gini_impurity(rows):
    """Probability that a randomly placed item will
    be in the wrong category

    :param rows: dataset
    :type rows: list
    :return: Gini impurity
    :rtype: float
    """
    total = len(rows)
    counts = unique_counts(rows)
    imp = 0
    for k1 in counts:
        p1 = float(counts[k1]) / total
        for k2 in counts:
            if k1 == k2:
                continue
            p2 = float(counts[k2]) / total
            imp += p1 * p2
    return imp


def entropy(rows):
    """Ентропијата е сума од p(x)log(p(x)) за сите
    можни резултати

    :param rows: податочно множество
    :type rows: list
    :return: вредност за ентропијата
    :rtype: float
    """
    log2 = lambda x: log(x) / log(2)
    results = unique_counts(rows)
    # Пресметка на ентропијата
    ent = 0.0
    for r in results.keys():
        p = float(results[r]) / len(rows)
        ent = ent - p * log2(p)
    return ent


class DecisionNode:
    def __init__(self, col=-1, value=None, results=None, tb=None, fb=None):
        """
        :param col: индексот на колоната (атрибутот) од тренинг множеството
                    која се претставува со оваа инстанца т.е. со овој јазол
        :type col: int
        :param value: вредноста на јазолот според кој се дели дрвото
        :param results: резултати за тековната гранка, вредност (различна
                        од None) само кај јазлите-листови во кои се донесува
                        одлуката.
        :type results: dict
        :param tb: гранка која се дели од тековниот јазол кога вредноста е
                   еднаква на value
        :type tb: DecisionNode
        :param fb: гранка која се дели од тековниот јазол кога вредноста е
                   различна од value
        :type fb: DecisionNode
        """
        self.col = col
        self.value = value
        self.results = results
        self.tb = tb
        self.fb = fb


def compare_numerical(row, column, value):
    """Споредба на вредноста од редицата на посакуваната колона со
    зададена нумеричка вредност

    :param row: дадена редица во податочното множество
    :type row: list
    :param column: индекс на колоната (атрибутот) од тренирачкото множество
    :type column: int
    :param value: вредност на јазелот во согласност со кој се прави
                  поделбата во дрвото
    :type value: int or float
    :return: True ако редицата >= value, инаку False
    :rtype: bool
    """
    return row[column] >= value


def compare_nominal(row, column, value):
    """Споредба на вредноста од редицата на посакуваната колона со
    зададена номинална вредност

    :param row: дадена редица во податочното множество
    :type row: list
    :param column: индекс на колоната (атрибутот) од тренирачкото множество
    :type column: int
    :param value: вредност на јазелот во согласност со кој се прави
                  поделбата во дрвото
    :type value: str
    :return: True ако редицата == value, инаку False
    :rtype: bool
    """
    return row[column] == value


def divide_set(rows, column, value):
    """Поделба на множеството според одредена колона. Може да се справи
    со нумерички или номинални вредности.

    :param rows: тренирачко множество
    :type rows: list(list)
    :param column: индекс на колоната (атрибутот) од тренирачкото множество
    :type column: int
    :param value: вредност на јазелот во зависност со кој се прави поделбата
                  во дрвото за конкретната гранка
    :type value: int or float or str
    :return: поделени подмножества
    :rtype: list, list
    """
    # Направи функција која ни кажува дали редицата е во
    # првата група (True) или втората група (False)
    if isinstance(value, int) or isinstance(value, float):
        # ако вредноста за споредба е од тип int или float
        split_function = compare_numerical
    else:
        # ако вредноста за споредба е од друг тип (string)
        split_function = compare_nominal

    # Подели ги редиците во две подмножества и врати ги
    # за секој ред за кој split_function враќа True
    set1 = [row for row in rows if
            split_function(row, column, value)]
    # за секој ред за кој split_function враќа False
    set2 = [row for row in rows if
            not split_function(row, column, value)]
    return set1, set2


def build_tree(rows, scoref=entropy):
    if len(rows) == 0:
        return DecisionNode()
    current_score = scoref(rows)

    # променливи со кои следиме кој критериум е најдобар
    best_gain = 0.0
    best_criteria = None
    best_sets = None

    column_count = len(rows[0]) - 1
    for col in range(0, column_count):
        # за секоја колона (col се движи во интервалот од 0 до
        # column_count - 1)
        # Следниов циклус е за генерирање на речник од различни
        # вредности во оваа колона
        column_values = {}
        for row in rows:
            column_values[row[col]] = 1
        # за секоја редица се зема вредноста во оваа колона и се
        # поставува како клуч во column_values
        for value in column_values.keys():
            (set1, set2) = divide_set(rows, col, value)

            # Информациона добивка
            p = float(len(set1)) / len(rows)
            gain = current_score - p * scoref(set1) - (1 - p) * scoref(set2)
            if gain > best_gain and len(set1) > 0 and len(set2) > 0:
                best_gain = gain
                best_criteria = (col, value)
                best_sets = (set1, set2)

    # Креирај ги подгранките
    if best_gain > 0:
        true_branch = build_tree(best_sets[0], scoref)
        false_branch = build_tree(best_sets[1], scoref)
        return DecisionNode(col=best_criteria[0], value=best_criteria[1],
                            tb=true_branch, fb=false_branch)
    else:
        return DecisionNode(results=unique_counts(rows))


def print_tree(tree, indent=''):
    # Дали е ова лист јазел?
    if tree.results:
        print(str(tree.results))
    else:
        # Се печати условот
        print(str(tree.col) + ':' + str(tree.value) + '? Level= (' + str(tree.level) + ')' )
        # Се печатат True гранките, па False гранките
        print(indent + 'T->', end='')
        print_tree(tree.tb, indent + '  ')
        print(indent + 'F->', end='')
        print_tree(tree.fb, indent + '  ')


def classify(observation, tree):
    if tree.results:
        return tree.results
    else:
        value = observation[tree.col]
        if isinstance(value, int) or isinstance(value, float):
            compare = compare_numerical
        else:
            compare = compare_nominal

        if compare(observation, tree.col, tree.value):
            branch = tree.tb
        else:
            branch = tree.fb

        return classify(observation, branch)
dataset = [
    [6.3, 2.9, 5.6, 1.8, 0],
    [6.5, 3.0, 5.8, 2.2, 0],
    [7.6, 3.0, 6.6, 2.1, 0],
    [4.9, 2.5, 4.5, 1.7, 0],
    [7.3, 2.9, 6.3, 1.8, 0],
    [6.7, 2.5, 5.8, 1.8, 0],
    [7.2, 3.6, 6.1, 2.5, 0],
    [6.5, 3.2, 5.1, 2.0, 0],
    [6.4, 2.7, 5.3, 1.9, 0],
    [6.8, 3.0, 5.5, 2.1, 0],
    [5.7, 2.5, 5.0, 2.0, 0],
    [5.8, 2.8, 5.1, 2.4, 0],
    [6.4, 3.2, 5.3, 2.3, 0],
    [6.5, 3.0, 5.5, 1.8, 0],
    [7.7, 3.8, 6.7, 2.2, 0],
    [7.7, 2.6, 6.9, 2.3, 0],
    [6.0, 2.2, 5.0, 1.5, 0],
    [6.9, 3.2, 5.7, 2.3, 0],
    [5.6, 2.8, 4.9, 2.0, 0],
    [7.7, 2.8, 6.7, 2.0, 0],
    [6.3, 2.7, 4.9, 1.8, 0],
    [6.7, 3.3, 5.7, 2.1, 0],
    [7.2, 3.2, 6.0, 1.8, 0],
    [6.2, 2.8, 4.8, 1.8, 0],
    [6.1, 3.0, 4.9, 1.8, 0],
    [6.4, 2.8, 5.6, 2.1, 0],
    [7.2, 3.0, 5.8, 1.6, 0],
    [7.4, 2.8, 6.1, 1.9, 0],
    [7.9, 3.8, 6.4, 2.0, 0],
    [6.4, 2.8, 5.6, 2.2, 0],
    [6.3, 2.8, 5.1, 1.5, 0],
    [6.1, 2.6, 5.6, 1.4, 0],
    [7.7, 3.0, 6.1, 2.3, 0],
    [6.3, 3.4, 5.6, 2.4, 0],
    [5.1, 3.5, 1.4, 0.2, 1],
    [4.9, 3.0, 1.4, 0.2, 1],
    [4.7, 3.2, 1.3, 0.2, 1],
    [4.6, 3.1, 1.5, 0.2, 1],
    [5.0, 3.6, 1.4, 0.2, 1],
    [5.4, 3.9, 1.7, 0.4, 1],
    [4.6, 3.4, 1.4, 0.3, 1],
    [5.0, 3.4, 1.5, 0.2, 1],
    [4.4, 2.9, 1.4, 0.2, 1],
    [4.9, 3.1, 1.5, 0.1, 1],
    [5.4, 3.7, 1.5, 0.2, 1],
    [4.8, 3.4, 1.6, 0.2, 1],
    [4.8, 3.0, 1.4, 0.1, 1],
    [4.3, 3.0, 1.1, 0.1, 1],
    [5.8, 4.0, 1.2, 0.2, 1],
    [5.7, 4.4, 1.5, 0.4, 1],
    [5.4, 3.9, 1.3, 0.4, 1],
    [5.1, 3.5, 1.4, 0.3, 1],
    [5.7, 3.8, 1.7, 0.3, 1],
    [5.1, 3.8, 1.5, 0.3, 1],
    [5.4, 3.4, 1.7, 0.2, 1],
    [5.1, 3.7, 1.5, 0.4, 1],
    [4.6, 3.6, 1.0, 0.2, 1],
    [5.1, 3.3, 1.7, 0.5, 1],
    [4.8, 3.4, 1.9, 0.2, 1],
    [5.0, 3.0, 1.6, 0.2, 1],
    [5.0, 3.4, 1.6, 0.4, 1],
    [5.2, 3.5, 1.5, 0.2, 1],
    [5.2, 3.4, 1.4, 0.2, 1],
    [5.5, 2.3, 4.0, 1.3, 2],
    [6.5, 2.8, 4.6, 1.5, 2],
    [5.7, 2.8, 4.5, 1.3, 2],
    [6.3, 3.3, 4.7, 1.6, 2],
    [4.9, 2.4, 3.3, 1.0, 2],
    [6.6, 2.9, 4.6, 1.3, 2],
    [5.2, 2.7, 3.9, 1.4, 2],
    [5.0, 2.0, 3.5, 1.0, 2],
    [5.9, 3.0, 4.2, 1.5, 2],
    [6.0, 2.2, 4.0, 1.0, 2],
    [6.1, 2.9, 4.7, 1.4, 2],
    [5.6, 2.9, 3.6, 1.3, 2],
    [6.7, 3.1, 4.4, 1.4, 2],
    [5.6, 3.0, 4.5, 1.5, 2],
    [5.8, 2.7, 4.1, 1.0, 2],
    [6.2, 2.2, 4.5, 1.5, 2],
    [5.6, 2.5, 3.9, 1.1, 2],
    [5.9, 3.2, 4.8, 1.8, 2],
    [6.1, 2.8, 4.0, 1.3, 2],
    [6.3, 2.5, 4.9, 1.5, 2],
    [6.1, 2.8, 4.7, 1.2, 2],
    [6.4, 2.9, 4.3, 1.3, 2],
    [6.6, 3.0, 4.4, 1.4, 2],
    [6.8, 2.8, 4.8, 1.4, 2],
    [6.7, 3.0, 5.0, 1.7, 2],
    [6.0, 2.9, 4.5, 1.5, 2],
    [5.7, 2.6, 3.5, 1.0, 2],
    [5.5, 2.4, 3.8, 1.1, 2],
    [5.4, 3.0, 4.5, 1.5, 2],
    [6.0, 3.4, 4.5, 1.6, 2],
    [6.7, 3.1, 4.7, 1.5, 2],
    [6.3, 2.3, 4.4, 1.3, 2],
    [5.6, 3.0, 4.1, 1.3, 2],
    [5.5, 2.5, 4.0, 1.3, 2],
    [5.5, 2.6, 4.4, 1.2, 2],
    [6.1, 3.0, 4.6, 1.4, 2],
    [5.8, 2.6, 4.0, 1.2, 2],
    [5.0, 2.3, 3.3, 1.0, 2],
    [5.6, 2.7, 4.2, 1.3, 2],
    [5.7, 3.0, 4.2, 1.2, 2],
    [5.7, 2.9, 4.2, 1.3, 2],
    [6.2, 2.9, 4.3, 1.3, 2],
    [5.1, 2.5, 3.0, 1.1, 2],
    [5.7, 2.8, 4.1, 1.3, 2],
    [6.4, 3.1, 5.5, 1.8, 0],
    [6.0, 3.0, 4.8, 1.8, 0],
    [6.9, 3.1, 5.4, 2.1, 0],
    [6.8, 3.2, 5.9, 2.3, 0],
    [6.7, 3.3, 5.7, 2.5, 0],
    [6.7, 3.0, 5.2, 2.3, 0],
    [6.3, 2.5, 5.0, 1.9, 0],
    [6.5, 3.0, 5.2, 2.0, 0],
    [6.2, 3.4, 5.4, 2.3, 0],
    [4.7, 3.2, 1.6, 0.2, 1],
    [4.8, 3.1, 1.6, 0.2, 1],
    [5.4, 3.4, 1.5, 0.4, 1],
    [5.2, 4.1, 1.5, 0.1, 1],
    [5.5, 4.2, 1.4, 0.2, 1],
    [4.9, 3.1, 1.5, 0.2, 1],
    [5.0, 3.2, 1.2, 0.2, 1],
    [5.5, 3.5, 1.3, 0.2, 1],
    [4.9, 3.6, 1.4, 0.1, 1],
    [4.4, 3.0, 1.3, 0.2, 1],
    [5.1, 3.4, 1.5, 0.2, 1],
    [5.0, 3.5, 1.3, 0.3, 1],
    [4.5, 2.3, 1.3, 0.3, 1],
    [4.4, 3.2, 1.3, 0.2, 1],
    [5.0, 3.5, 1.6, 0.6, 1],
    [5.9, 3.0, 5.1, 1.8, 0],
    [5.1, 3.8, 1.9, 0.4, 1],
    [4.8, 3.0, 1.4, 0.3, 1],
    [5.1, 3.8, 1.6, 0.2, 1],
    [5.5, 2.4, 3.7, 1.0, 2],
    [5.8, 2.7, 3.9, 1.2, 2],
    [6.0, 2.7, 5.1, 1.6, 2],
    [6.7, 3.1, 5.6, 2.4, 0],
    [6.9, 3.1, 5.1, 2.3, 0],
    [5.8, 2.7, 5.1, 1.9, 0],
]

def extract_column(row, index):
    return [row[i] for i in range(len(row)) if i != index]

column_id = int(input())
split = int(len(dataset)*0.8)
training_set1 = dataset[:split]
testing_set1 = dataset[split:]

training_set2 = [extract_column(row, column_id) for row in training_set1]
testing_set2 = [extract_column(row, column_id) for row in testing_set1]
tree1 = build_tree(training_set1)
tree2=build_tree(training_set2)
correct1=0
correct2=0

for(row1, row2) in zip(testing_set1, testing_set2):
    prediction1 = classify(row1, tree1)
    prediction2 = classify(row2, tree2)

    if prediction1 == row1[-1]:
        correct1+=1
    if prediction2 == row2[-1]:
        correct2+=1

    f1 = correct1/len(training_set1)
    f2 = correct2/len(training_set2)
print("Tocnost na prvo drvo ", f1)
print("Tocnost na vtoro drvo", f2)
