from utils import Problem
from uninformed_search import breadth_first_graph_search

stars = [(x1,y1), (x2, y2), (x3, y3)]

def mrdni_konj_dole_levo(konj, lovec, stars):
    x = konj[0]
    y = konj[1]
    while (0 < x < 9 and 0 < y < 9) and (konj not in lovec):
        konj_new = x-1, y-2
        if konj_new in stars and stars != 0:
            print("star--;")
    position_new = konj_new, lovec, stars
    return position_new
def mrdni_konj_dole_desno(konj, lovec, stars):
    x = konj[0]
    y = konj[1]
    while (0 < x < 9 and 0 < y < 9) and (konj not in lovec):
        konj_new = x+1, y-2
        if konj_new in stars and stars != 0:
            print("star--;")
    position_new = konj_new, lovec, stars
    return position_new

def mrdni_konj_gore_levo(konj, lovec, stars):
    x = konj[0]
    y = konj[1]
    while (0 < x < 9 and 0 < y < 9) and (konj not in lovec):
        konj_new = x-1, y+2
        if konj_new in stars and stars != 0:
            print("star--;")
    position_new = konj_new, lovec, stars
    return position_new

def mrdni_konj_dole_levo(konj, lovec, stars):
    x = konj[0]
    y = konj[1]
    while (0 < x < 9 and 0 < y < 9) and (konj not in lovec):
        konj_new = x+1, y+2
        if konj_new in stars and stars != 0:
            print("star--;")
    position_new = konj_new, lovec, stars
    return position_new

def mrdni_konj_levo_dole(konj, lovec, stars):
    x = konj[0]
    y = konj[1]
    while (0 < x < 9 and 0 < y < 9) and (konj not in lovec):
        konj_new = x-2, y-1
        if konj_new in stars and stars != 0:
            print("star--;")
    position_new = konj_new, lovec, stars
    return position_new

def mrdni_konj_levo_gore(konj, lovec, stars):
    x = konj[0]
    y = konj[1]
    while (0 < x < 9 and 0 < y < 9) and (konj not in lovec):
        konj_new = x-2, y+1
        if konj_new in stars and stars != 0:
            print("star--;")
    position_new = konj_new, lovec, stars
    return position_new

def mrdni_konj_desno_dole(konj, lovec, stars):
    x = konj[0]
    y = konj[1]
    while (0 < x < 9 and 0 < y < 9) and (konj not in lovec):
        konj_new = x+2, y-1
        if konj_new in stars and stars != 0:
            print("star--;")
    position_new = konj_new, lovec, stars
    return position_new

def mrdni_konj_desno_gore(konj, lovec, stars):
    x = konj[0]
    y = konj[1]
    while (0 < x < 9 and 0 < y < 9) and (konj not in lovec):
        konj_new = x+2, y+1
        if konj_new in stars and stars != 0:
            print("star--;")
    position_new = konj_new, lovec, stars
    return position_new

def mrdni_lovec_gore_levo(konj, lovec, stars):
    x = lovec[0]
    y = lovec[1]
    while ( 0 < x < 9 and 0 < y < 9) and (lovec not in konj):
        lovec_new = x-1, y+1
        if lovec_new in stars and stars != 0:
            print("star--;")
    position_new = konj, lovec_new, stars
    return position_new

def mrdni_lovec_gore_desno(konj, lovec, stars):
    x = lovec[0]
    y = lovec[1]
    while ( 0 < x < 9 and 0 < y < 9) and (lovec not in konj):
        lovec_new = x+1, y+1
        if lovec_new in stars and stars != 0:
            print("star--;")
    position_new = konj, lovec_new, stars
    return position_new

def mrdni_lovec_dole_levo(konj, lovec, stars):
    x = lovec[0]
    y = lovec[1]
    while ( 0 < x < 9 and 0 < y < 9) and (lovec not in konj):
        lovec_new = x-1, y-1
        if lovec_new in stars and stars != 0:
            print("star--;")
    position_new = konj, lovec_new, stars
    return position_new

def mrdni_lovec_dole_desno(konj, lovec, stars):
    x = lovec[0]
    y = lovec[1]
    while ( 0 < x < 9 and 0 < y < 9) and (lovec not in konj):
        lovec_new = x+1, y-1
        if lovec_new in stars and stars != 0:
            print("star--;")
    position_new = konj, lovec_new, stars
    return position_new

class Star(Problem):
    def __init__(self, initial_state):
        super().__init__(initial_state, None)
    def goal_test(self, state):
        number_stars = len(state[-1])
        return number_stars == 0
    def successor(self,state):
        suc = dict()
        konj = state[0], state[1]
        lovec = state[2], state[3]
        stars = state[4]

        konj_new = mrdni_konj_desno_dole(konj, lovec, stars)
        stars_new = [s for s in stars if konj_new != s]
        state_new = konj_new[0], konj_new[1], lovec[0], lovec[1], stars_new
        if state_new != state:
            suc['k1'] = state_new
        konj_new = mrdni_konj_desno_gore(konj, lovec, stars)
        stars_new = [s for s in stars if konj_new != s]
        state_new = konj_new[0], konj_new[1], lovec[0], lovec[1], stars_new
        if state_new != state:
            suc['k2'] = state_new
        konj_new = mrdni_konj_dole_desno(konj, lovec, stars)
        stars_new = [s for s in stars if konj_new != s]
        state_new = konj_new[0], konj_new[1], lovec[0], lovec[1], stars_new
        if state_new != state:
            suc['k2'] = state_new
