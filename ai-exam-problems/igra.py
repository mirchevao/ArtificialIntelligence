srce = ((1,4),(4,4))
karo = ((1,0),(4,3))
listt = ((1,1),(2,3))
tref = ((0,0),(3,4))
znaci = (srce, karo, list, tref)

def daliNaStrelka(igrac, strelka):
    s = (strelka[0], strelka[1])
    return igrac == s

def daliZnak(igrac):
    return igrac in srce or igrac in karo or igrac in listt or igrac in tref

def transportation(igrac):
    for znak in znaci:
        if igrac == znak[0]:
            igrac == znak[1]
        elif igrac == znak[1]:
            igrac == znak[0]
    return igrac

def daliSmeeNasoka(s, strelka):
    return s == strelka[2]

class Igrac(Problem):
    def __init__(self, initial):
        self.initial = initial

    def goal_test(self, state):
        for i in state[3]:
            if i == 1:
                return False
        return True

    def h(self, node):
        state = node.state
        vrednosti = state[3]
        sum = 0
        for v in vrednosti:
            if v > 0:
                sum += 1
        return sum

    def successor(self,state):
        suc = dict()
        igrac, strelka1, strelka2, vrednosti = state
        iredica, ikolona = igrac
        vrednosti_new = vrednosti

        #desno
        ikolona_new = ikolona + 1
        vrednosti_new=vrednosti
        if ikolonaNew < 5 and (daliNaStrelka(igrac, strelka1) == False or daliSmeeNasoka("Desno", strelka1) == True) \
                and (daliNaStrelka(igrac, strelka2) == False or daliSmeeNasoka("Desno", strelka2) == True):
            igracNew = (iredica, ikolonaNew)
        if daliZnak(igracNew)==True:
            igracNew=transportation(igracNew)
            new_state = (igracNew, strelka1, strelka2, vrednosti)
        