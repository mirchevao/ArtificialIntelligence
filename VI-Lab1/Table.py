"""Tablica na kvadrati, kubovi i koreni"""
import math
def tablica(n):

        d = { n: (n*n, n*n*n, round(math.sqrt(n),5) ) }
        print(d)

def main():

    r = int(input("Enter range: "))
    key = int(input("Enter key value: "))
    for x in range(1, r):
        if(x == key):
            print(tablica(key))
        else:
            print("Nema podatoci")



main()

