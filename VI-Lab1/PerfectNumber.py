"""PERFECT NUMBER"""
"""6 is a perfect number because 6 == 1+2+3, 1,2,3->deliteli na 6"""

def perfect_number(num):
    sum=0;
    for x in range(1,num):
        if num%x == 0:
            sum+=x
    if sum == num:
        return True
    else:
        return False
def main():
    number = int(input("Enter a number to check if it is perfect or not: "))

    print(perfect_number(number))

main()






