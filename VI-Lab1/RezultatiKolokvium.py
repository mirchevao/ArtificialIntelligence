def poeni_kolokvium(poeni1, poeni2):
    return int(poeni1 + poeni2)

def main():
    n = int(input("Enter number of students: "))
    for x in range(0,n):
        id = int(input("Enter the id number of the student: "))
        subject = input("Enter the subject: ")
        points_1stTerm = int(input("Enter the points from the first term: "))
        points_2ndTerm = int(input("Enter the points from the secon term: "))
        d = {'ID: ': id, 'Subject: ': subject, 'Total points: ' : poeni_kolokvium(points_1stTerm, points_2ndTerm)}
        print(d)

main()

