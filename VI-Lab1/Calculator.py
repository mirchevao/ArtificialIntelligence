"""CALCULATOR"""

def add(number1, number2):
    return number1+number2

def sub(number1, number2):
    return number1-number2

def multiply(number1, number2):
    return number1*number2

def division(number1, number2):
    if(number2!=0):
        return float(number1/number2)
    else:
        return print("Error")

def divisionINT(number1, number2):
    return int(number1//number2)

def module(number1, number2):
    return number1%number2

def power(number1, number2):
    return number1**number2

def main():
    operand1 = int(input("Enter the first operand: "))
    operation = input("Enter the operand. Possible operands: +, -, *, /, //, %, **")
    if(operation != "+" and operation != "-" and operation != "*" and operation != "/" and operation != "//" and operation != "%" and operation != "**"):
        print("Invalid operation input")
    else:

        operand2 = int(input("Enter the second operand: "))
        if(operation == "+"):
            print(add(operand1, operand2))
        elif(operation =="-"):
            print(sub(operand1, operand2))
        elif(operation=="*"):
            print(multiply(operand1,operand2))
        elif(operation=="/"):
            print(division(operand1,operand2))
        elif(operation=="//"):
            print(divisionINT(operand1,operand2))
        elif(operation=="%"):
            print(module(operand1,operand2))
        elif(operation=="**"):
            print(power(operand1,operand2))
main()