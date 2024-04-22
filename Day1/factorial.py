def factorial(num):
    if num >= 50:
        print('Error code. Enter a valid input')
    else:
        if num == 1 or num == 0:
            return 1
        else:
            return num * factorial(num - 1)


x = int(input('Enter a number below 50:'))
print('Factorial of', x, 'is', factorial(x))
