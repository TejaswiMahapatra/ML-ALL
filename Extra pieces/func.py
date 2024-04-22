def calculate_total(exp):
    total = 0
    for item in exp:
        total = total + item
    return total


toms_list = [2100, 3100, 4100]
hoes_list = [6000, 9000, 12000]
toms_total = calculate_total(toms_list)
hoes_total = calculate_total(hoes_list)
print("Tom's total expenses:", toms_total)
print("Hoe's total expenses:", hoes_total)
