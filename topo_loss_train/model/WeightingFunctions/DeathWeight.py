"""
B should be the birth (?) of the most persistent feature in all trials of the experiment
"""


def death_weight(x, y, b=2):
    if y <= 0:
        return 0
    elif 0 < y < b:
        return y / b
    else:
        return 1
