from math import factorial
from itertools import permutations

# Return permutation
def nPr(n, r):
    return int(factorial(n)/factorial(n-r))

def permutation():
    print('#### Permutation ####\n')
    # https://en.wikipedia.org/wiki/Permutation
    print('A permutation of a set is an arrangement of its members into a sequence or linear order, or if the set is already ordered, a rearrangement of its elements. ~Wikipedia')
    print('NOTE: With Permutation ORDER matters.')
    print('0 ==> Calculation Permutations \n1 ==> Generate Permutations\n')

    opt = int(input('Please select an option: '))

    match opt:
        case 0:
            # https://www.britannica.com/science/permutation
            print('Denoted by the symbol nPr / nPk,')
            print('Read “n permute r”')
            print('If there are n objects available from which to select, and permutations (P) are to be formed using k of the objects at a time, the number of different permutations possible is denoted by the symbol nPk. ~Britannica')
            # TODO Add type validation
            n = int(input('Enter n value (e.g., 5, 23, 42): '))
            r = int(input('Enter r value (e.g., 5, 23, 42): '))
            print(f'nPr:', nPr(n, r))
            exit() # Add loop to allow user to pick a different option or exit
        case 1:
            s = input('Enter permutation values (e.g., ABCD, 1234): ')
            grouping = int(input('Enter permutation grouping (e.g., 1, 2, 3): '))
            print(list(permutations(s, grouping)))
            exit() # Add loop to allow user to pick a different option or exit
        case _:
            print("invalid option")
            exit()