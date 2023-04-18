from numpy import corrcoef

def _get_x():
    try:
        n = int(input('\nEnter length of x related values: '))
        x_list = []
        for i in range(n):
            if i < 1:
                element = float(input('Enter first x value: '))
            else:
                element = float(input('Enter next x value: '))
            x_list.append(element)
        return x_list
    except ValueError:
        print('You entered an invalid input')

def _get_y(n):
    try:
        print('\nNow enter y related values')
        y_list = []
        for i in range(n):
            if i < 1:
                element = float(input('Enter first y value: '))
            else:
                element = float(input('Enter next y value: '))
            y_list.append(element)
        return y_list
    except ValueError:
        print('You entered an invalid input')

def linear_corr_regr():
    print('\n#### LINEAR CORRELATION/REGRESSION ####')

    x = _get_x()
    n = len(x)
    y = _get_y(n)

    corr_matrix = corrcoef(x, y)
    corr = corr_matrix[0,1]
    r_sq = corr**2

    print(f'\nCorrelation (r): {corr}')
    print(f'r² {r_sq}')
