from math import sqrt
from scipy import stats
import pandas as pd
import numpy as np

# Return squared value of each list element
def _square_list(list):
    return [i ** 2 for i in list]

def _get_x():
    try:
        n = int(input('\nEnter length of random variables (x): '))
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

def _get_x_probability(n):
    try:
        print('\nNow enter P(x) values')
        x_probability_list = []
        for i in range(n):
            if i < 1:
                element = float(input('Enter first P(x) value: '))
            else:
                element = float(input('Enter next P(x) value: '))
            x_probability_list.append(element)
        return x_probability_list
    except ValueError:
        print('You entered an invalid input')

#  x = [0,1,2,3] # Random variables
#  x_probability = [0.46,0.41,0.09,0.04] # Random variables probabilities
def probability_dist():
    print('\n#### PROBABILITY DISTRIBUTION ####')

    print('NOTE: Where x assumes all possible values.')
    print('µ = [x • P(x)] ==> Mean')
    print(' = [(x - µ)² • P(x)] ==> Variance')
    print(' = [x² • P(x)] - µ² ==> Variance (shortcut)')
    print(' = √[x² • P(x)] - µ² ==> Standard Deviation')

    x = _get_x()
    n = len(x)
    x_probability = _get_x_probability(n)
    x_multiply_x_probability = [a * b for a, b in zip(x, x_probability)]

    d = {
        'x': x,
        'P(x)': x_probability,
        'xP(x)': x_multiply_x_probability
    }

    df = pd.DataFrame(data=d)
    total_p_x = np.sum(x_probability) # Probability
    total_x_p_x = np.sum(x_multiply_x_probability) # Mean
    x_squared = _square_list(x)
    x_squared_multiply_x_probability = [a * b for a, b in zip(x_squared, x_probability)]
    x_squared_multiply_x_probability_total = np.sum(x_squared_multiply_x_probability)
    variance = x_squared_multiply_x_probability_total - (total_x_p_x ** 2)
    standard_deviation = sqrt(variance)

    print('\nProbability Distribution Table')
    print(df)
    print('\nTotal P(x): ', total_p_x)
    print('Total xP(x) / Mean (μ): ', total_x_p_x)
    print('Standard Deviation (prob. dist.): ', standard_deviation)

    exit()


# Hypothesis Testing
def hypothesis_testing():
    print('\n#### TEST STATISTICS (ONE POPULATION ####)\n')
    print('QUESTION: Is the population standard deviation know?')
    print('QUESTION: Is the distribution left tailed, right tailed or 2 tailed?')
    print('QUESTION: Is normal or t distribution?')
    print('NOTE: If 2 tailed, level of significance is divided by 2')

    is_left_tailed = False
    is_right_tailed = False
    is_two_tailed = False
    is_normal_distribution = False
    is_t_distribution = False
    has_population_standard_deviation = True
    p_value = None

    print('\n#### Operators #### \n0 ==> = \n1 ==> < \n2 ==> > \n3 ==> <= \n4 ==> >= \n5 ==> != \n')

    operators = ['=', '<', '>', '<=', '>=', '!=']
    operator = int(input('Enter claim operator: '))
    claim_value = float(input('Enter claim value: '))

    H0 = f'H0: μ {operators[operator]} {claim_value} claim'

    match operator:
        case 0:
            H1 = f'H1: μ {operators[5]} {claim_value}'
        case 1:
            H1 = f'H1: μ {operators[2]} {claim_value}'
        case 2:
            H1 = f'H1: μ {operators[1]} {claim_value}'
        case 3:
            H1 = f'H1: μ {operators[4]} {claim_value}'
        case 4:
            H1 = f'H1: μ {operators[3]} {claim_value}'
        case 5:
            H1 = f'H1: μ {operators[0]} {claim_value}'
        case _:
            print("invalid operator")
            exit()

    print(H0)
    print(H1)

    random_sample = int(input('Enter random sample: '))

    mean_x = float(input('Enter mean x (x = placeholder variable): '))
    population_standard_deviation = input('Enter population standard deviation or None: ')

    if not population_standard_deviation:
        has_population_standard_deviation = False
        print('No population standard devation.')
    else:
        population_standard_deviation = float(population_standard_deviation)

    if random_sample >= 30 and population_standard_deviation:
        is_normal_distribution = True
        print('\nNOTE: Random sample is greater than 30 and population standard deviation is present; therefore, we will use normal distrubition')
    else:
        is_t_distribution = True
        print('\nNOTE: Random sample is less than 30 and population standard deviation is NOT present; therefore, we will use t distrubition')


    level_of_significance = float(input('\nEnter level of significance: '))
    print()


    # Distribution type
    if '<' in H1:
        is_left_tailed = True
        critical_value_1 = stats.norm.ppf(level_of_significance)
        print('Distribution is left tailed')
        print('Critical value: ', critical_value_1)
    elif '>' in H1:
        is_right_tailed = True
        critical_value_1 = stats.norm.cdf(level_of_significance)
        print('Distribution is right tailed')
        print('Critical value: ', critical_value_1)
    elif '=' in H1:
        is_two_tailed = True
        print('Distribution is two tailed')
        print('Level of significance is divide by 2 since the distribution is two tailed')
        area_in_left_tail = round((level_of_significance / 2), 4)
        area_in_right_tail = area_in_left_tail
        print(f'Area in left tail: {area_in_left_tail} || Area in right tail {area_in_right_tail}')
        critical_value_left = round(stats.norm.ppf(area_in_left_tail), 4)
        critical_value_right = -critical_value_left # Negative of
        print(f'Critical values: {critical_value_left} and {critical_value_right}')


    # Test statistic
    if not has_population_standard_deviation:
        # TODO Add alternate calculation
        s = 0 # A different value
    else:
        s = population_standard_deviation

    z_obtain = round((mean_x - claim_value) / (s / sqrt(random_sample)), 4)
    print('Z-obt:', z_obtain)

    if is_left_tailed or is_right_tailed:
        p_value = round(stats.norm.cdf(z_obtain), 4)
        print('P-value for left or right tail:', p_value)
    else:
        p_value = round(stats.norm.cdf(z_obtain), 4)
        area_in_left_tail = round(1 - p_value, 4)
        area_in_right_tail = area_in_left_tail
        print(f'Value in left tail {area_in_left_tail} || Value in right tail {area_in_right_tail}')
        print('P-value:', area_in_left_tail * 2)

    if (1 - p_value) * 2 < level_of_significance:
        print('\nReject the null.\nThere is not enough significant evidence to support the claim')
    else:
        print('\nFailed to reject the null.\nThere is enough significant evidence to support the claim')

    exit()