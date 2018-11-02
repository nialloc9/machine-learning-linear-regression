import numpy, random
from statistics import mean
from matplotlib import pyplot, style

style.use('fivethirtyeight')

'''
Description: 
    creates a data set

Attributes:
    (1) [int] size -> number of data points
    (2) [int] variance -> how varied the data set is
    (3) [int] step -> how far on average to step up the y value per point
    (4) [bool] correlation -> correlation betwen the data

Returns
    (1) [array] xs -> array of x values
    (2) [array] ys -> array of y values
'''


def create_data_set(size, variance, step=2, correlation=False):
    val = 1
    y_list = []

    for i in range(size):
        y = val + random.randrange(-variance, variance)
        y_list.append(y)

        if correlation and correlation == "positive":
            val += step
        elif correlation and correlation == "negative":
            val -= step

    x_list = [i for i in range(size)]

    return numpy.array(x_list, dtype=numpy.float64), numpy.array(y_list, dtype=numpy.float64)


xs, ys = create_data_set(40, 40, 2, correlation="negative")

'''
Description: 
    squares a value

Attributes:
    (1) [int | [int]] x -> value to square

Returns
    (1) [int | [int]] squared value
'''


def squared(x):
    return x * x


'''
Description: 
    gets the best fit slope
    m = ((mean(x) * mean(y)) - mean(x * y) ) / mean(x)^2 - mean(x^2)

Attributes:
    (1) [[int]] x -> list of x values
    (2) [[int]] y -> list of y values

Returns
    (1) [int] -> best fit slope
'''


def get_best_fit_slope(x, y):
    return (mean(x) * mean(y) - mean(x * y)) / ((squared(mean(x))) - mean(squared(x)))


'''
Description: 
    gets the y intercept of a line
    b = mean(y) - (m * mean(x))

Attributes:
    (1) [[int]] slope -> the slope of a line
    (2) [[int]] x -> list of x values
    (3) [[int]] y -> list of y values

Returns
    (1) [int] -> y intercept
'''


def get_y_intercept(slope, x, y):
    return mean(y) - (slope * mean(x))


'''
Description: 
    gets the y position of the next dat point
    y = mx + b

Attributes:
    (1) [[int]] slope -> the slope of a line
    (2) [[int]] x -> list of x values
    (3) [int] y_intercept -> the point where the line intercepts y

Returns
    (1) [int] -> y
'''


def get_y(slope, x, y_intercept):
    return (slope * x) + y_intercept

'''
    Description:
        gets teh squared error or 2^2
        
    Attributes:
        (1) [list] original_y -> original list of y values
        (2) [list] line_y -> regression line
          
    Returns:
        (1) [float] squared error
'''

def get_squared_error(original_y, line_y):
    return sum(squared(line_y - original_y))


'''
    Algorithm:
        r^2 = (1-(se) - (e^2 * y)) / (e^2 * mean(ys))

    Description: 
        gets the squared error or r2 of linear regression line.
        
    Attributes:
        (1) [list] original_y -> original list of y values
        (2) [list] line_y -> regression line
          
    Returns:
        (1) [float] coefficient_of_determination -> the amount of error in the regression line
'''


def coefficient_of_determination(original_y, line_y):
    y_mean_line = [mean(original_y) for y in original_y]
    squared_error = get_squared_error(original_y, line_y)
    squared_error_y_mean = get_squared_error(original_y, y_mean_line)
    return 1 - (squared_error / squared_error_y_mean)


'''
Description: 
    gets the y position of the next dat point
    y = mx + b

Attributes:
    (1) [[int]] line -> data set representing a line with xs and ys
    (2) [int] predict_x -> the x position of the next x
    (3) [int] predict_y -> the y position of the next y
    (3) [int] r_squared -> the r_squared value of the line

Returns
    (1) [int] -> y
'''


def show_graph(line, predict_x, predict_y, r_squared):
    pyplot.title("Regression line example")

    # show on graph
    pyplot.scatter(xs, ys)

    pyplot.scatter(predict_x, predict_y, color='r', s=100)

    # plot line on graph
    pyplot.plot(line)

    pyplot.xlabel('X')
    pyplot.ylabel('Y')

    pyplot.figtext(x=0, y=0.95, s="R squared is:" + str(r_squared))
    pyplot.show()


m = get_best_fit_slope(xs, ys)

b = get_y_intercept(m, xs, ys)

# one line for loop
regression_line = [get_y(m, i, b) for i in xs]

# prediction
x_to_predict = 8
y_to_predict = get_y(m, x_to_predict, b)


r_squared = coefficient_of_determination(ys, regression_line)

# show graph
show_graph(regression_line, x_to_predict, y_to_predict, r_squared)