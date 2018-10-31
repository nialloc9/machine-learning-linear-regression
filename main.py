import quandl, math, datetime, pickle, numpy
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot, style
from os import path

# hard code ticker but this could be passed from client

style.use('ggplot')

'''
Description:

    Requests data from quandl.com using a ticker supplied

Attributes:
    [1] ticker symbol of stock
    [2] start date
    [3] end date

Returns:
    dataframe
'''


def request_data_frame (ticker):
    quandl_ticker = "WIKI/" + ticker
    return quandl.get(quandl_ticker)


'''
Description:

    Gets number of days to forecast for based on months

Attributes:
    (1) [int] number of months
    (2) [list] data frame

Returns:
    integer
'''

'''
    Description:
        gets the number of days to forecast
    
    Attributes:
        (1) [int] rows
        (2) [list] data frame
    Returns:
        [int] number of days
'''


def get_forecast_days(rows, d_frame):
    return int(math.ceil(rows * len(d_frame)))


'''
    Description:
        filters a data frame and returns a new data frame with only the rows requested
        
    Attributes:
        (1) [list] data frame
        (2) [list] rows
        
    Returns:
        [list] data frame
'''


def filter_data_frame(d_frame, frame_filter):
    return d_frame[frame_filter]


'''
    Description:
        calculates high low percentage
        
    Attributes:
        (1) [list] data frame
        
    Returns:
        [float] percentage
'''


def calculate_hl_percentage(d_frame):
    return (d_frame['Adj. High'] - d_frame["Adj. Close"]) / d_frame["Adj. Close"] * 100.0


'''
    Description:
        calculates change percentage

    Attributes:
        (1) [list] data frame

    Returns:
        [float] percentage
'''


def calculate_percentage_change(d_frame):
    return (d_frame['Adj. Close'] - d_frame["Adj. Open"]) / d_frame["Adj. Open"]


'''
    Description:
        gets label

    Attributes:
        (1) [list] data frame
        (2) [string] column to use as label
        (3) [int] num of days to forecast

    Returns:
        [list] label
'''


def get_label(d_frame, column, days_out):
    return d_frame[column].shift(-days_out)



'''
    Description:
        gets features

    Attributes:
        (1) [list] data frame

    Returns:
        [array] features
'''


def get_features(d_frame):
    return numpy.array(d_frame.drop(['label'], 1))


'''
    Description:
        processes features and shuffles them with mean at the center

    Attributes:
        (1) [list] data frame
        (2) [int] num of days to forecast

    Returns:
        [list] data frame
'''


def process_features(features_to_process):
    return  preprocessing.scale(features_to_process)

'''
    Description:
        gets recent features

    Attributes:
        (1) [list] features
        (2) [int] num of days to forecast

    Returns:
        [list] features
'''


def get_recent_features(features_to_filter, days):
    return features_to_filter[-days:]


'''
    Description:
        splits features and labels into test and train

    Attributes:
        (1) [list] features
        (2) [list] labels
        (3) [int] size

    Returns:
        [list] tests and training data
'''


def split_data(features_to_test, labels_to_test, size):
    return model_selection.train_test_split(features_to_test, labels_to_test, test_size=size)


'''
    Description:
        creates a classifier to test and train

    Attributes:
        (1) [int] jobs number of threads to use

    Returns:
        [*] classifier
'''


def get_classifier(jobs):
    return LinearRegression(n_jobs=jobs)


'''
    Description:
        serializes data to a file

    Attributes:
        (1) [string] location to save file

    Returns:
        [void]
'''


def serialize(uri, obj):
    with open(uri, "wb") as file:
        pickle.dump(obj, file)


'''
    Description:
        creates a classifier to test and train

    Attributes:
        (1) [string] location to save file
        (2) [list] features to train
        (3) [list] labels to train

    Returns:
        [*] classifier
'''


def get_trained_classifier(uri, features_to_train, labels_to_train):
    if path.isfile(uri):
        pickle_in = open(uri, "rb")
        return pickle.load(pickle_in)
    else:
        clf = get_classifier(-1)
        clf.fit(features_to_train, labels_to_train)
        serialize(uri, clf)
        return clf


'''
    Description:
        add dates to data frame

    Attributes:
        (1) [list] data frame

    Returns:
        [list] data frame
'''


def add_dates(d_frame, predicted_forecast):
    last_date = d_frame.iloc[-1].name

    last_unix = last_date.timestamp()

    one_day = 86400 * 365 / 12

    next_unix = last_unix + one_day

    for i in predicted_forecast:
        next_date = datetime.datetime.fromtimestamp(next_unix)
        next_unix += one_day

        d_frame.loc[next_date] = [numpy.nan for _ in range(len(d_frame.columns) - 1)] + [i]

    return d_frame


'''
    Description:
        creates a chart to display the data

    Attributes:
        (1) [list] data frame

    Returns:
        [void]
'''


def create_chart(d_frame, title):
    d_frame["Adj. Close"].plot()
    d_frame["FORECAST"].plot()
    pyplot.legend(loc=4)
    pyplot.title(title)
    pyplot.xlabel('Date')
    pyplot.ylabel('Price')
    pyplot.show()


def main():
    default_ticker = input("Please input a ticker symbol e.g GOOGL ").upper()

    date_now = datetime.datetime.today().strftime('%Y-%m-%d')

    date_one_year_ago = (datetime.datetime.utcnow().replace(day=1) - datetime.timedelta(days=365)).strftime('%Y-%m-%d')

    data_frame = request_data_frame(default_ticker)

    forecast_days = get_forecast_days(0.01, data_frame)

    # retrieve only the following columns
    data_frame = filter_data_frame(data_frame, ["Adj. Open", "Adj. High", "Adj. Low", "Adj. Close", "Adj. Volume"])

    # assign new features
    data_frame['HL_PCT'] = calculate_hl_percentage(data_frame)
    data_frame["PCT_CHANGE"] = calculate_percentage_change(data_frame)

    data_frame = filter_data_frame(data_frame, ["Adj. Close", "HL_PCT", "PCT_CHANGE"])

    data_frame.fillna(-9999, inplace=True)

    data_frame['label'] = get_label(data_frame, "Adj. Close", forecast_days)

    features = get_features(data_frame)

    features = process_features(features)

    recent_features = get_recent_features(features, forecast_days)

    features = features[:-forecast_days]

    data_frame.dropna(inplace=True)

    labels = numpy.array(data_frame['label'])

    features_train, features_test, labels_train, labels_test = split_data(features, labels, 0.2)

    classifier = get_trained_classifier(default_ticker + ".pickle", features_train, labels_train)

    accuracy = classifier.score(features_test, labels_test)

    # prediction
    predicted_forecast = classifier.predict(recent_features)

    data_frame['FORECAST'] = numpy.nan

    data_frame = add_dates(data_frame, predicted_forecast)

    print(data_frame)

    create_chart(data_frame, default_ticker + " forecast prices.")


main()
