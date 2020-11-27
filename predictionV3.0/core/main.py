import core.lstm_prediction as lp
import core.cnn_prediction as cp
import data.data_loader as dl


if __name__ == '__main__':
    company_list = ["AAPL", "GOOG", "MSFT", "AMZN"]
    company_name = ["APPLE", "GOOGLE", "MICROSOFT", "AMAZON"]

    for company, com_name in zip(company_list, company_name):
        print(company)
        print(com_name)