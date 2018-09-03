
def convert_time(str_time):
    m, s = map(int, str_time.split(':'))
    return m * 60 + s
