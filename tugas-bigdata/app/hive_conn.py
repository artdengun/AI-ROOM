from pyhive import hive

def get_conn():
    return hive.Connection(host='localhost', port=10000, username='hive')
