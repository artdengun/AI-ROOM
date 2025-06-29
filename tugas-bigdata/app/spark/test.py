import os
import subprocess
import time
import findspark
import spark


findspark.init()

from pyspark.sql import SparkSession, Row
def create_spark_session():
    return SparkSession.builder \
        .appName("BigData") \
        .master("spark://localhost:7077") \
        .config("spark.driver.memory", "1g") \
        .config("spark.executor.memory", "1g") \
        .config("spark.executor.cores", "1") \
        .config("spark.executor.instances", "8") \
        .config("spark.cores.max", "8") \
        .config("spark.sql.shuffle.partitions", "8") \
        .config("spark.default.parallelism", "8") \
        .getOrCreate()


spark = create_spark_session()

rdd = spark.sparkContext.parallelize(range(100), 8)

def f(x):
    import socket
    return (x, socket.gethostname())

result = rdd.map(f).collect()

for item in result:
    print(item)
