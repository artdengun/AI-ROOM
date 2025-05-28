from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("StringTest") \
    .getOrCreate()

data = ["Hello", "World", "Ini", "Test", "Spark"]
rdd = spark.sparkContext.parallelize(data)

# Contoh transformasi: ubah huruf ke huruf besar
result = rdd.map(lambda x: x.upper()).collect()

print(result)
spark.stop()
