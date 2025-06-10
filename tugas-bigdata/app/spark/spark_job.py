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
        .master("local[*]") \
        .config("spark.driver.memory", "2g") \
        .config("spark.sql.shuffle.partitions", "4") \
        .config("spark.default.parallelism", "4") \
        .config("spark.executor.memory", "2g") \
        .config("spark.ui.showConsoleProgress", "true") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .getOrCreate()

HDFS_CMD = r"C:\Hadoop\hadoop-3.3.6\bin\hdfs.cmd"


def hdfs_file_exists(hdfs_dir: str, filename: str) -> bool:
    """Cek apakah file dengan nama tertentu ada di dalam folder HDFS."""
    result = subprocess.run(
        [HDFS_CMD, "dfs", "-ls", hdfs_dir],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    print(f"[DEBUG] Check exists in {hdfs_dir}: returncode={result.returncode}")
    print(f"[DEBUG] stderr: {result.stderr.strip()}")

    if result.returncode != 0:
        return False

    for line in result.stdout.splitlines():
        if line.strip().endswith(f"/{filename}"):
            return True

    return False

def process_csv(spark: SparkSession, local_file_path: str):
    """Upload file ke HDFS jika belum ada dan baca pakai Spark, lalu broadcast data."""

    filename = os.path.basename(local_file_path)
    hdfs_dir = "/uploads"
    hdfs_path = f"{hdfs_dir}/{filename}"
    hdfs_uri = f"hdfs://localhost:9000{hdfs_path}"

    try:
        if not hdfs_file_exists(hdfs_dir, filename):
            subprocess.run([HDFS_CMD, "dfs", "-mkdir", "-p", hdfs_dir], check=True)
            subprocess.run([HDFS_CMD, "dfs", "-put", local_file_path, "/uploads/"], check=True)
            print(f"uploads files berhasil")
            time.sleep(1)
        else:
            print(f"File {hdfs_path} sudah ada di HDFS. Lewati upload.")
            df = spark.read.option("header", "true").csv(hdfs_uri)

            # Ambil SparkContext
            sc = spark.sparkContext

            # Broadcast kolom pertama sebagai list
            first_col = df.columns[0]
            data_list = df.select(first_col).rdd.map(lambda row: row[0]).collect()
            broadcast_data = sc.broadcast(data_list)
            print(f"[INFO] Broadcast data: {broadcast_data.value}")

            # Siapkan data untuk paralelisasi (bisa pakai index baris misal)
            data = list(range(1, 11))  # 10 worker

            rdd = sc.parallelize(data, 10)  # 10 partisi supaya ada 10 worker

            def process_partition(x):
                # Akses broadcast data di tiap worker
                broadcast_value = broadcast_data.value
                broadcast_str = ", ".join(broadcast_value)  # atau kamu sesuaikan tampilannya
                return Row(worker=f"Worker {x}", broadcast_data=broadcast_str)

            result = rdd.map(process_partition).collect()
            result_df = spark.createDataFrame(result)
            result_df.show(truncate=False)
            for r in result:
                print(r)
        print("=========================================")
        df.show(5)
        # Kamu bisa return df dan broadcast_data kalau perlu
        return df, broadcast_data

    except subprocess.CalledProcessError as e:
        print("Error saat menjalankan perintah HDFS:", e)
        raise
    except Exception as e:
        print("General error:", e)
        raise
