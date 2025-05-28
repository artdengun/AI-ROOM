import os
import subprocess
import time
from pyspark.sql import SparkSession

def create_spark_session():
    return SparkSession.builder \
        .appName("MyAwesomeSparkApp") \
        .master("local[*]") \
        .config("spark.driver.memory", "2g") \
        .config("spark.sql.shuffle.partitions", "4") \
        .config("spark.default.parallelism", "4") \
        .config("spark.executor.memory", "2g") \
        .config("spark.ui.showConsoleProgress", "true") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .getOrCreate()

HDFS_CMD = r"C:\hadoop\bin\hdfs.cmd"

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
    """Upload file ke HDFS jika belum ada dan baca pakai Spark."""

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
        df.show(5)
        return df

    except subprocess.CalledProcessError as e:
        print("Error saat menjalankan perintah HDFS:", e)
        raise
    except Exception as e:
        print("General error:", e)
        raise
