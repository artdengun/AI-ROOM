1. Buat Folder 
```
mkdir flask_hive_api
cd flask_hive_api
```

2. Buat Environment
```
python -m venv venv
```
 * Cara Mengaktifkan Environment 
   // windows atau linux 
    ```
    source venv/bin/activate
    ```
    // windows 
    ```
    venv\Scripts\activate
    ```

3. installasi kebutuhan untuk library / dependencies
```
pip install flask
pip install pyhive[hive]
pip install thrift==0.13.0 
pip install thrift-sasl==0.4.3 
pip install pyspark
```

spark-class.cmd org.apache.spark.deploy.master.Master --host localhost
spark-class.cmd org.apache.spark.deploy.worker.Worker spark://localhost:7077 --cores 1 --memory 1g
spark-class.cmd org.apache.spark.deploy.worker.Worker spark://localhost:7077 --cores 1 --memory 1g
spark-class.cmd org.apache.spark.deploy.worker.Worker spark://localhost:7077 --cores 1 --memory 1g
