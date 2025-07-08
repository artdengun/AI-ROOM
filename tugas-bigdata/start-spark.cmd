@echo off
echo =============== Starting Spark Master ===============
start "" "%SPARK_HOME%\bin\spark-class.cmd" org.apache.spark.deploy.master.Master

timeout /t 3 >nul

echo =============== Starting Spark Worker ===============
start "" "%SPARK_HOME%\bin\spark-class.cmd" org.apache.spark.deploy.worker.Worker spark://172.20.10.2:7077

echo Spark Master & Worker started. Cek UI di:
echo http://localhost:8080 (Master)
echo http://localhost:8081 (Worker)
pause