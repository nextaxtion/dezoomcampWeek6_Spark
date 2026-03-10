from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import os
import subprocess

spark = SparkSession.builder.master("local[*]").appName("HW6").getOrCreate()

# Q1
print("Q1:", spark.version)

# Q2 - Repartition to 4, save, measure avg parquet file size
df = spark.read.parquet("yellow_tripdata_2025-11.parquet")
df.repartition(4).write.mode("overwrite").parquet("output/yellow_repartitioned")

result = subprocess.run(
    ["find", "output/yellow_repartitioned", "-name", "*.parquet", "-type", "f"],
    capture_output=True,
    text=True,
    check=True,
)
files = [f for f in result.stdout.strip().split("\n") if f]
sizes = [os.path.getsize(f) for f in files]
avg_mb = (sum(sizes) / len(sizes)) / (1024 * 1024)
print(f"Q2: Avg file size = {avg_mb:.1f} MB")

# Q3 - Trips on Nov 15
count = df.filter(F.to_date("tpep_pickup_datetime") == "2025-11-15").count()
print(f"Q3: Trips on Nov 15 = {count}")

# Q4 - Longest trip in hours
longest = (
    df.withColumn(
        "dur_h",
        (
            F.unix_timestamp("tpep_dropoff_datetime")
            - F.unix_timestamp("tpep_pickup_datetime")
        )
        / 3600,
    )
    .agg(F.max("dur_h"))
    .collect()[0][0]
)
print(f"Q4: Longest trip = {longest:.1f} hours")

# Q5
print("Q5: Spark UI port = 4040")

# Q6
zones = spark.read.csv("taxi_zone_lookup.csv", header=True, inferSchema=True)
(
    df.join(zones, df.PULocationID == zones.LocationID, "left")
    .groupBy("Zone")
    .count()
    .orderBy("count")
    .filter(
        F.col("Zone").isin(
            "Governor's Island/Ellis Island/Liberty Island",
            "Arden Heights",
            "Rikers Island",
            "Jamaica Bay",
        )
    )
    .show(truncate=False)
)

spark.stop()
