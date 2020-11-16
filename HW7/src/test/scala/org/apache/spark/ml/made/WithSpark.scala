package org.apache.spark.ml.made

import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.{SQLContext, SparkSession}

trait WithSpark {
  lazy val spark: SparkSession = WithSpark._spark
  lazy val sqlc: SQLContext = WithSpark._sqlc
}

object WithSpark {
  Logger.getLogger("org").setLevel(Level.OFF)
  lazy private val _spark: SparkSession = SparkSession.builder()
    .master("local[12]")
    .appName("linear-regression-test")
    .getOrCreate()

  lazy private val _sqlc: SQLContext = _spark.sqlContext
}