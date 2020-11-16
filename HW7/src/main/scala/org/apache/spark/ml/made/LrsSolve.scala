package org.apache.spark.ml.made

import org.apache.log4j.{Level, Logger}
import breeze.linalg.DenseVector
import org.apache.spark.sql.SparkSession

/**
 * Поиск коэффициентов линейной модели.
 */
object LrsSolve {
  /** Количество точек */
  val N = 100000
  /**
   * @param args без параметров.
   */
  def main(args: Array[String]): Unit = {
    // отключаем мусор в логах
    Logger.getLogger("org").setLevel(Level.OFF)
    // Создает сессию спарка
    val spark: SparkSession = SparkSession.builder()
      .master("local[12]")
      .appName("linear-regression-solve")
      .getOrCreate()

    // параметры модели
    val a = DenseVector(1.5, 0.3, -0.7)
    val b: Double = 2.0
    // создаем тестовые данные на основе параметров
    val df = LinearRegression.createTestData(spark, N, a, b)
    df.show(10)
    // создаем эстиматор и обучаем
    val estimator = new LinearRegression().setInputCol("features").setLabelCol("y")
    val model = estimator.fit(df)

    println(f"fit complete: model.a=${model.a} model.b=${model.b}%.5f")
  }
}
