package org.apache.spark.ml.made

import org.apache.log4j.{Level, Logger}
import breeze.linalg.{*, DenseVector}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.{DataFrame, SparkSession}

/**
 * Поиск коэффициентов линейной модели.
 */
object LrsSolve {
  /** Количество точек */
  val N = 100000
  /** Минимальная граница градиента для остановки вычисления */
  val epsilon = 1e-15
  /** Максимальное количество итераций */
  val maxIter = 10000
  /** Инициализация шага градиентного спуска. потом нормируется на длину вектора градиента. */
  val learningRate = 0.5

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
    val df = createData(spark, N, a, b)
    df.show(10)
    // создаем эстиматор и обучаем
    val estimator = new LinearRegression(epsilon, maxIter, learningRate).setInputCol("features").setLabelCol("y")
    val model = estimator.fit(df)

    println(f"fit complete: model.a=${model.a} model.b=${model.b}%.5f")
  }

  /** создание данных для тестирования модели */
  def createData(spark: SparkSession, n: Int, a: breeze.linalg.DenseVector[Double], b: Double): DataFrame = {
    import spark.implicits._
    val normal01 = breeze.stats.distributions.Gaussian(0, 1)
    val X = breeze.linalg.DenseMatrix.rand(n, a.length, normal01)

    val y = X * a + b
    val data = breeze.linalg.DenseMatrix.horzcat(X, y.asDenseMatrix.t)
    val df: DataFrame = data(*, ::).iterator.map(x => Tuple2(Vectors.dense(x(0), x(1), x(2)), x(3))).toSeq.toDF("features", "y")
    df
  }
}
