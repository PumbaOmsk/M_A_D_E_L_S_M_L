package org.apache.spark.ml.made

import breeze.linalg.{*, DenseMatrix}
import com.google.common.io.Files
import org.scalatest._
import flatspec._
import matchers._
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.DataFrame

/**
 * Тест для {@link LinearRegression}
 */
class LinearRegressionTest extends AnyFlatSpec with should.Matchers with WithSpark {
  /** точность сравнения чисел с плавающей точкой */
  private val accuracy = 1e-5
  /**
   * Минимальная граница градиента для остановки вычисления.
   */
  private val epsilon = 1e-15

  /** Максимальное количество итераций */
  private val maxIter = 10000
  /** Инициализация шага градиентного спуска. потом нормируется на длину вектора градиента. */
  private val learningRate = 0.5

  /** вектор коэффициентов */
  private val aV: Vector = Vectors.dense(1.0, 2.0, 3.0)
  /** свободный член */
  private val bV: Double = 4
  /** размер тестовых данных */
  private val nTest = 1000

  /** простые тестовые данные для проверки модели */
  lazy val testVectors: Seq[Vector] = LinearRegressionTest._testVectors
  lazy val testData: DataFrame = LinearRegressionTest._testData

  "Model" should "calc eq" in {
    val model = new LinearRegressionModel(aV, bV)
      .setInputCol("features")
      .setOutputCol("features")

    validateModel(model, testData)
  }

  "Model" should "calc eq with Breeze" in {
    val model = new LinearRegressionModel(aV, bV)
      .setInputCol("features")
      .setOutputCol("features")

    val m = DenseMatrix(
    breeze.linalg.DenseVector[Double](1, 2, 3),
    breeze.linalg.DenseVector[Double](2, 4, 6),
    breeze.linalg.DenseVector[Double](3, 6, 9))

    val vectors: Array[Double] = model.calcMatrix(m).toArray
    vectors.length should be(testVectors.length)
    vectors(0) should be(18.0 +- accuracy)
    vectors(1) should be(32.0 +- accuracy)
    vectors(2) should be(46.0 +- accuracy)
  }

  "Estimator" should "solve eq" in {
    import LinearRegressionTest.sqlc.implicits._
    val normal01 = breeze.stats.distributions.Gaussian(0, 1)
    val X = DenseMatrix.rand(nTest, 3, normal01)
    val data = X(*, ::).iterator.map(x => Tuple1(Vectors.fromBreeze(x).toDense)).toSeq.toDF("features")

    val estimator = new LinearRegression(epsilon, maxIter, learningRate)
      .setInputCol("features").setOutputCol("features").setLabelCol("y")

    val multUdf = data.sqlContext.udf.register(estimator.uid + "_transformUdf", (x: Vector) => (x.asBreeze dot aV.asBreeze) + bV)
    val frame: DataFrame = data.withColumn("y", multUdf(data("features")))

    val model = estimator.fit(frame)

    checkModel(model)
  }

  def checkModel(model: LinearRegressionModel): Unit = {
    model.a(0) should be(aV(0) +- accuracy)
    model.a(1) should be(aV(1) +- accuracy)
    model.a(2) should be(aV(2) +- accuracy)
    model.b should be(bV +- accuracy)
  }

  "Estimator" should "produce functional model" in {
    val estimator: LinearRegression = new LinearRegression(epsilon, maxIter, learningRate)
      .setInputCol("features")
      .setOutputCol("features")

    val data = LinearRegressionTest.createTestData(nTest, aV, bV)
    val model = estimator.fit(data)
    checkModel(model)

  }

  "Estimator" should "work after re-read" in {
    val data = LinearRegressionTest.createTestData(nTest, aV, bV)

    val pipeline = new Pipeline().setStages(Array(
      new LinearRegression(epsilon, maxIter, learningRate)
        .setInputCol("features")
        .setLabelCol("y")
        .setOutputCol("features")
    ))

    val tmpFolder: String = Files.createTempDir().getAbsolutePath
    pipeline.write.overwrite().save(tmpFolder)

    val model = Pipeline.load(tmpFolder).fit(data).stages(0).asInstanceOf[LinearRegressionModel]

    validateModel(model, testData)
  }

  "Estimator" should "work after fit and re-read" in {
    val data = LinearRegressionTest.createTestData(nTest, aV, bV)

    val pipeline = new Pipeline().setStages(Array(
      new LinearRegression(epsilon, maxIter, learningRate)
        .setInputCol("features")
        .setLabelCol("y")
        .setOutputCol("features")
    ))
    val model = pipeline.fit(data)
    val tmpFolder = Files.createTempDir()
    model.write.overwrite().save(tmpFolder.getAbsolutePath)
    val reRead = PipelineModel.load(tmpFolder.getAbsolutePath)

    validateModel(model.stages(0).asInstanceOf[LinearRegressionModel], reRead.transform(testData))
  }

  private def validateModel(model: LinearRegressionModel, data: DataFrame) = {
    val vectors: Array[Double] = model.transform(testData).collect().map(_.getAs[Double](0))
    vectors.length should be(testVectors.length)
    vectors(0) should be(18.0 +- accuracy)
    vectors(1) should be(32.0 +- accuracy)
    vectors(2) should be(46.0 +- accuracy)
  }
}

object LinearRegressionTest extends WithSpark {
  lazy val _testVectors = Seq(
    Vectors.dense(1, 2, 3),
    Vectors.dense(2, 4, 6),
    Vectors.dense(3, 6, 9)
  )
  lazy val _testData: DataFrame = {
    import sqlc.implicits._
    _testVectors.map(x => Tuple1(x)).toDF("features")
  }

  /** создание данных для тестирования модели */
  def createTestData(n: Int, a: Vector, b: Double): DataFrame = {
    import sqlc.implicits._
    val aBr = a.asBreeze
    val normal01 = breeze.stats.distributions.Gaussian(0, 1)
    val X = breeze.linalg.DenseMatrix.rand(n, aBr.length, normal01)

    val y = X * aBr + b
    val data = breeze.linalg.DenseMatrix.horzcat(X, y.asDenseMatrix.t)
    val df: DataFrame = data(*, ::).iterator.map(x => Tuple2(Vectors.dense(x(0), x(1), x(2)), x(3))).toSeq.toDF("features", "y")
    df
  }
}