package org.apache.spark.ml.made

import breeze.linalg.{*, DenseMatrix, DenseVector}
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
  val epsilon = 0.0000001
  /** простые тестовые данные для проверки модели */
  lazy val testVectors: Seq[Vector] = LinearRegressionTest._testVectors
  lazy val testData: DataFrame = LinearRegressionTest._testData

  "Model" should "calc eq" in {
    val model = new LinearRegressionModel(Vectors.dense(1, 2, 3).toDense, 4.0)
      .setInputCol("features")
      .setOutputCol("features")

    validateModel(model, testData)
  }

  "Estimator" should "solve eq" in {
    import LinearRegressionTest.sqlc.implicits._
    val normal01 = breeze.stats.distributions.Gaussian(0, 1)
    val X = DenseMatrix.rand(1000, 3, normal01)
    val data = X(*, ::).iterator.map(x => Tuple1(Vectors.fromBreeze(x).toDense)).toSeq.toDF("features")

    val estimator = new LinearRegression().setInputCol("features").setOutputCol("features").setLabelCol("y")

    val a = Vectors.dense(1, 2, 3).toDense
    val b: Double = 4
    val multUdf = data.sqlContext.udf.register(estimator.uid + "_transformUdf", (x: Vector) => (x.asBreeze dot a.asBreeze) + b)
    val frame: DataFrame = data.withColumn("y", multUdf(data("features")))

    val model = estimator.fit(frame)

    model.a(0) should be(a(0) +- epsilon)
    model.a(1) should be(a(1) +- epsilon)
    model.a(2) should be(a(2) +- epsilon)
    model.b should be(b +- epsilon)
  }

  "Estimator" should "produce functional model" in {
    val estimator: LinearRegression = new LinearRegression()
      .setInputCol("features")
      .setOutputCol("features")

    val data = LinearRegression.createTestData(LinearRegressionTest.spark, 1000, DenseVector(1, 2, 3), 4)
    val model = estimator.fit(data)

    validateModel(model, testData)
  }

  "Estimator" should "work after re-read" in {
    val data = LinearRegression.createTestData(LinearRegressionTest.spark, 1000, DenseVector(1, 2, 3), 4)

    val pipeline = new Pipeline().setStages(Array(
      new LinearRegression()
        .setInputCol("features")
        .setLabelCol("y")
        .setOutputCol("features")
    ))

    val tmpFolder = Files.createTempDir()
    pipeline.write.overwrite().save(tmpFolder.getAbsolutePath)
    val model = Pipeline.load(tmpFolder.getAbsolutePath).fit(data).stages(0).asInstanceOf[LinearRegressionModel]

    validateModel(model, testData)
  }

  "Estimator" should "work after fit and re-read" in {
    val data = LinearRegression.createTestData(LinearRegressionTest.spark, 1000, DenseVector(1, 2, 3), 4)

    val pipeline = new Pipeline().setStages(Array(
      new LinearRegression()
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
    vectors(0) should be(18.0 +- epsilon)
    vectors(1) should be(32.0 +- epsilon)
    vectors(2) should be(46.0 +- epsilon)
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
}