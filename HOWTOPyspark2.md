# Engineering-Notes
2020 Engineering Notes
# PySpark Part 2

[Apache Spark in Python: Beginner's Guide
](https://www.datacamp.com/community/tutorials/apache-spark-python)

### Python vs Scala
When you’re working with the DataFrame API, there isn’t really much of a difference between Python and Scala, but you do need to be wary of User Defined Functions (UDFs), which are less efficient than its Scala equivalents. That’s why you should **favor built-in expressions if you’re working with Python**. When you’re working with Python, also make sure **not to pass your data between DataFrame and RDD unnecessarily**, as the serialization and deserialization of the data transfer is particularly expensive.

thon is a good choice when you’re doing smaller ad hoc experiments, while Scala is great when you’re working on bigger projects in production. Remember that when a language is **statically typed**, every variable name is bound both to a type and an object. Scala is statically typed, Python is dynamically typed.

### RDD, Dataset, and DataFrame

**RDDs** have three main characteristics: they are compile-time type safe (they have a type!), they are lazy and they are based on the Scala collections API.
- it’s easy to build inefficient transformation chains, they are slow with non-JVM languages such as Python, they can not be optimized by Spark.

**DataFrames** are optimized: more intelligent decisions will be made when you’re transforming data and that also explains why they are faster than RDDs.

**Dataset** can take on two distinct characteristics: a strongly-typed API and an untyped API

since Python has no compile-time type-safety, only the untyped DataFrame API is available. Or, in other words, Spark DataSets are statically typed, while Python is a dynamically typed programming language. 

Use RDDs when you want to manipulate the data with functional programming constructs rather than domain specific expressions.
### Persist vs Broadcast Variables
Instead of creating a copy of the variable for each machine, you use broadcast variables to send some immutable state once to each worker. **Broadcast variables allow the programmer to keep a cached read-only variable in every machine.** In short, you use these variables when you want a local copy of a variable



# PySpark SQL
import pyspark as spark
from pyspark.sql import Row

sc = spark.sparkContext

# Load a text file and convert each line to a Row.
lines = sc.textFile("examples/src/main/resources/people.txt")
parts = lines.map(lambda l: l.split(","))
people = parts.map(lambda p: Row(name=p[0], age=int(p[1])))

# Infer the schema, and register the DataFrame as a table.
schemaPeople = spark.createDataFrame(people)
schemaPeople.createOrReplaceTempView("people")

# SQL can be run over DataFrames that have been registered as a table.
teenagers = spark.sql("SELECT name FROM people WHERE age >= 13 AND age <= 19")
#result of SQL query is a DataFrame

`class pyspark.sql.SparkSession(sparkContext, jsparkSession=None)`
*The entry point to programming Spark with the Dataset and DataFrame API.*
    
    `.createDataFrame(data, schema=None, samplingRatio=None, verifySchema=True)`
    
Creates a DataFrame from an RDD, a list or a pandas.DataFrame.

When schema is a list of column names, the type of each column will be inferred from data.

When schema is None, it will try to infer the schema (column names and types) from data, which should be an RDD of Row, or namedtuple, or dict.

`SparkSession.builder.enableHiveSupport().getOrCreate()`

 --------------------------------------------------------------- 

`class pyspark.sql.DataFrame(jdf, sql_ctx)`
*A DataFrame is equivalent to a relational table in Spark SQL, and can be created using various functions in SparkSession:*

    .coalesce(numPartitions)
    *Returns a new DataFrame that has exactly numPartitions partitions. Creates a narrow transformation that combines partitions*
    
    .createTempView(name)
    *Creates a local temporary view with this DataFrame. Can then be queried with SQL*
