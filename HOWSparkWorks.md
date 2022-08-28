# Engineering-Notes
2020 Engineering Notes
# How Spark Works


Any Spark application creates __RDDs__ out of some input, runs lazy __transformations__ of that RDD to some other form, and performs __actions__ to collect or store data.


## Key Terms


### __SparkContext__
is the foundation of the application. The entrygate of Spark functionality. It establishes a connection to the Spark Expecution environment. Used to create RDDs, accumulators and broadcast variables, access Spark services and run jobs.
MASTER of the Spark application.

- Gets current status of Spark application.
- Cancels the job
- Cancels the stage
- runs job asynchronously/synchronously
- accessing persistent RDD
- Unpersisting RDDs

### Apache Spark Shell
Application written in Scala that allows a Spark CLI

### Job vs Tasks vs Stage
A __task__ is a unit of work that is distributed across the partitions of data. Each partition has one task. A __job__ is a parallel computation consisiting of multiple tasks that are spawned by an action in Spark. A __Stage__ is a subset of tasks contained in a job. Each stage depends on each other.

### Apache Spark Driver
The main() function of the program runs in the __driver__, The driver runs the user code to create RDDs, perform Transformations and actions, and create SparkContext. The driver application splits the Spark application into tasks and schedules them among workers.


### Executing a Spark Job

- Using spark-submit, the user submits an application.
- In spark-submit, we invoke the main() method that the user specifies. It also launches the driver program.
- The driver program asks for the resources to the cluster manager that we need to launch executors.
- The cluster manager launches executors on behalf of the driver program.
- The driver process runs with the help of user application. Based on the actions and transformation on RDDs, the driver sends work to executors in the form of tasks.
- The executors process the task and the result sends back to the driver through the cluster manager.

<img src=https://data-flair.training/blogs/wp-content/uploads/sites/2/2017/08/Internals-of-job-execution-in-spark.jpg>


## In- Memory Computing (<a href=https://data-flair.training/blogs/spark-in-memory-computing/>link</a>)

RDDs are cached using *cache()* or *persist()*. When *cache()* is called, RDD is stored on RAM, with excess spilled to disk. This allows the RDD to be accessed faster. *persist()* stores on RAM, however more likely to spill.

## Spark RDD (<a href=https://data-flair.training/blogs/spark-rdd-tutorial/>link</a>)

Fundamental datastructure of Apache Spark, immutable collection of objects, logically partitioned across many servers.
__Resilient__ - fault tolerant thanks to the RDD lineage graph DAG,
__Distributed__ - Data sits on multiple nodes
__Dataset__-JSON,CSV,txt,database

Spark RDDs can be __cached__ and __manually partitioned__. Caching is useful when the dataset needs to be accessed several times. RDDs can __persist__ to indicate they will be reused in future operations. These persistent RDDs are stored in memory by default (but may spill to secondary if there isnt enough RAM) Nonpersistent will be thrown away after execution is complete.

Keeping RDDs in memory allows for faster iterative algorithms and interactive data mining. 

To achieve __fault-tolerance__, RDDs provide restructed form of shared memory.__Coarse-grained transformations__ means we can transform the whole dataset, but not an individual element of the dataset. Immutable in nature means changes to an RDD are permanent.

### RDDs vs DSM (Distrib. Shared Memory)

RDDs are immutable, use coarse-grained transformations, with recovery from the lineage graph, disk spill if low memory

DSMs mutable, use fine-grained transformations, utilized checkpointing for data recovery.

### RDD Operations

__Transformations__ on RDDs are functions that take an RDD as input and output one or more RDDs as output.(lazy execution) There are two kinds of transformations, narrow and wide.

__Narrow Transformations__ - Data is from a single partition only. The output RDD has partitions containing data from a single partition in the parent RDD. (MAP, FlatMap, MapPartition, Filter, Sample, Union)

__Wide Transformations__ - result of group by and reduce like functions. Data required or the output RDD's partitions might come from several partitions in the parent RDD. These transformations may depend on __shuffle__.

<img src=https://miro.medium.com/max/1400/0*-fmFL32Tne6JiFpz.png>


__Actions__ returns the final result of RDD computations. It triggers execution using lineage graph to load the original RDD, carry out int. steps, and return final results to Driver Program or write to file system.

Actions are opertations that produce non-RDD values. (First(), Take(), Reduce(), Collect(), count())


### Limitations of RDD
__No Built in Optimization Engine__ (other data structures can utilize catalyst opt. and Tungsten exec. engine)
__Don't Infer Schema__ (requires users specification of each dtype)
__JVM Objects__ Have JVM overhead, Garbage Collection and Java Serialization.

# PySpark

Spark Context allows user to handle the cluster via the Python API.

#INSIDE THE PYSPARK SHELL YOU DO NOT NEED TO INSTANTIATE SC
from pyspark import SparkContext

sc = SparkContext("local","Simple_App")

#Read in a text file to an RDD
RDD = sc.textFile("notes/data/changes.txt").cache() #store RDD in memory
#see the contents of an RDD (action)
RDD.Collect()

## Scaling Relational Databases
Hive, Pig and SparkSQL provide declarative querying mechanismes to Big Data stores.

### SparkSQL
- provides a __DataFrame API__ that can perform relational operations on external data or built in collections.
- Builtin optimizer that allows mixed data types
- Supports data formats, user defined functions and the metastore.
- allows __mixed relational and procedural operations__

### Resilient Distributed Datasets (RDDs)
__RDDS__ are a distributed memory abstraction allowing in-memory computation with fault-tolerance. You can parallelize computations and track the lineage of transformations.

RDDs are created by reading in data from files, databases or parallelizeing existing collections. __Transformations__ transform a RDD into different RDD. This is __lazily computed__ (executed only when an action is required)

__DataFrames__ are a distributed collection of rows with the same schema, a Spark Dataset organized into named columns. Equivalent to a table in a relational database.

 Data Frames keep track of their schema and support relational operations, can be manipulated with direct SQL queries 
 
 ### DataFrames are Performant
 <img src=https://miro.medium.com/max/700/1*Kolwjg356xdqPv-8JmQGRQ.png>
 
 ### Catalyst
 __Catalyst__ is SparkSQLs optimizer written in Scala. Takes a users defined query plan, then optimizes the plan, then converts the plan to machine runnable code.
 
 from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql import Row


sc=SparkContext() #instantiate SparkContext
raw_rdd = sc.textFile(data_file).cache() #load text file to RDD and cache 
sqlContext=SQLContext(sc)

#SPLIT TEXT RDD BY COMMA
csv_rdd = raw_rdd.map(lambda row: row.split(","))
print(csv_rdd.take(3))

#MAKING A PARSED RDD
parsed_rdd = csv_rdd.map(lambda r: Row(duration=int(r[0]),
                                      protocol = r[1],
                                      service = r[2],
                                      flag=r[3],
                                      label=r[-1]))

#CONSTRUCTING A DATAFRAME
df = sqlContext.createDataFrame(parsed_rdd)
display(df.head(5))
df.printSchema()


### SQL on Spark DataFrames



#A TEMPORARY TABLE FOR RUNNING SQL ON DATAFRAME IN THIS SESSION
#DOES NOT PERSIST
df.registerTempTable("connections")


#NORMAL DATAFRAME SYNTAX FOR AGGREGATION
display(df.groupBy("protocol").count().orderBy('count', ascending=False))

#SPARKSQL SYNTAX
protocols=sqlContext("""SELECT protocol, count(*) as freq
                        FROM connections
                        GROUP BY protocol
                        ORDER BY 2 DESC""")
display(protocols)

attack_protocol = sqlContext.sql("""
                           SELECT protocol_type, 
                           CASE label
                               WHEN 'normal.' THEN 'no attack'
                               ELSE 'attack'
                               END AS state,
                           COUNT(*) as freq
                           FROM connections
                           GROUP BY protocol_type, state
                           ORDER BY 3 DESC
                           """)
display(attack_protocol)


# Spark Documentation
[link](https://spark.apache.org/docs/latest/rdd-programming-guide.html#using-the-shell)

Before Spark 2.0 the main programming interface of Spark was the RDD. After 2.0 the __RDD was improved by Dataset__, which is strongly typed with rich optimizations under the hood. 

## Basics
__Dataset__ distributed collection of items. Can be called a DataFrame to be consistent with the Python/R data concept.

## RDD Programming
Paralleized collections are created with the *SparkContext* parallelize() method on an existing iterable in the driver program. This distributed dataset can be operated on in parallel.

Number of partitions for a parallelized dataset should be 2-4 per cpu in the cluster. this can be manually passed to the parallelize() method/

### External datasets
Data can be loaded from HDFS, Cassandra, HBase, S3 by modifying the URI passed when loading the dataset (local path ~ C://, hdfs://, s3a://

### Understanding Closures
How do we understand the scope and life cycle of variables and methods that are being split up and distributed to multiple nodes in a cluster?

SparkContext breaks RDD operations into tasks, each executed by an executor. Prior to execution Spark computes the task's __closure__ *the variables and methods that must be visible to the executor to perform the computation on the RDD.* Closure is serialized and sent to each executor.

__Accumulator__ used specifically to update a variable when execution is split across nodes in a cluster.

__Closures (loops and locally defined methods) should not be used to mutate a global state. Use an Accumulator if global aggregation is needed.__

To print elements of an RDD use the *collect()* method. Which brings the entire RDD to the driver node (potential memory issues) One may also use *take(n)* to get the first n rows of the dataframe.

## Transformations

__map(func)__ - Return a new RDD formed by passing each element of the source through function func.

__filter(func)__ - Returns a new RDD by selecting elements of the source on which func returns true.

__flatMap(func)__ - Similar to map, but each input item can be mapped to 0 or more output items. (func returns a sequence)

__sample(withReplacement, fraction, seed)__ - Sample a fraction of the data with or without replacement.

__union(otherDataset)__ - returns a new dataset that contains the union of elements in source and argument.

__intersection(otherDataset)__ - returns intersection of source and argument.

__distinct(numPartitions)__ - returns a new dataset with the distinct elements of source

__reducebyKey(func, numPartitions)__ - on a dataset of (K,V) pairs, returns a dataset of (K,U) pairs where values are aggregated using the reduce function func.

__aggregatebyKey(zeroValue)(seqOp, CombOp)__ - on a dataset of (K,V) pairs, returns (K,U) pairs

__sortByKey()__ - on a dataset (K, V) pairs where K implements ordering, returns the (K,V) dataset ordered by K

__join(otherDataset)__ - called on datasets (K, V) and (K, W) returns a dataset (K, (V,W))

__pipe(command)__ - pipe each partition of the RDD through a shell command


## Actions

__reduce(func)__ - Aggregate the elements of the dataset using a function func which takes two arguments and returns one. Function should be commutative and associative. 
(Commutative ~ 5+8=8+5) (Associative ~ (1+2)+3 = 1+(2+3) )

__collect()__ return all elements of the dataset as an array to the driver program. (CAREFUL)

__count()__ - returns the number of elements in the dataset

__first()__ - returns the first element of the dataset

__take(n)__ - returns the first n elements of the dataset

__takeSample(withReplacement, num)__ - return a randomized array of elements

__countByKey()__ - called on dataset (K, V) returns the hashmap (K, Int)

__foreach(func)__ - Run a function func on all elements of the dataset. (UNDERSTAND CLOSURES FIRST)


## Shuffle Operations

Certain operations trigger a Shuffle. Shuffle is a redistribution of data, so data is grouped differently across partitions. __Shuffle is complex and costly__ (involves copying data across executors and machines.

### Background

ex. __reduceByKey()__ attempts to reduce all values with the same key are combined to a single tuple. For this to work, all values for the same key must be seen by the executor to produce a correct result.

For this to work, Spark needs to read from all partitions to find all the values for all keys, and then transfer identical keys to unique executors. This is the __shuffle__.


### Performance Impact

__Shuffle__ is expensive since it involves disk I/O, data setialization, and network I/O.

Shuffle is split into map and reduce tasks. Results from map tasks are kept in memory till they can't fit. Then these are sorted based on target partition and written to a single file. Reduce tasks read the relevant sorted blocks.

Shuffle operations use a boat load of heap memory since they store and organize records before or after transferring them. __reduceByKey__ and __aggregateByKey__ organize on the map side, __ByKey__ operationsorganize on the reduce side.

Shuffle generates many intermediate files on disk that are preserved until the corresponding RDDs are trashed.


## RDD Persistent

*caching* a dataset in memory across operations is the main performance booster. When an RDD is persisted, each node stores any partition of it that it computes in memory and reuses them in other datasets. (If the node needs a partition for an op, it will take it and keep it)

__persist()__ or __cache()__ can be set at different memory levels.

to menually remove data you can use __RDD.unpersist()__.

## Shared Variables
Spark provides two limited types of *shared variables* __broadvast variables__ and __accumulators__. During a task, a function and variables are passed to each execution node. The operations performed on these variables are not propagated back to the driver node.

### Broadcast Variables
Allow programmer to keep a read-only variable cached on each machine. Explicitly creating broadcast variables is only useful when tasks across multiple stages need the same data.


broadcastVar = sc.broadcast([1,2,3])
broadcastVar.value

Once the broadcast variable is created it should be used instead of the value *v* in any functions run on the cluster. The object *v* should not be modified after it is broadcast to ensure all nodes get the same data.


### Accumulators
__Accumulators__ are variables that are only added to through an associative and commutative operation, and therefore can be supported in parallel. They can be used to implement counters or sums.

accum = sc.accumulator(0)
sc.parallelize([1,2,3,4]).foreach(lambda x: accum.add(x))
accum.value

An update to an accumulator may be performed more than once during a Transformation if the task or stage are re-executed (this cannot happen in an Action)


# Spark Mechanics



## Tuning 
[Tuning Spark](https://spark.apache.org/docs/latest/tuning.html)

Spark can be bottlenecked by any resource in the cluster: CPU, network bandwidth, or memory. Most often (if data fits in memory), its bandwidth.

### Data Serialization
Serialization is important for distributed applications. Significantly reduces the memory footprint of objects to be transferred or objects not in use. Also makes theJava garbage collection faster, since multiple objects are combinged to a single serialized file.

### Memory Tuning
3 things: amount of memory used by objects, cost of accessing objects, and garbage collection overhead. 

Java objects have a large overhead. (a lot of metadata) This overhead can be reduced with serialization and smarter use of data structures.

#### Memory Management in Spark
Memory usage is either execution or storage. __Execution memory__ is used for execution operations (shuffle, join, agg.) __Storage memory__ used for caching and propagating data across the cluster.

Execution and Storage share a unified region that adjusts to the amount of data cached by the app program. Execution can evict storage (cached data) if the total mem. usage passes a set threshold (R). R is a parameter that controls a minimum subregion of shared memory where cached data can never be evicted.

__spark.memory.fraction__ the fraction of the total memory for the shared memory space of execution and storage.
__spark.memory.StorageFraction__ the size of R as a fraction of the shared storage.

#### Determining Memory Consumption
Create an RDD, put it into cache, then look at the "Storage" section of the Spark UI.

### Garbage Collection Tuning
JVM garbage collection can become a slowdown when there is a large 'churn' in the RDDs stored by a program. It is not a problem when the program reads te RDD once and performs many operations on it.

__The cost of garbage collection is proportional to the number of Java objects.__





## RDD Programming Guide

### Initializing Spark
A Spark program must first create a `SparkContext` object, which tells Spark how to access a cluster.First build a `SparkConf` object with info about the app.

```conf = SparkConf().setAppName("Application Name").setMaster("IP Address")
sc = SparkContext(conf=conf)```

`appName` will show up on the Spark cluster UI. `master` is a Spark/YARN URL. In dev. dont hardcode `master`, instead launch the application with `spark-submit`.

## Using the Shell

In the __PySpark Shell__, a special interpreter aware SparkContext is already created for you, in variable `sc`. 

You can set the master the context connects to using the `--master` argument. You can add supplemental .py or .zip files to the runtime path by passing a comma seperated list to `--py-files`. there is also the `--packages` arg for Spark packagest.

Any Python dependencies a Spark package has must be manually installed with `pip`. Below you can run the import code in code.py.

```./bin/pyspark --master local[4] --py-files code.py```


## Resilient Distributed Datasets (RDDs)


To create an RDD, either *parallelize* an object in the driver program or referencing a dataset in HDFS/HBase.

#### Parallelized Collections
Created using the `parallelize` method on an iterable in the driver program. Elements are copied to form a distributed dataset.

```data =[1,2,3,4,5]
distData=sc.parallelize(data)```

`distData` can be operated on in parallel. ex. `distData.reduce(lambda a, b: a+b)` to add up elements in the list.

Number of paritions. Spark will run one task per partion, typically want 2-4 partitions per CPU in cluster. This can be set within the `parallelize` method.

#### External Datasets
RDDs cannbe collected from any storage source supported by Hadoop.

Text file RDDs can be created with sc's `textFile` method. Takes a URI for the file, and reads it as a collection of lines.

`distFile = sc.textFile("s3://data.txt")`

- If using a path on the local filesystem, the file must also be accessible at the same path on worker nodes. Either copy the file to all workers or use a network-mounted shared file system.

- All of Spark’s file-based input methods, including textFile, support running on directories, compressed files, and wildcards as well. For example, you can use textFile("/my/directory"), textFile("/my/directory/*.txt"), and textFile("/my/directory/*.gz").

- The textFile method also takes an optional second argument for controlling the number of partitions of the file. By default, Spark creates one partition for each block of the file (blocks being 128MB by default in HDFS), but you can also ask for a higher number of partitions by passing a larger value. Note that you cannot have fewer partitions than blocks

__Other data formats__ - 
`SparkContext.wholeTextFiles` reads a directory of small text files as (filename, content) pairs. (`textFile` returned one record per line in the file.

`RDD.saveasPickleFile` and `sc.pickleFile` support saving an RDD in pickled format
