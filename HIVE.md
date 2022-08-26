# Engineering-Notes
2020 Engineering Notes
# HIVE


__HIVE__ is a data warehousing software that enables SQL like queries to extract data from Apache Hadoop. [link](https://www.talend.com/resources/what-is-apache-hive/)


## How it Works.
It translates HiveQL into a Java MapReduce/Spark job. (Uses YARN) HIVE organizes the data into tables, then runs the job on a cluster.

Limitations of Hive:
• Hive is not designed for Online transaction processing (OLTP ), it is only used for the Online Analytical Processing.

• Hive supports overwriting or apprehending data, but not updates and deletes.

• In Hive, sub queries are not supported.

show databases;
use databasename;
#copy data to HDFS
hadoop dfs -copyFromLocal C://Desktop/data.txt hdfs:/ 
#create table in HIVE
create table training_data(txnno INT, txdate STRING, custno INT, category STRING) row format delimited fields terminated by ',' stored as textfile;
describe training_data;

#load data into HIVE table
load data inpath 'hdfs:/file.txt' OVERWRITE INTO TABLE training_data;

select count(*) from training_data
