# Engineering-Notes
2020 Engineering Notes
# Data Warehousing and Data Mining

<a href=https://panoply.io/data-warehouse-guide/data-warehousing-and-data-mining-101/>source</a>


**Data Warehouse** pulls in raw data from disparate sources. (could be production systems, logs, ...) Stores data in a clean, standardized form available for analysis. Generally a relational database that stores historic operational data from across the organization. Optimized to support complex multidimensional queries.

**Data mining** extracting value from data. General use cases are analysis and predictive modelling for future business endeavors.


## Three Data Mining Principles

1. Information discovered must be previously unknown.
2. Information must be valid. (Statistical testing)
3. Information must be actionable.

## Common Data Mining Analyses 

**Association Rules** Objects that satisfy condition X are likely to satisfy condition Y. Great for Market Basket Analysis (Group similar items), cross selling, catalogue design, store layout, financial forecasting, likelihood of illness.

**Sequential Pattern** Discovery of frequent subsequences in a collection of sequences, where order matters.  Web Traffic analysis, Market funnel analysis.

**Classification/Regression** Discovery of a function that maps objects into predicted values. Selective Marketing, Performance Prediction, Diagnosing Illness.


# ETL Process
<a href=https://panoply.io/data-warehouse-guide/3-ways-to-build-an-etl-process/>source</a>

**ETL** Loads

## What is Hashing in DBMS?
In DBMS, hashing is a technique to directly search the location of desired data on the disk without using index structure. 
### Use Hashing When:
- For a huge db, an index may not be fast enough
- When you don't want an index (??)

#### Indexing
- Addresses in mem are sorted by the key.
- Use for range retrieval od data
- Slower than hash for freq updates (since the whole index must be reorganized

#### Hashing
- Ideal for retrieval of individual records
- Fast updates for frequent update situations
- Hash file must be constantly managed


## Indexing in DBMS
### Clustered Index
Sorts the data rows in the table on key values. Can only be one Clustered index per table. (Usually the primary key)

### Non-Clustered Index
Stores data and index seperately in different files. The index is a b-tree that contains pointers to the location of each unit of data. A table can have many non-clustered indices (each will have its own index that references the table)


