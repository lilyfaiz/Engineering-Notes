# Engineering-Notes
2020 Engineering Notes

# Pyspark - Part 1

import pyspark
from pyspark import SparkContext
sc = SparkContext()
nums=sc.parallelize([1,2,3,4]) #RDD parallelized across cluster
nums.take(1)#get first row
nums.map(lambda x: x*x).collect() #map function to data


#Create a Spark DataFrame
import pyspark.sql as psql
sqlContext = psql.SQLContext(sc)

#create a DataFrame context
list_p = [('John',19),('Smith',29),('Adam',35),('Henry',50)]
rdd=sc.parallelize(list_p)
ppl = rdd.map(lambda x: psql.Row(name=x[0], age=x[1]))
DF_ppl = sqlContext.createDataFrame(ppl)

#to access the type of each feature
DF_ppl.printSchema()

### 1. BASIC OPERATIONS WITH PYSPARK

from pyspark import SparkFiles
url = "https://raw.githubusercontent.com/guru99-edu/R-Programming/master/adult_data.csv"
sc.addFile(url)
sqlContext = SQLContext(sc)

#read the csv file with InfetSchema=True
df = sqlContext.read.csv(SparkFiles.get("adult_data.csv"),header=True,inferSchema=True)
df.printSchema()

df.show(5)

#Recasting the columns to a different format
#use withColumn to apply a transformation to one column
from pyspark.sql.types import *

def convertColumn(df, names, newType):
    for name in names:
        df = df.withColumn(name,df[name].cast(newType))
    return df

Continuous_Features = ['age', 'fnlwgt','capital_gain', 'education_num', 'capital_loss', 'hours_week']
#convert type of above columns
df = convertColumn(df,Continuous_Features. FloatType())
df.printSchema()

### EDA

#Select and show the names of the features
df.select('age','fnlwgt').show(5)

#Count the number of occurences by group
df.groupBy('age').count().sort('count', ascending=True).show(5)

#get summary statistics withdescribe()
df.describe().show()
df.describe('capital_gain').show()

#drop column
df.drop('education_num')
#access the columns 
df.columns
#fillna
df.fillna(0)

#filter data 
df.filter(df.age>40).show(5)

#groupBy and Aggregation
df.groupBy("marital").agg({'capital_gain':'mean'}).show(5)

### Data Preprocessing

from pyspark.sql.function import *

#select the column ??WHY??
age_square = df.select(col("age")**2) #????????
#apply transformation and add it to the dataframe
df = df.withColumn('age_square',col("age")**2)

#change the order of columns with select
#cols = [colname1, colname2,...]
#df = df.selects(cols)

#remove entries with a given criteria
df = df.filter(df.native_country!="Holand-Netherlands")

### Data Processing Pipeline

from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler

#OHE encoding a column of strings
#index the string col to numeric
stringIndexer = StringIndexer(inputCol='workclass',outputCol='workclass_encoded')
model = stringIndexer.fit(df)
df = model.transform(df)

#OneHotEncode the numeric column
encoder = OneHotEncoder(dropLast=False, inputCol='workclass_encoded', outputCol='workclass_vec')
df = encoder.transform(df)
df.show(2)


#BUILD THE PIPELINE
from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoderEstimator

CATE_FEATURES = ['workclass', 'education', 'marital', 'occupation', 'relationship', 'race', 'sex', 'native_country']
Continuous_Features = ['age', 'fnlwgt','capital_gain', 'education_num', 'capital_loss', 'hours_week']
stages = [] #stages to be addded to the Pipeline

#loop to create a OHE encoder for each categorical variable
for categorical_col in CATE_FEATURES:
    stringIndexer = StringIndexer(inputCol=categoricalCol, outputCol=categoricalCol + "Index")
    encoder = OneHotEncoderEstimator(inputCols=[stringIndexer.getOutputCol()],
                                    ouputCols=[categoricalCol_+'classVec'])
    stages+=[stringIndexer, encoder]
    #This OHE for a column is one stage

#OHE the target_labels
label_stringIDX = StringIndexer(inputCol='label', outputCol='newlabel')
stages+=[label_stringIDX]   

#Combine the features into one matrix
assembler_inputs = [c + "classVec" for c in CATE_FEATURES] + CONTI_FEATURES
assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features") 
stages+=[assembler]

#Note that the stages variable is a list of stages (grouped in lists)

#Push to Pipeline
pipeline=Pipeline(stages=stages)
pipelineModel = pipeline.fit(df)
model = pipelineModel.transform(df)
model.take(1)

### Build the Classifier

from pyspark.mi.linalg import DenseVectore
input_data = model.rdd.map(lambda x: (x['newlabel'], DenseVector(x['features'])))

#create a DataFrame
df_train = sqlContect.createDataFrame(input_data,['label', 'features'])
df_train.show(5)

#train/test split
train, test = df_train.randomSplit([.8,.2],seed=1234)
train.groupBy('label').agg({'label':'count'}).show()

#Build Logistic Regressor
from pyspark.ml.classification import LogisticRegression
lr = LogisticRegression(labelCol='label',
                       featuresCol='features',
                       maxIter=20,
                       regParams=0.1)
lr = lr.fit(train)

#print coefficients
print("Coefficients: " + str(lr.coefficients))
print(str(lr.intercept))

### Evaluate the Model

preds = linearModel.transform(test_data)
preds.printSchema() #includes true label, feats, probs, and preds

preds = preds.select('label','prediction','probability')
#accuracy
preds.filter(preds.label==preds.prediction).count()/preds.count()

### HyperParameter Tuning

from pyspark.ml.tuning import ParamGridBuilder,CrossValidator
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Evaluate model
evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction")

#create a Parameter Grid
paramGrid = (ParamGridBuilder().addGrid(lr.regParam,[0.01, 0.5]).build())

#create a 5-fold CrossValidator
cv = CrossValidator(estimator=lr,
                   estimatorParamMaps=paramGrid,
                   evaluator=evaluator,
                   numFolds=5)

model = cv.fit(train_data)
preds = model.transform(test)
preds = preds.select('label','prediction','probability')
#accuracy
preds.filter(preds.label==preds.prediction).count()/preds.count()


#extract the recommended best Parameter
model.bestModel.extractParamMap()
