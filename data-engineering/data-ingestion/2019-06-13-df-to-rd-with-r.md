---
slug: data-engineering/data-ingestion/loading-data-frame-to-relational-database-with-r
title: Loading Data Frame to Relational Database with R
tags:
  [
    Data Engineering,
    Data Ingestion,
    Database,
    ODBC,
    R,
    RODBC,
    Truncate and Load,
  ]
---

Once you create a data frame with R, you may need to load it to a relational database for data persistence. <!--truncate-->You might have a data transformation batch job written in R and want to load database in a certain frequency.

Here, I created a function to load data into a relational database. I opted to use RODBC because it is probably the easiest way to interact with databases with R. As long as you install and configure ODBC for whatever database you are using, this should work.

RODBC has many functions that does database operations for you. The insertion is taken care of by the sqlSave method. RODBC has fantastic documentation so that you can check what options are available.

Function Parameters

The function below takes 6 parameters.

dsn: data source name that you configured for ODBC.

user: database user name.

pw: database password.

tableName: Name of the table including the schema

df: R data frame

columnTypes: List of SQL column types.

Usage

Call the function like this:

```r
dsn = "my-database-dsn"
dbUser = "user"
dbPass = "password"
tableName = "datamart.customer_dim"
df = dataFrameCreated
columnTypes <- list(account_name="varchar(255)", customer_id="int", last_updated="date")

dbLoader(dsn, dbUser, dbPass, "usermanaged.session_budget_2018", dfTransformed, columnTypes)
```

Function: dbLoader

I could probably make drop table SQL statement as a parameter so that it can be used for any database. In this function, it is hard coded. The statement should work for most databases. Most of databases have the same drop statement syntax.

The db load pattern is the classic truncate and load. You can customise it to make it to upsert.

```r
# Arguments - dsn: data source name for ODBC, user: db username, pw: password, df: Input DataFrame,
#             tableName: Output table name, columnTypes: list of column definition

dbLoader <- function(dsn, user, pw, tableName, df, columnTypes) {

  channel <- odbcConnect(dsn, uid=user, pwd=pw)
  print('Database connection initiated.')
  tableName <- tableName
  dropSQL <- sprintf('Drop Table If Exists %s;', tableName)

  # Drop Table If exists
  sqlQuery(channel, dropSQL)
  print(sprintf('Executed %s', dropSQL))

  # Insert Data
  sqlSave(channel, df, tablename=tableName, fast=T, colnames=F, rownames = F, varTypes=columnTypes)
  print(sprintf('DataFrame has been inserted into %s', tableName ))

  # Close DB connection
  close(channel)
  print('DB connection closed.')
}
```
