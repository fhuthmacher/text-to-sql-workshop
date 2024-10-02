# text-to-sql-workshop

This workshop (Level 200) has an expected duration of 2 hours and is ideal for Data Engineers, Data Analysts, Data Scientists, Cloud Architects, Application Developers and Data Managers. Basic familiarity with Amazon Bedrock is desired, but not required.

All AWS services run in the us-west-2 region and if run in your own AWS account occur charges. Therefore if you run this workshop in your own account, please follow the clean up steps at the end of this workshop. 

## Business Scenario
A lot of enterprises are still unable to fully capitalize on their treasure of data.

“Text-to-SQL” tries to bridge this gap, by using natural language processing (NLP) for accessing data. Instead of dealing with complex SQL queries, business users and customers can now ask questions related to data and insights in plain language. To do this, the user prompt is transformed into a structured representation, and from this representation, a SQL query is generated and run against a database.

The hardest components of creating an accurate SQL query out of natural language are the same ones we might have struggled with as newcomers to SQL. Concepts like identifying foreign key relationships, breaking down the question into smaller, nested queries, and properly joining tables, are among the hardest components of SQL query generation. According to researchers, over 50% of SQL generation tests fail on schema linking and joins alone.

On top of these core components of the query, each database engine has its own syntax that may warrant mastery of in order to write a valid query. Further, in many organizations, there are many overlapping data attributes - a value is aggregated in one table and not aggregated in another, for example - as well as abbreviated table and column names that require tribal knowledge to use correctly.

## Overview
This workshop is designed to provide a hands-on learning experience for those looking to
1) Learn how to implement a Text-to-SQL use case and evaluate different implementation techniques. 
2) Use GenAI with Amazon SageMaker, Amazon Bedrock, and Amazon Aurora.
3) Use different evaluation metrics (e.g. Execution Accuracy, Exact Set Match Accuracy, Valid Efficiency Score, LLM as a Judge)

## Workshop Content
This workshop is designed to be a progression of Text-to-SQL techniques, starting with zero-shot prompting, few-shot prompting, and then moving to fine tuning and evaluation.

Below is an outline of the workshop content:

Lab 1: zero-shot prompting for Text-to-SQL with two different LLMs

Lab 2: few-shot prompting for Text-to-SQL

Lab 3: fine tuning of Mistral 7b for Text-to-SQL

Lab 4: Text-to-SQL evaluation

## Getting started

### self-guided / personal AWS account

This workshop is presented as a series of **Python notebooks**, which you can run from the environment of your choice:

- For a fully-managed environment with rich AI/ML features, we'd recommend using [SageMaker Studio](https://aws.amazon.com/sagemaker/studio/). To get started quickly, you can refer to the [instructions for domain quick setup](https://docs.aws.amazon.com/sagemaker/latest/dg/onboard-quick-start.html).
- For a fully-managed but more basic experience, you could instead [create a SageMaker Notebook Instance](https://docs.aws.amazon.com/sagemaker/latest/dg/howitworks-create-ws.html).
- If you prefer to use your existing (local or other) notebook environment, make sure it has [credentials for calling AWS](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-configure.html).

## Authors

- Tanner McRae - *Initial work* - [github](https://github.com/tannermcrae)
- Felix Huthmacher  - *Initial work* - [github](https://github.com/fhuthmacher)
