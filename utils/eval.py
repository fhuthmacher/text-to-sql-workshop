from utils.bedrock import BedrockLLMWrapper
from utils.sagemaker import SagemakerLLMWrapper
from utils.database import DatabaseUtil
from utils.util import Util
from utils.chroma import BaseRetrievalTask, ChromaDBRetrievalTask
import sqlparse
import sqlite3
import json
import time
import pandas as pd
from botocore.config import Config
import boto3

class AnswerTaskRunner:
    def __init__(self, eval_df: pd.DataFrame, 
                 model_id:str = 'anthropic.claude-3-sonnet-20240229-v1:0',
                 endpoint_name:str = '',
                 eval_model_id:str = 'anthropic.claude-3-sonnet-20240229-v1:0',
                 sql_dialect=None,
                 temperature: float = 0.0,
                 max_token_count: int = 2000,
                 max_attempts: int = 3, 
                 prompt_template: str = '',
                 prompt_eval_template: str = '',
                 datasource_url: str =["https://d3q8adh3y5sxpk.cloudfront.net/sql-workshop/data/redshift-sourcedb.sql"],
                 sql_database: str = "LOCAL",
                 region: str = "us-west-2",
                 retrieval_task: BaseRetrievalTask = None):
        self.eval_df = eval_df
        self.model_id = model_id
        self.endpoint_name = endpoint_name
        self.eval_model_id = eval_model_id
        self.sql_dialect = sql_dialect
        self.temperature = temperature
        self.max_token_count = max_token_count
        self.max_attempts = max_attempts
        self.prompt_template = prompt_template
        self.prompt_eval_template = prompt_eval_template
        self.datasource_url = datasource_url
        self.sql_database = sql_database
        self.region = region
        self.retrieval_task = retrieval_task
        if self.endpoint_name == '':
            self.llm = BedrockLLMWrapper(model_id=self.model_id, 
                                             max_token_count=self.max_token_count,
                                             temperature=self.temperature,
                                             region=self.region
                                             )
        else:
            self.llm = SagemakerLLMWrapper(endpoint_name=self.endpoint_name,
                                           max_token_count=self.max_token_count,
                                           debug=False,
                                           region=self.region
                                          )
        
        self.eval_llm = BedrockLLMWrapper(model_id=self.eval_model_id, 
                                             max_token_count=self.max_token_count,
                                             temperature=self.temperature,
                                             region=self.region
                                             )
        self.util = Util()
        self.DatabaseUtil = DatabaseUtil(datasource_url=datasource_url,
                                         sql_database= self.sql_database,
                                         region = self.region)


    def get_prompt(self, user_question, sql_database_schema):
        if self.retrieval_task:
            sql_examples = self.retrieval_task.retrieve(query_text=user_question, n_results=3)
            sql_examples_str = " ".join([example.document for example in sql_examples])
            prompt = self.prompt_template.format(
                    user_question=user_question,
                    sql_database_schema=sql_database_schema,
                    sql_dialect=self.sql_dialect,
                    sql_examples=sql_examples_str
                )
        else:
            
            prompt = self.prompt_template.format(
                        user_question=user_question,
                        sql_database_schema=sql_database_schema,
                        sql_dialect=self.sql_dialect
                    ) 
        return prompt


    def build_grader_prompt(self, 
                            question: str, 
                            sql_schema: str, 
                            sql_query:str, 
                            sql_query_run_error, 
                            sql_query_run_result:str,
                            groundtruth_sql_query:str,
                            ex_score:str,
                            em_score:str,
                            ves_score:str):
    
        prompt = self.prompt_eval_template.format(
                    question=question,
                    sql_schema=sql_schema,
                    sql_query= sql_query,
                    sql_query_run_error= sql_query_run_error,
                    sql_query_run_result= sql_query_run_result,
                    groundtruth_sql_query= groundtruth_sql_query,
                    ex_score=ex_score,
                    em_score=em_score,
                    ves_score=ves_score,
                ) 
        return prompt


    def execution_accuracy(self, generated_sql, labeled_sql):
        """
        Calculate Execution Accuracy (EX)
        
        Args:
        generated_sql (str): The SQL query generated by the model
        labeled_sql (str): The labeled (ground truth) SQL query
        
        Returns:
        float: 1.0 if the queries match, 0.0 otherwise
        """
        # Normalize and compare the SQL queries

        # remove ; and public. from generated_sql string prior to normalization
        generated_sql = generated_sql.replace(";", "").replace("public.", "")
        labeled_sql = labeled_sql.replace(";", "").replace("public.", "")

        gen_normalized = sqlparse.format(generated_sql, strip_comments=True, reindent=True)
        lab_normalized = sqlparse.format(labeled_sql, strip_comments=True, reindent=True)
        return 1.0 if gen_normalized == lab_normalized else 0.0

    def convert_to_str(self, result):
        if isinstance(result, list):
            return str(result)
        return result

    def exact_set_match_accuracy(self, generated_sql, labeled_sql):
        """
        Calculate Exact Set Match Accuracy (EM)
        
        Args:
        generated_sql (str): The SQL query generated by the model
        labeled_sql (str): The labeled (ground truth) SQL query
        db_connection: A database connection object
        
        Returns:
        float: 1.0 if the result sets match, 0.0 otherwise
        """
        try:
            # Execute both queries
            gen_result = self.DatabaseUtil.run_sql(generated_sql)
            # print(f'labeled_sql: {gen_result} - result: {gen_result}')
            lab_result = self.DatabaseUtil.run_sql(labeled_sql)
            # print(f'labeled_sql: {labeled_sql} - result: {lab_result}')
            gen_result = self.convert_to_str(gen_result)
            lab_result = self.convert_to_str(lab_result)

            # Compare the result sets
            return 1.0 if gen_result==lab_result else 0.0
        except Exception as e:
            return 0.0


    def valid_efficiency_score(self, generated_sql, labeled_sql):
        """
        Calculate Valid Efficiency Score (VES)
        
        Args:
        generated_sql (str): The SQL query generated by the model
        labeled_sql (str): The labeled (ground truth) SQL query
        db_connection: A database connection object
        
        Returns:
        float: The VES score
        """
        try:
            # Execute both queries and measure execution time
            gen_start = time.time()
            gen_result = self.DatabaseUtil.run_sql(generated_sql)
            gen_time = time.time() - gen_start
            # print(f'generated_sql_execution_time: {gen_time}')
            lab_start = time.time()
            lab_result = self.DatabaseUtil.run_sql(labeled_sql)
            lab_time = time.time() - lab_start
            # print(f'labeled_sql_execution_time: {lab_time}')
            
            gen_result = self.convert_to_str(gen_result)
            lab_result = self.convert_to_str(lab_result)

            # Check if results match
            if not gen_result==lab_result:
                return 0.0
            
            # Calculate VES
            ves = min(lab_time / gen_time, 1.0)
            return ves
        except Exception as e:
            print(f"Error executing SQL: {e}")
            return 0.0


    def run(self) -> pd.DataFrame:
        # Make a copy of the dataframe so we don't modify the original.
        df = pd.DataFrame(self.eval_df)
        results = []
        
        # Prepare prompts for all questions
        prompts = []
        for _, row in df.iterrows():
            question: str = row['Question']
            sql_database_schema: str = row['Context']
            model_prompt = self.get_prompt(question, sql_database_schema)
            prompts.append(model_prompt)
        
        # Generate SQL queries using threaded approach
        answers = self.llm.generate_threaded(prompts,max_workers=5)
        
        # bottleneck: from here on we are back to processing in sequence
        for index, (answer, row) in enumerate(zip(answers, df.iterrows())):
            _, row = row  # Unpack the row
            question: str = row['Question']
            sql_database_schema: str = row['Context']
            groundtruth_sql_query: str = row['Query']
            sql_query_run_error = None
            sql_query_run_result = None
            usage = 0
            latency = 0
            cost = 0
            ex_score = 0
            em_score = 0
            ves_score = 0

            if answer[1] is not None:
                cost = self.util.calculate_cost(answer[1], self.model_id)
                usage = json.dumps(answer[1])
            
            if answer[2] is not None:
                latency = answer[2]
            
            if answer and answer[0]:
                generated_sql_query = self.util.extract_with_regex(str(answer[0]), self.util.SQL_PATTERN)
                
                if generated_sql_query:
                    generated_sql_query = generated_sql_query.replace("\\", "")
                else:
                    # just return answer
                    generated_sql_query = answer[0]
            
                # Calculate eval metrics
                try:
                    sql_query_run_result = self.DatabaseUtil.run_sql(generated_sql_query)
                except Exception as e:
                    sql_query_run_error = e
                
                # print(f'generated_sql: {generated_sql_query} - query_result: {sql_query_run_result}')
                
                ex_score = self.execution_accuracy(generated_sql_query, groundtruth_sql_query)
                # print(f'ex_score: {ex_score}')
                em_score = self.exact_set_match_accuracy(generated_sql_query, groundtruth_sql_query)
                # print(f'em_score: {em_score}')
                ves_score = self.valid_efficiency_score(generated_sql_query, groundtruth_sql_query)
                # print(f'ves_score: {ves_score}')
            
            # Create eval prompt
            prompt = self.build_grader_prompt(question=question, 
                                            sql_schema=sql_database_schema, 
                                            sql_query=generated_sql_query, 
                                            sql_query_run_error=sql_query_run_error,
                                            sql_query_run_result=sql_query_run_result,
                                            groundtruth_sql_query=groundtruth_sql_query,
                                            ex_score=ex_score,
                                            em_score=em_score,
                                            ves_score=ves_score)
            
            # Parse eval results
            eval_result = self.eval_llm.generate(prompt)
            reasoning = self.util.extract_with_regex(str(eval_result[0]), self.util.REASONING_PATTERN)
            score = self.util.extract_with_regex(str(eval_result[0]), self.util.SCORE_PATTERN)
            
            # Create new record
            result = {
                'user_question': question,
                'groundtruth_query': groundtruth_sql_query,
                'generated_sql_query': generated_sql_query,
                'score': score,
                'reasoning': reasoning,
                'usage': usage,
                'latency': latency,
                'cost': cost,
                'ex_score': ex_score,
                'em_score': em_score,
                'ves_score': ves_score,
                'sql_query_run_error': sql_query_run_error,
                'sql_query_run_result': sql_query_run_result,
                'context': sql_database_schema,
            }
            results.append(result)
        
        new_dataframe = pd.DataFrame(results)
        return new_dataframe