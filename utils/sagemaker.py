# SageMaker LLM Class

from transformers import AutoTokenizer
from botocore.config import Config
import boto3
import json
import time
import re
from queue import Queue
from threading import Thread
from concurrent.futures import ThreadPoolExecutor, as_completed

class SagemakerLLMWrapper():
    def __init__(self,
        endpoint_name: str = '',
        max_token_count: int = 512,
        max_attempts: int = 3,
        debug: bool = False,
        region: str = 'us-west-2'

    ):

        self.endpoint_name = endpoint_name
        self.max_token_count = max_token_count
        self.max_attempts = max_attempts
        self.debug = debug
        self.region = region
        config = Config(
            retries = {
                'max_attempts': 10,
                'mode': 'standard'
            }
        )
        self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
        self.sagemaker_runtime = boto3.client(service_name="sagemaker-runtime", config=config, region_name=self.region)
        
    def generate(self,prompt):
        if self.debug: 
            print('entered SagemakerLLMWrapper generate')
        
        prompt = re.sub(r'\s+', ' ', prompt).strip()
 
        messages = [
            {
                'role': 'system',
                'content': "You are an AI assistant that generates SQL queries from natural language and given schema information. Create accurate SQL queries based on the user's request and the provided table structures."
            },
            {
                'role': 'user',
                'content': f'''{prompt}'''
            }
        ]
        
        
        if self.debug:
            print(f'messages type: {type(messages)} - messages value: {messages}')
        # Use tokenizers chat template to format the incomming request
        adjusted_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if self.debug:
            print(f'adjusted_prompt: {adjusted_prompt}')
        
        # Add hyperparams and inputs into sagemaker call.
        input_data = {
            "inputs": adjusted_prompt,
            "parameters": {
                "max_new_tokens": self.max_token_count,
                "do_sample": False,
                "return_full_text": False,
                "stop": ["<|im_end|>"],
              }
        }
    
        # Convert input data to JSON string
        input_json = json.dumps(input_data)
        if self.debug:
            print(f'json input: {input_data}')
        try:
            # Call the SageMaker endpoint
            start_time = time.time()
            response = self.sagemaker_runtime.invoke_endpoint(
                EndpointName=self.endpoint_name,
                ContentType='application/json',
                Body=input_json
            )
            end_time = time.time()
            if self.debug: 
                print(f'sagemaker response: {response}')
            # Get the response body and decode it
            result = json.loads(response['Body'].read().decode())
            if result != None:
                text = f"<sql>{result[0]['generated_text']}</sql>"
            if self.debug: 
                print(f'text: {text}')
            usage = None
            latency = end_time - start_time
            return [text,usage,latency]
    
        except Exception as e:
            print(f"Error calling SageMaker endpoint: {str(e)}")
            return None
            

     # Threaded function for queue processing.
    def thread_request(self, q, results):
        while True:
            try:
                index, prompt = q.get(block=False)
                data = self.generate(prompt)
                results[index] = data
            except Queue.Empty:
                break
            except Exception as e:
                print(f'Error with prompt: {str(e)}')
                results[index] = str(e)
            finally:
                q.task_done()

    def generate_threaded(self, prompts, max_workers=15):
        results = [None] * len(prompts)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {executor.submit(self.generate, prompt): i for i, prompt in enumerate(prompts)}
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                except Exception as exc:
                    print(f'Generated an exception: {exc}')
                    results[index] = str(exc)
        
        return results