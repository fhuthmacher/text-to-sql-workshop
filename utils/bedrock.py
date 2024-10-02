# Bedrock LLM Class
from botocore.config import Config
import boto3
from queue import Queue
from threading import Thread
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

class BedrockLLMWrapper():
    def __init__(self,
        model_id: str = 'anthropic.claude-3-haiku-20240307-v1:0', #'anthropic.claude-3-sonnet-20240229-v1:0',
        top_k: int = 5,
        top_p: int = 0.7,
        temperature: float = 0.0,
        max_token_count: int = 4000,
        max_attempts: int = 3,
        debug: bool = False,
        region: str = 'us-west-2'

    ):

        self.model_id = model_id
        self.top_k = top_k
        self.top_p = top_p
        self.temperature = temperature
        self.max_token_count = max_token_count
        self.max_attempts = max_attempts
        self.debug = debug
        config = Config(
            retries = {
                'max_attempts': 10,
                'mode': 'standard'
            }
        )
        self.region=region

        self.bedrock_runtime = boto3.client(service_name="bedrock-runtime", config=config, region_name=self.region)
        
    def generate(self,prompt):
        if self.debug: 
            print('entered BedrockLLMWrapper generate')
        attempt = 1

        message = {
            "role": "user",
            "content": [{"text": prompt}]
        }
        messages = []
        messages.append(message)
        
        # model specific inference parameters to use.
        if "anthropic" in self.model_id.lower():
            # system_prompts = [{"text": "You are a helpful AI Assistant."}]
            system_prompts = []
            # Base inference parameters to use.
            inference_config = {
                                "temperature": self.temperature, 
                                "maxTokens": self.max_token_count,
                                "stopSequences": ["\n\nHuman:"],
                                "topP": self.top_p,
                            }
            additional_model_fields = {"top_k": self.top_k}
        else:
            system_prompts = []
            # Base inference parameters to use.
            inference_config = {
                                "temperature": self.temperature, 
                                "maxTokens": self.max_token_count,
                            }
            additional_model_fields = {"top_k": self.top_k}

        if self.debug: 
            print("Sending:\nSystem:\n",system,"\nMessages:\n",str(messages))

        while True:
            try:

                # Send the message.
                response = self.bedrock_runtime.converse(
                    modelId=self.model_id,
                    messages=messages,
                    system=system_prompts,
                    inferenceConfig=inference_config,
                    additionalModelRequestFields=additional_model_fields
                )

                # Log token usage.
                text = response['output'].get('message').get('content')[0].get('text')
                usage = response['usage']
                latency = response['metrics'].get('latencyMs')

                if self.debug: 
                    print(f'text: {text} ; and token usage: {usage} ; and query_time: {latency}')    
                
                break
               
            except Exception as e:
                print("Error with calling Bedrock: "+str(e))
                attempt+=1
                if attempt>self.max_attempts:
                    print("Max attempts reached!")
                    result_text = str(e)
                    break
                else:#retry in 10 seconds
                    print("retry")
                    time.sleep(10)

        # return result_text
        return [text,usage,latency]

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