# Utility class to get cost and visualizations 
import re
import typing as t
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class Util():
    def __init__(self,
        debug: bool = False

    ):
        self.debug = debug
    
    SCORE_PATTERN = r'<score>(.*?)</score>'
    REASONING_PATTERN = r'<thinking>(.*?)</thinking>'
    SQL_PATTERN = r'<[sS][qQ][lL]>(.*?)</[sS][qQ][lL]>'
    DIFFICULTY_PATTERN = r'<difficulty>(.*?)</difficulty>'
    USER_QUESTION_PATTERN = r'<user_question>(.*?)</user_question>'
    SQL_DATABASE_SCHEMA_PATTERN = r'<sql_database_schema>(.*?)</sql_database_schema>'
    SQL_DIALECT_PATTERN = r'<sql_dialect>(.*?)</sql_dialect>'


    def compare_results(self, answer_results1, answer_results2, metrics):


        # # Function to convert 'score' column
        def convert_score(df):
            # df['score'] = df['score'].map({'correct': 1, 'incorrect': 0})
            df['score'] = pd.to_numeric(df['score'], errors='coerce').fillna(0).astype(int)
            return df

        # Apply the conversion to both dataframes
        answer_results1 = convert_score(answer_results1)
        answer_results2 = convert_score(answer_results2)

        # Calculate the average values for each metric
        # metrics = ['score', 'latency' ,'cost', 'ex_score', 'em_score','ves_score']
        
        avg_results1 = [answer_results1[metric].mean() for metric in metrics]
        avg_results2 = [answer_results2[metric].mean() for metric in metrics]

        # Calculate percentage change, handling divide-by-zero and infinite cases
        def safe_percent_change(a, b):
            if pd.isna(a) or pd.isna(b):
                return 0
            if a == 0 and b == 0:
                return 0
            elif a == 0:
                return 100  # Arbitrarily set to 100% increase if original value was 0
            else:
                change = (b - a) / a * 100
                return change if np.isfinite(change) else 0

        percent_change = [safe_percent_change(a, b) for a, b in zip(avg_results1, avg_results2)]

        # Set up the bar chart
        x = np.arange(len(metrics))
        width = 0.5

        fig, ax = plt.subplots(figsize=(12, 6))

        # Create the bars
        bars = ax.bar(x, percent_change, width)

        # Customize the chart
        ax.set_ylabel('Percentage Change (%)')
        ax.set_title('Percentage Change in Metrics (Results 2 vs Results 1)')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)

        # Add a horizontal line at y=0
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        # Add value labels on top of each bar
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.2f}%',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3 if height >= 0 else -3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom' if height >= 0 else 'top')

        autolabel(bars)

        # Color the bars based on positive (green) or negative (red) change
        # For latency & cost, reverse the color logic
        for bar, change, metric in zip(bars, percent_change, metrics):
            if metric == 'latency' or metric == 'cost':
                bar.set_color('green' if change <= 0 else 'red')
            else:
                bar.set_color('green' if change >= 0 else 'red')
            

        # Adjust layout and display the chart
        fig.tight_layout()
        plt.show()

    def visualize_distribution(self, df, key, key_labels=None):
        # Check if 'key' column exists in the DataFrame
        if key not in df.columns:
            raise ValueError(f"The DataFrame does not contain a '{key}' column.")
        
        # Count the frequency of each value
        value_counts = df[key].value_counts().sort_index()
        
        # If key_labels are provided, use them for x-axis
        if key_labels:
            x = range(len(key_labels))
            tick_labels = key_labels
        else:
            x = value_counts.index
            tick_labels = x
        
        # Create a bar chart
        plt.figure(figsize=(10, 6))
        bars = plt.bar(x, value_counts.values)
        
        # Customize the chart
        plt.title(f'Distribution of {key}')
        plt.xlabel(f'{key}')
        plt.ylabel('Frequency')
        plt.xticks(x, tick_labels, rotation=45, ha='right')
        
        # Add value labels on top of each bar
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                     f'{height}', ha='center', va='bottom')
        
        # Display the chart
        plt.tight_layout()
        plt.show()

    # Strip out the portion of the response with regex.
    def extract_with_regex(self, response, regex):
        matches = re.search(regex, response, re.DOTALL)
        # Extract the matched content, if any
        return matches.group(1).strip() if matches else None

    def calculate_cost(self, usage, model_id):
        '''
        Takes the usage tokens returned by Bedrock in input and output, and coverts to cost in dollars.
        '''
        
        input_token_haiku = 0.25/1000000
        output_token_haiku = 1.25/1000000
        input_token_sonnet = 3.00/1000000
        output_token_sonnet = 15.00/1000000
        input_token_opus = 15.00/1000000
        output_token_opus = 75.00/1000000
        
        input_token_titan_embeddingv1 = 0.1/1000000
        input_token_titan_embeddingv2 = 0.02/1000000
        input_token_titan_embeddingmultimodal = 0.8/1000000
        input_token_titan_premier = 0.5/1000000
        output_token_titan_premier = 1.5/1000000
        input_token_titan_lite = 0.15/1000000
        output_token_titan_lite = 0.2/1000000
        input_token_titan_express = 0.2/1000000
        output_token_titan_express = 0.6/1000000
       
        input_token_cohere_command = 0.15/1000000
        output_token_cohere_command = 2/1000000
        input_token_cohere_commandlight = 0.3/1000000
        output_token_cohere_commandlight = 0.6/1000000
        input_token_cohere_commandrplus = 3/1000000
        output_token_cohere_commandrplus = 15/1000000
        input_token_cohere_commandr = 5/1000000
        output_token_cohere_commandr = 1.5/1000000
        input_token_cohere_embedenglish = 0.1/1000000
        input_token_cohere_embedmultilang = 0.1/1000000

        input_token_llama3_8b = 0.4/1000000
        output_token_llama3_8b = 0.6/1000000
        input_token_llama3_70b = 2.6/1000000
        output_token_llama3_70b = 3.5/1000000

        input_token_mistral_8b = 0.15/1000000
        output_token_mistral_8b = 0.2/1000000
        input_token_mistral_large = 4/1000000
        output_token_mistral_large = 12/1000000

        cost = 0

        if 'haiku' in model_id:
            cost+= usage['inputTokens']*input_token_haiku
            cost+= usage['outputTokens']*output_token_haiku
        if 'sonnet' in model_id:
            cost+= usage['inputTokens']*input_token_sonnet
            cost+= usage['outputTokens']*output_token_sonnet
        if 'opus' in model_id:
            cost+= usage['inputTokens']*input_token_opus
            cost+= usage['outputTokens']*output_token_opus
        if 'amazon.titan-embed-text-v1' in model_id:
            cost+= usage['inputTokens']*input_token_titan_embeddingv1
        if 'amazon.titan-embed-text-v2' in model_id:
            cost+= usage['inputTokens']*input_token_titan_embeddingv2
        if 'cohere.embed-multilingual' in model_id:
            cost+= usage['inputTokens']*input_token_cohere_embedmultilang
        if 'cohere.embed-english' in model_id:
            cost+= usage['inputTokens']*input_token_cohere_embedenglish 
        if 'meta.llama3-8b-instruct' in model_id:
            cost+= usage['inputTokens']*input_token_llama3_8b
            cost+= usage['outputTokens']*output_token_llama3_8b
        if 'meta.llama3-70b-instruct' in model_id:
            cost+= usage['inputTokens']*input_token_llama3_70b
            cost+= usage['outputTokens']*output_token_llama3_70b
        if 'cohere.command-text' in model_id:
            cost+= usage['inputTokens']*input_token_cohere_command
            cost+= usage['outputTokens']*output_token_cohere_command
        if 'cohere.command-light-text' in model_id:
            cost+= usage['inputTokens']*input_token_cohere_commandlight
            cost+= usage['outputTokens']*output_token_cohere_commandlight
        if 'cohere.command-r-plus' in model_id:
            cost+= usage['inputTokens']*input_token_cohere_commandrplus
            cost+= usage['outputTokens']*output_token_cohere_commandrplus
        if 'cohere.command-r' in model_id:
            cost+= usage['inputTokens']*input_token_cohere_commandr
            cost+= usage['outputTokens']*output_token_cohere_commandr
        if 'amazon.titan-text-express' in model_id:
            cost+= usage['inputTokens']*input_token_titan_express
            cost+= usage['outputTokens']*output_token_titan_express
        if 'amazon.titan-text-lite' in model_id:
            cost+= usage['inputTokens']*input_token_titan_lite
            cost+= usage['outputTokens']*output_token_titan_lite
        if 'amazon.titan-text-premier' in model_id:
            cost+= usage['inputTokens']*input_token_titan_premier
            cost+= usage['outputTokens']*output_token_titan_premier
        if 'mistral.mixtral-8x7b-instruct-v0:1' in model_id:
            cost+= usage['inputTokens']*input_token_mistral_8b
            cost+= usage['outputTokens']*output_token_mistral_8b

        return cost