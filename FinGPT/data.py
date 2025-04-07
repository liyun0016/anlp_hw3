import os
import re
import csv
import math
import time
import json
import finnhub
from tqdm import tqdm
import pandas as pd
import yfinance as yf
from datetime import datetime
from collections import defaultdict
import datasets
from datasets import Dataset
import requests
from indices import *
from prompt import get_all_prompts
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from utils import summarize_financials
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

finnhub_client = finnhub.Client(api_key="cvf04q9r01qjugsfbuu0cvf04q9r01qjugsfbuug")

model_id = "mistralai/Mistral-7B-Instruct-v0.2"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",  
    torch_dtype=torch.float16 
)

text_generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    do_sample=False,
    temperature=0.7
)

# ----------------------------------------------------------------------------------- #
# ---------------------------- RAW FINANCIAL ACQUISITION ---------------------------- #
# ----------------------------------------------------------------------------------- #

def bin_mapping(ret):
    up_down = 'U' if ret >= 0 else 'D'
    integer = math.ceil(abs(100 * ret))
    return up_down + (str(integer) if integer <= 5 else '5+')

def get_returns(stock_symbol, start_date, end_date):
    stock_data = yf.download(stock_symbol, start=start_date, end=end_date, group_by='ticker')
    if isinstance(stock_data.columns, pd.MultiIndex):
        available_prices = stock_data.columns.get_level_values(1).unique()
        price_field = 'Adj Close' if 'Adj Close' in available_prices else 'Close'
        close_series = stock_data.loc[:, (stock_symbol, price_field)]
    else:
        price_field = 'Adj Close' if 'Adj Close' in stock_data.columns else 'Close'
        close_series = stock_data[price_field]
    weekly_data = close_series.resample('W').ffill()
    weekly_returns = weekly_data.pct_change()[1:]
    weekly_start_prices = weekly_data[:-1]
    weekly_end_prices = weekly_data[1:]
    weekly_data = pd.DataFrame({
        'Start Date': weekly_start_prices.index,
        'Start Price': weekly_start_prices.values,
        'End Date': weekly_end_prices.index,
        'End Price': weekly_end_prices.values,
        'Weekly Returns': weekly_returns.values
    })
    weekly_data['Bin Label'] = weekly_data['Weekly Returns'].map(bin_mapping)
    return weekly_data

def get_news(symbol, data, retries=3, delay=5, timeout=30):
    news_list = []
    for end_date, row in data.iterrows():
        start_date = row['Start Date'].strftime('%Y-%m-%d')
        end_date_str = row['End Date'].strftime('%Y-%m-%d')
        for attempt in range(retries):
            try:
                time.sleep(1)
                response = finnhub_client._session.get(
                    "https://finnhub.io/api/v1/company-news",
                    params={"symbol": symbol, "from": start_date, "to": end_date_str, "token": finnhub_client.api_key},
                    timeout=timeout
                )
                weekly_news = response.json()
                break
            except requests.exceptions.ReadTimeout:
                print(f"[Timeout] {symbol} news from {start_date} to {end_date_str}, retry {attempt+1}/{retries}")
                time.sleep(delay)
                weekly_news = []
            except Exception as e:
                print(f"[Error] {symbol} news fetch error: {e}")
                weekly_news = []
                break
        formatted_news = [
            {"date": datetime.fromtimestamp(n['datetime']).strftime('%Y%m%d%H%M%S'), "headline": n['headline'], "summary": n['summary']}
            for n in weekly_news if 'datetime' in n and 'headline' in n and 'summary' in n
        ]
        formatted_news.sort(key=lambda x: x['date'])
        news_list.append(json.dumps(formatted_news))
    data['News'] = news_list
    return data

def get_basics(symbol, data, start_date, always=False):
    basic_financials = finnhub_client.company_basic_financials(symbol, 'all')
    final_basics, basic_list, basic_dict = [], [], defaultdict(dict)
    for metric, value_list in basic_financials['series']['quarterly'].items():
        for value in value_list:
            basic_dict[value['period']].update({metric: value['v']})
    for k, v in basic_dict.items():
        v.update({'period': k})
        basic_list.append(v)
    basic_list.sort(key=lambda x: x['period'])
    for i, row in data.iterrows():
        start_date = row['End Date'].strftime('%Y-%m-%d')
        last_start_date = start_date if i < 2 else data.loc[i-2, 'Start Date'].strftime('%Y-%m-%d')
        used_basic = {}
        for basic in basic_list[::-1]:
            if (always and basic['period'] < start_date) or (last_start_date <= basic['period'] < start_date):
                used_basic = basic
                break
        final_basics.append(json.dumps(used_basic))
    data['Basics'] = final_basics
    return data

def prepare_data_for_symbol(symbol, data_dir, start_date, end_date, with_basics=True):
    data = get_returns(symbol, start_date, end_date)
    data = get_news(symbol, data)
    if with_basics:
        data = get_basics(symbol, data, start_date)
        data.to_csv(f"{data_dir}/{symbol}_{start_date}_{end_date}.csv")
    else:
        data['Basics'] = [json.dumps({})] * len(data)
        data.to_csv(f"{data_dir}/{symbol}_{start_date}_{end_date}_nobasics.csv")
    return data

# ----------------------------------------------------------------------------------- #
# ---------------------------------- GPT4 ANALYSIS ---------------------------------- #
# ----------------------------------------------------------------------------------- #


def append_to_csv(filename, input_data, output_data):
    
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([input_data, output_data])

        
def initialize_csv(filename):
    
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["prompt", "answer"])

def query_gpt4(symbol_list, data_dir, start_date, end_date, min_past_weeks=1, max_past_weeks=3, with_basics=True):
    import os
    import pandas as pd
    from tqdm import tqdm
    from prompt import get_all_prompts
    from data import append_to_csv, initialize_csv
    from data import SYSTEM_PROMPTS

    for symbol in tqdm(symbol_list):
        
        csv_file = f'{data_dir}/{symbol}_{start_date}_{end_date}_mistral.csv' if with_basics else \
                   f'{data_dir}/{symbol}_{start_date}_{end_date}_nobasics_mistral.csv'
        
        if not os.path.exists(csv_file):
            initialize_csv(csv_file)
            pre_done = 0
        else:
            df = pd.read_csv(csv_file)
            pre_done = len(df)

        prompts = get_all_prompts(symbol, data_dir, start_date, end_date, min_past_weeks, max_past_weeks, with_basics)
        system_prompt = SYSTEM_PROMPTS["crypto"] if symbol in CRYPTO else SYSTEM_PROMPTS["company"]

        for i, user_prompt in enumerate(prompts):
            
            if i < pre_done:
                continue

            print(f"{symbol} - generating prompt {i}")

            full_prompt = f"{system_prompt}\n\n{user_prompt}"

            try:
                output = text_generator(full_prompt)[0]["generated_text"]
                answer = output[len(full_prompt):].strip()
            except Exception as e:
                print(f"Error generating for {symbol}-{i}: {e}")
                answer = ""

            append_to_csv(csv_file, user_prompt, answer)


# ----------------------------------------------------------------------------------- #
# -------------------------- TRANSFORM INTO TRAINING FORMAT ------------------------- #
# ----------------------------------------------------------------------------------- #

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

SYSTEM_PROMPTS = {
    "company": "You are a seasoned stock market analyst. Your task is to list the positive developments and potential concerns for companies based on relevant news and basic financials from the past weeks, then provide an analysis and prediction for the companies' stock price movement for the upcoming week. " \
    "Your answer format should be as follows:\n\n[Positive Developments]:\n1. ...\n\n[Potential Concerns]:\n1. ...\n\n[Prediction & Analysis]:\n...\n",

    "crypto": "You are a seasoned crypto market analyst. Your task is to list the positive developments and potential concerns for cryptocurrencies based on relevant news and basic financials from the past weeks, then provide an analysis and prediction for the cryptocurrencies price movement for the upcoming week. " \
    "Your answer format should be as follows:\n\n[Positive Developments]:\n1. ...\n\n[Potential Concerns]:\n1. ...\n\n[Prediction & Analysis]:\n...\n",
}

def gpt4_to_llama(symbol, data_dir, start_date, end_date, with_basics=True):

    csv_file = f'{data_dir}/{symbol}_{start_date}_{end_date}_mistral.csv' if with_basics else \
            f'{data_dir}/{symbol}_{start_date}_{end_date}_nobasics_mistral.csv'

    df = pd.read_csv(csv_file)
    
    prompts, answers, periods, labels = [], [], [], []
    
    for i, row in df.iterrows():
        
        prompt, answer = row['prompt'], row['answer']
        
        res = re.search(r"Then let's assume your prediction for next week \((.*)\) is ((:?up|down) by .*%).", prompt)
        
        period, label = res.group(1), res.group(2)
#         label = label.replace('more than 5', '5+')
        
        prompt = re.sub(
            r"Then let's assume your prediction for next week \((.*)\) is (up|down) by ((:?.*)%). Provide a summary analysis to support your prediction. The prediction result need to be inferred from your analysis at the end, and thus not appearing as a foundational factor of your analysis.", 
            f"Then make your prediction of the {symbol} cryptocurrency price movement for next week ({period}). Provide a summary analysis to support your prediction.",
            prompt
        )
        try:
            answer = re.sub(
                r"\[Prediction & Analysis\]:\s*",
                f"[Prediction & Analysis]:\nPrediction: {label.capitalize()}\nAnalysis: ",
                answer
            )
        except Exception:
            print(symbol, i)
            print(label)
            print(answer)
            continue
            
        system_prompt = SYSTEM_PROMPTS["crypto"] if symbol in CRYPTO else SYSTEM_PROMPTS["company"]
        new_system_prompt = system_prompt.replace(':\n...', '\nPrediction: ...\nAnalysis: ...')
#         new_system_prompt = SYSTEM_PROMPT.replace(':\n...', '\nPrediction: {Up|Down} by {1-2|2-3|3-4|4-5|5+}%\nAnalysis: ...')
        
        prompt = B_INST + B_SYS + new_system_prompt + E_SYS + prompt + E_INST
        
        prompts.append(prompt)
        answers.append(answer)
        periods.append(period)
        labels.append(label)
        
    return {
        "prompt": prompts,
        "answer": answers,
        "period": periods,
        "label": labels,
    }


def create_dataset(symbol_list, data_dir, start_date, end_date, train_ratio=0.8, with_basics=True):

    train_dataset_list = []
    test_dataset_list = []

    for symbol in symbol_list:

        data_dict = gpt4_to_llama(symbol, data_dir, start_date, end_date,  with_basics)
#         print(data_dict['prompt'][-1])
#         print(data_dict['answer'][-1])
        symbols = [symbol] * len(data_dict['label'])
        data_dict.update({"symbol": symbols})

        dataset = Dataset.from_dict(data_dict)
        train_size = round(train_ratio * len(dataset))

        train_dataset_list.append(dataset.select(range(train_size)))
        if train_size >= len(dataset):
            continue
        test_dataset_list.append(dataset.select(range(train_size, len(dataset))))

    train_dataset = datasets.concatenate_datasets(train_dataset_list)
    test_dataset = datasets.concatenate_datasets(test_dataset_list)

    dataset = datasets.DatasetDict({
        'train': train_dataset,
        'test': test_dataset
    })
    
    return dataset

def summarize_financials(basics):
    highlights = []

    # P/E Ratio
    pe = basics.get("peTTM")
    if pe is not None:
        highlights.append(f"High valuation (P/E: {pe})")

    # Return on Equity
    roe = basics.get("roeTTM")
    if roe is not None and roe > 0.3:
        highlights.append(f"Strong return on equity (ROE: {roe})")

    # Operating Margin
    operating_margin = basics.get("operatingMargin")
    if operating_margin is not None and operating_margin > 0.3:
        highlights.append(f"Healthy operating margin ({operating_margin})")

    # Debt to Equity
    debt_to_equity = basics.get("totalDebtToEquity")
    if debt_to_equity is not None and debt_to_equity > 2:
        highlights.append(f"High leverage (Debt/Equity: {debt_to_equity})")

    # Net Margin
    net_margin = basics.get("netMargin")
    if net_margin is not None and net_margin > 0.2:
        highlights.append(f"Solid profitability (Net Margin: {net_margin})")

    # ROA
    roa = basics.get("roaTTM")
    if roa is not None and roa > 0.1:
        highlights.append(f"Efficient asset utilization (ROA: {roa})")

    return highlights
