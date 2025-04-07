import os
import json
import random
import finnhub
import yfinance as yf
import pandas as pd
from utils import summarize_financials
from indices import *

finnhub_client = finnhub.Client(api_key="cvf04q9r01qjugsfbuu0cvf04q9r01qjugsfbuug")

def get_company_prompt(symbol):
    
    profile = finnhub_client.company_profile2(symbol=symbol)

    company_template = "[Company Introduction]:\n\n{name} is a leading entity in the {finnhubIndustry} sector. Incorporated and publicly traded since {ipo}, the company has established its reputation as one of the key players in the market. As of today, {name} has a market capitalization of {marketCapitalization:.2f} in {currency}, with {shareOutstanding:.2f} shares outstanding." \
        "\n\n{name} operates primarily in the {country}, trading under the ticker {ticker} on the {exchange}. As a dominant force in the {finnhubIndustry} space, the company continues to innovate and drive progress within the industry."

    formatted_str = company_template.format(**profile)
    
    return formatted_str


def get_crypto_prompt(symbol):

    profile = yf.Ticker(symbol).info

    crpyto_template = """[Cryptocurrency Introduction]: {description}. It has a market capilization of {marketCap}."""
    
    formatted_str = crpyto_template.format(**profile)
    
    return formatted_str


def get_prompt_by_row(symbol, row):

    start_date = row['Start Date'] if isinstance(row['Start Date'], str) else row['Start Date'].strftime('%Y-%m-%d')
    end_date = row['End Date'] if isinstance(row['End Date'], str) else row['End Date'].strftime('%Y-%m-%d')
    term = 'increased' if row['End Price'] > row['Start Price'] else 'decreased'
    head = "From {} to {}, {}'s stock price {} from {:.2f} to {:.2f}. News during this period are listed below:\n\n".format(
        start_date, end_date, symbol, term, row['Start Price'], row['End Price'])
    
    news = json.loads(row["News"])
    news = ["[Headline]: {}\n[Summary]: {}\n".format(
        n['headline'], n['summary']) for n in news if n['date'][:8] <= end_date.replace('-', '') and \
        not n['summary'].startswith("Looking for stock market analysis and research with proves results?")]

    basics = json.loads(row['Basics'])
    if basics:
        basics = "Some recent basic financials of {}, reported at {}, are presented below:\n\n[Basic Financials]:\n\n".format(
            symbol, basics['period']) + "\n".join(f"{k}: {v}" for k, v in basics.items() if k != 'period')
    else:
        basics = "[Basic Financials]:\n\nNo basic financial reported."
    
    return head, news, basics


def get_crypto_prompt_by_row(symbol, row):

    start_date = row['Start Date'] if isinstance(row['Start Date'], str) else row['Start Date'].strftime('%Y-%m-%d')
    end_date = row['End Date'] if isinstance(row['End Date'], str) else row['End Date'].strftime('%Y-%m-%d')
    term = 'increased' if row['End Price'] > row['Start Price'] else 'decreased'
    head = "From {} to {}, {}'s stock price {} from {:.2f} to {:.2f}. News during this period are listed below:\n\n".format(
        start_date, end_date, symbol, term, row['Start Price'], row['End Price'])
    
    news = json.loads(row["News"])
    news = ["[Headline]: {}\n[Summary]: {}\n".format(
        n['headline'], n['summary']) for n in news if n['date'][:8] <= end_date.replace('-', '') and \
        not n['summary'].startswith("Looking for stock market analysis and research with proves results?")]

    return head, news, None


def sample_news(news, k=5):
    
    return [news[i] for i in sorted(random.sample(range(len(news)), k))]


def map_bin_label(bin_lb):
    
    lb = bin_lb.replace('U', 'up by ')
    lb = lb.replace('D', 'down by ')
    lb = lb.replace('1', '0-1%')
    lb = lb.replace('2', '1-2%')
    lb = lb.replace('3', '2-3%')
    lb = lb.replace('4', '3-4%')
    if lb.endswith('+'):
        lb = lb.replace('5+', 'more than 5%')
#         lb = lb.replace('5+', '5+%')
    else:
        lb = lb.replace('5', '4-5%')
    
    return lb

PROMPT_END = {
    "company": "\n\nBased on all the information before {start_date}, complete the following three steps:\n\n"
                "1. List 2-4 positive developments inferred from company-related news or financials.\n"
                "2. List 2-4 potential concerns that might negatively affect the company's performance.\n"
                "3. Make a prediction for {symbol}'s stock movement from {start_date} to {end_date}.\n\n"
                "Your answer must strictly follow this format:\n\n"
                "[Positive Developments]:\n1. ...\n2. ...\n\n"
                "[Potential Concerns]:\n1. ...\n2. ...\n\n"
                "[Prediction & Analysis]:\n"
                "Prediction: [Up or Down] by [0-1%, 1-2%, 2-3%, 3-4%, 4-5%, more than 5%]\n"
                "Analysis: Base your prediction on the listed developments and concerns only, not price movements.",
}

def get_company_prompt(symbol):
    profile = finnhub_client.company_profile2(symbol=symbol)
    return (f"[Company Introduction]:\n\n{profile['name']} is a leading entity in the {profile['finnhubIndustry']} sector. "
            f"Incorporated and publicly traded since {profile['ipo']}, it currently has a market cap of {profile['marketCapitalization']:.2f} {profile['currency']} with "
            f"{profile['shareOutstanding']:.2f} shares outstanding. It operates primarily in {profile['country']}, trading as {profile['ticker']} on the {profile['exchange']}.")

def get_prompt_by_row(symbol, row):
    start = row['Start Date'][:10]
    end = row['End Date'][:10]
    movement = 'increased' if row['End Price'] > row['Start Price'] else 'decreased'
    header = f"From {start} to {end}, {symbol}'s stock price {movement} from {row['Start Price']:.2f} to {row['End Price']:.2f}."

    news_list = json.loads(row['News'])
    news_items = [f"- {n['headline']}: {n['summary']}" for n in news_list if not n['summary'].startswith("Looking for stock market analysis")]
    news_text = "\n".join(news_items[:5]) if news_items else "No relevant news reported."

    basics = json.loads(row['Basics']) if isinstance(row['Basics'], str) else {}
    basics_summary = summarize_financials(basics)

    return header, news_text, basics_summary

def map_bin_label(bin_lb):
    return bin_lb.replace('U', 'Up by ').replace('D', 'Down by ').replace('5+', 'more than 5%')\
                  .replace('5', '4-5%').replace('4', '3-4%').replace('3', '2-3%')\
                  .replace('2', '1-2%').replace('1', '0-1%')

def get_all_prompts(symbol, data_dir, start_date, end_date, min_past_weeks=1, max_past_weeks=2, with_basics=True):
    df_path = f'{data_dir}/{symbol}_{start_date}_{end_date}.csv'
    df = pd.read_csv(df_path)

    info_prompt = get_company_prompt(symbol)
    prev_rows = []
    prompts = []

    for i, row in df.iterrows():
        if len(prev_rows) < min_past_weeks:
            prev_rows.append(row)
            continue

        num_prev = min(len(prev_rows), max_past_weeks)
        history_parts = []
        for prev_row in prev_rows[-num_prev:]:
            h, n, _ = get_prompt_by_row(symbol, prev_row)
            history_parts.append(h + "\n" + n)

        this_head, this_news, this_basics = get_prompt_by_row(symbol, row)
        prompt = info_prompt + "\n\n" + "\n\n".join(history_parts) + "\n\n" + this_head + "\n" + this_news + "\n\n" + this_basics

        prompt += PROMPT_END['company'].format(
            start_date=row['Start Date'],
            end_date=row['End Date'],
            symbol=symbol
        )

        prompts.append(prompt.strip())
        prev_rows.append(row)
        if len(prev_rows) > max_past_weeks:
            prev_rows.pop(0)

    return prompts
