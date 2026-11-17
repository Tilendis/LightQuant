# -*- coding: utf-8 -*-
"""
gen_embeddings_cpu.py — CPU adaptation of utils/word2vec.py

Generates per-(stock, trading_date) news embeddings of shape (768,) (BERT CLS of
that date's concatenated news text) using a Chinese BERT on CPU, with
multiprocessing for speed.

Rationale: the repo's word2vec.py stacks a variable number of past days into
(1, N, 768), which is incompatible with StockNet (news input_dim=768 per day)
and Backtest_Dataset (news_dim=768). This script produces the per-day (768,)
format the models actually consume; dates without news get a zero vector.

Usage (run from repo root after utils/util.py has created trading_date_list.csv):
    python utils/gen_embeddings_cpu.py --dataset CSMD50 --model_path <local bert dir>
"""
import os
import argparse
import numpy as np
import pandas as pd
from multiprocessing import Pool
import torch

def _worker_init(model_path, device):
    global tokenizer, model, device_name
    from transformers import AutoTokenizer, AutoModel
    device_name = device
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path).to(device)
    model.eval()

def _embed_text(text):
    global tokenizer, model, device_name
    if text is None or (isinstance(text, str) and not text.strip()):
        return None
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device_name) for k, v in inputs.items()}
        outputs = model(**inputs)
        emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()  # (1, 768)
    return emb

def _stock_job(args):
    stock, news_dir, out_dir, trading_dates = args
    out_stock_dir = os.path.join(out_dir, stock)
    os.makedirs(out_stock_dir, exist_ok=True)
    # map date -> concatenated text
    date_text = {}
    if os.path.isdir(news_dir):
        for f in os.listdir(news_dir):
            if not f.endswith('.csv'):
                continue
            date = f[:-4]
            try:
                pd.to_datetime(date)
            except Exception:
                continue
            df = pd.read_csv(os.path.join(news_dir, f))
            if 'text' not in df.columns:
                continue
            date_text[date] = '. '.join(df['text'].dropna().astype(str))
    n_bert = 0
    for d in trading_dates:
        emb = None
        if d in date_text:
            emb = _embed_text(date_text[d])
            if emb is not None:
                n_bert += 1
        if emb is None:
            emb = np.zeros((1, 768), dtype=np.float32)
        np.save(os.path.join(out_stock_dir, f'{d}.npy'), emb.astype(np.float32))
    return stock, len(date_text), n_bert

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='CSMD50', type=str)
    parser.add_argument('--model_path', required=True, type=str, help='local dir of bert-base-chinese')
    parser.add_argument('--workers', default=8, type=int)
    parser.add_argument('--device', default='auto', type=str, help='cuda / cpu / auto')
    parser.add_argument('--trading_date_list', default='./dataset/trading_date_list.csv', type=str)
    args = parser.parse_args()

    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device: {args.device}')

    root = './dataset'
    news_root = os.path.join(root, args.dataset, 'news')
    out_root = os.path.join(root, args.dataset, 'news_embedding')
    os.makedirs(out_root, exist_ok=True)

    df_dates = pd.read_csv(args.trading_date_list)
    trading_dates = df_dates['Date'].astype(str).tolist()
    print(f'trading dates: {len(trading_dates)}')

    stocks = sorted(os.listdir(news_root))
    jobs = [(s, os.path.join(news_root, s), out_root, trading_dates) for s in stocks]
    print(f'stocks to process: {len(stocks)}')

    with Pool(args.workers, initializer=_worker_init, initargs=(args.model_path, args.device)) as pool:
        for stock, n_dates, n_bert in pool.imap_unordered(_stock_job, jobs):
            print(f'{stock}: news_dates={n_dates}, bert_calls={n_bert}')

    print('DONE')

if __name__ == '__main__':
    main()
