# Scheduled Retraining Setup Guide

## Overview

TrendPulse includes a scheduled retraining system that automatically updates models after market close.

## Quick Start

### 1. Manual Run (Test First)
```bash
cd /Users/kairavkaran/Documents/Stock_Predictor
python3 scheduled_train.py
```

This will train models for top 10 stocks and save them to `models/`.

### 2. Train Specific Tickers
```bash
python3 scheduled_train.py AAPL TSLA NVDA GOOGL
```

### 3. Set Up Cron Job (Automatic Daily Training)

Open crontab:
```bash
crontab -e
```

Add this line (runs at 5:30 PM EST every weekday):
```cron
30 17 * * 1-5 cd /Users/kairavkaran/Documents/Stock_Predictor && /usr/bin/python3 scheduled_train.py >> logs/cron.log 2>&1
```

### 4. Verify Cron Job
```bash
crontab -l
```

## File Structure
```
models/
├── SPY_model.keras           # Trained model file
├── SPY_metadata.json         # Training metadata
├── AAPL_model.keras
├── AAPL_metadata.json
└── last_training_summary.json
```

## The App Auto-Detects Pre-trained Models
When you select a ticker with a saved model, the app will show:
- ✅ "Pre-trained model available!"
- Option to use it (skips training, instant predictions)

## Logs
- `logs/scheduled_train.log` - Training output
- `logs/cron.log` - Cron job output
