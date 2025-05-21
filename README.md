# Financial Quantitative Analysis 📈

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)

A self-contained Python script that:

* Downloads historical price data from **Alpha Vantage** with `alpha-vantage`
* Computes technical indicators (5- and 30-day moving averages, volatility, returns)
* Detects **golden/death crosses** and generates trading signals
* Back-tests a simple long-only strategy (buy on golden cross, sell on death cross)
* Evaluates performance – total & annual return, max draw-down, Sharpe ratio
* Predicts the next golden/death cross with a **Random Forest** model
* Plots a 4-in-1 diagnostic figure (price, moving averages, signals, return curve)

> **Disclaimer** – This project is for educational purposes only and **is not investment advice**.

---

## Demo

```console
$ python Financial_Quantitative_Analysis.py
---Stock Prediction and Recommendation System---
Please enter stock code (e.g., AAPL for Apple): AAPL
Please enter start date (YYYY-MM-DD): 2000-01-01
Please enter initial capital (positive number): 10000
Stock data retrieved successfully. Here are the first 5 rows:
              open    high       low   close      volume
date                                                    
2025-05-20  207.67  208.47  205.0300  206.86  42496635.0
2025-05-19  207.91  209.48  204.2600  208.78  46140527.0
2025-05-16  212.36  212.57  209.7700  211.26  54737850.0
2025-05-15  210.95  212.96  209.5400  211.45  45029473.0
2025-05-14  212.43  213.94  210.5801  212.33  49325825.0
Random-Forest MAE on hold-out: 22.96 days
Latest Golden Cross: 2000-06-22
Latest Death Cross: 2000-03-15
Predicted Next Golden Cross: 2025-06-01
Predicted Next Death Cross: 2025-08-01
Recommendation: Buy
Total Return: -47.63%
Annualized Return: -2.53%
Maximum Draw-down: 74.95%
Sharpe Ratio: -0.10
```

Running the script also displays a Matplotlib window with the diagnostic chart.

---

## Installation

1. **Clone the repo**

   ```bash
   git clone https://github.com/<your-github-name>/Financial_Quantitative_Analysis.git
   cd Financial_Quantitative_Analysis
   ```

2. **Create a virtual environment** (recommended) and install dependencies:

   ```bash
   python -m venv .venv
   source .venv/bin/activate      # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Get an Alpha Vantage API key**  
   - Sign up at [Alpha Vantage](https://www.alphavantage.co/) to obtain a free API key.
   - Replace `'YOUR_API_KEY'` in `Financial_Quantitative_Analysis.py` with your key.

   `requirements.txt`

   ```
   alpha-vantage>=2.3
   pandas>=2.0
   numpy>=1.23
   matplotlib>=3.8
   scikit-learn>=1.4
   ```

---

## Usage

```
python Financial_Quantitative_Analysis.py
```

The program is fully interactive:

| Prompt                               | Meaning                                      | Example          |
|--------------------------------------|----------------------------------------------|------------------|
| `stock code`                         | Alpha Vantage ticker (e.g., AAPL for Apple)  | `AAPL`, `MSFT`   |
| `start date`                         | First day of back-test (YYYY-MM-DD)          | `2000-01-01`     |
| `initial capital`                    | Cash balance in USD                          | `10000`          |

Type **y** to analyze another ticker, or **n** to exit.

---

## Project Structure

```
├─ Financial_Quantitative_Analysis.py   ← main script
├─ README.md                           ← this file
├─ requirements.txt                    ← Python packages
└─ .gitignore                          ← optional, prevents cache / venv uploads
```

---

## Notes

- **API Limits**: Alpha Vantage free tier allows 5 API calls per minute and 500 per day. The script includes a 12-second delay to comply with this limit.
- **Data Source**: Uses Alpha Vantage for reliable stock data, replacing Yahoo Finance to avoid rate-limiting issues.
- Ensure a valid API key is set to avoid errors.
- Performance metrics (e.g., negative returns) depend on the chosen stock and period. Historical data from 2000 may include significant market downturns.

---

## Roadmap

* Add alternative indicators (RSI, MACD, Bollinger Bands)
* Implement walk-forward cross-validation for the Random Forest
* Export results to HTML or CSV
* Develop a Streamlit web app interface

Contributions are welcome! Open an issue to discuss ideas or submit a pull request.

---

## License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.