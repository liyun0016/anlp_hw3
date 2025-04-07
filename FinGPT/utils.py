# utils.py

def summarize_financials(basics: dict):
    if not basics:
        return "[Basic Financials]:\n\nNo basic financials reported."

    highlights = []
    try:
        pe = float(basics.get("peTTM", -1))
        if pe > 0:
            highlights.append(f"High valuation (P/E: {pe})")
        roe = float(basics.get("roeTTM", -1))
        if roe > 0.3:
            highlights.append(f"Strong return on equity (ROE: {roe})")
        margin = float(basics.get("operatingMargin", -1))
        if margin > 0.2:
            highlights.append(f"Healthy operating margin ({margin})")
        debt = float(basics.get("totalDebtToEquity", -1))
        if debt > 2:
            highlights.append(f"High leverage (Debt/Equity: {debt})")
        net_margin = float(basics.get("netMargin", -1))
        if net_margin > 0.2:
            highlights.append(f"Solid profitability (Net Margin: {net_margin})")
        roa = float(basics.get("roaTTM", -1))
        if roa > 0.1:
            highlights.append(f"Efficient asset utilization (ROA: {roa})")
    except Exception as e:
        highlights.append("Financial summary unavailable.")

    if not highlights:
        return "[Basic Financials Summary]:\n\nNo strong indicators reported."

    return "[Basic Financials Summary]:\n\n" + ", ".join(highlights)
