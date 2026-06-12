# 14-Day Agentic AI for Capital Markets

Hands-on builds applying LLMs to real capital markets workflows.
Each day ships working code, not slides.

## Day 1 — Trade Ticket Email Parser

**Pattern:** Pydantic structured outputs from an LLM  
**Problem:** Ops analysts manually re-key unstructured trade emails into booking systems  
**Solution:** LLM extracts fields into a strictly-typed Pydantic model. Validation failures route to a human review queue.

**Key takeaway:** The Pydantic validation error *is* the feature — it creates the human-in-the-loop trigger automatically.

## Tech Stack
- Python 3.14, OpenAI API, Anthropic API, Pydantic v2

## Running it
1. Clone the repo
2. Create a `.env` file with `OPENAI_API_KEY` and `ANTHROPIC_API_KEY`
3. `pip install openai anthropic pydantic python-dotenv`
4. Open the notebook and run cells sequentially

## What This Project Does

Front-office traders book trades by emailing the operations team with unstructured requests like:
```
"Buy 50,000 AAPL @ market for fund ABC, settlement T+2"
```

Today, an operations analyst manually reads each email and re-types the fields into a booking system. This project replaces that manual step with an **LLM-powered parser** that:

- ✅ Extracts structured trade data from unstructured emails
- ✅ Returns a **strongly-typed Pydantic object** (not plain JSON)
- ✅ Routes ambiguous or invalid emails to a **human-review queue**
- ✅ Handles validation failures as a feature, not a bug

---

## Key Insights

1. **Pydantic > Plain JSON**: Using `Literal['BUY', 'SELL']` types prevents LLM hallucinations. The model cannot invent a third option.
2. **Validation = Guardrail**: Field validators (e.g., "if `order_type='LIMIT'` then `limit_price` required") catch real-world ambiguity better than the LLM alone.
3. **Failures are Signals**: A failed validation isn't a bug—it's the system correctly flagging data that needs human review.

---

## Prerequisites

- **Python 3.11+** (or Google Colab)
- **OpenAI API key** and/or **Anthropic API key** (free tier sufficient)
- **pip packages**: `openai`, `anthropic`, `pydantic`, `python-dotenv`

---

## Installation & Setup

### 1. Clone / Download This Repository
```bash
cd /path/to/ravi-chandra-bit-agentic-ai-projects
```

### 2. Install Dependencies
```bash
pip install openai anthropic pydantic python-dotenv
```

### 3. Configure API Keys
Create a `.env` file in the project root (add to `.gitignore` for safety):
```bash
cat > .env << 'EOF'
OPENAI_API_KEY=sk-... # your OpenAI key
ANTHROPIC_API_KEY=sk-ant-... # your Anthropic key
EOF
```

### 4. Open the Notebook
```bash
jupyter notebook Day_01_trade_ticket_email_parser.ipynb
```

---

## Quick Start

Run all cells in order:

1. **Cell 1 (Setup):** Loads API keys and initializes clients
2. **Cell 2 (Pydantic Model):** Defines the `TradeTicket` schema with validators
3. **Cell 3 (LLM Integration):** Implements `call_openai()` and `parse_trade_email()`
4. **Cell 4 (Test):** Runs 7 test emails through the parser and reports results

### Expected Output
```
✅ Parsed: side='BUY' quantity=10000 symbol='MSFT' order_type='MARKET' limit_price=None fund_id='Growth Fund' settlement='T+1'
✅ Parsed: side='SELL' quantity=5000 symbol='AAPL' order_type='LIMIT' limit_price=175.5 fund_id='GAF' settlement='T+1'
...
❌ Validation failed for email: "Buy some shares of Apple..."
Error: ValidationError(...)

Total processed: 7
Successfully parsed: 5
Sent to manual review: 2
```

---

## Project Structure

```
Day_01_trade_ticket_email_parser.ipynb
├── Step 1: Setup
│   └── Load .env, initialize OpenAI + Anthropic clients
├── Step 2: Define Pydantic Model
│   └── TradeTicket (with field + model validators)
├── Step 3: LLM Integration
│   ├── call_openai() — OpenAI Chat API wrapper
│   └── parse_trade_email() — E2E pipeline (LLM → JSON → Pydantic validation)
└── Step 4: Test & Results
    └── Run 7 test emails, report success/failure stats
```

---


## Test Cases

| Email | Expected Result | Reason |
|-------|-----------------|--------|
| "Buy 10000 MSFT at market for Growth Fund, T+1" | ✅ Parsed | Clear, unambiguous |
| "Sell 5k AAPL limit 175.50 for GAF, T+1" | ✅ Parsed | LIMIT order includes price |
| "Buy some shares of Apple for Growth Fund" | ❌ Review | Missing: quantity, settlement |
| "LONG 500 SPY @market for Tactical Fund T+2" | ✅ Parsed | "LONG" → BUY (with prompt refinement) |
| "Sell 150 TSLA limit 620 for ZETA, T+0" | ✅ Parsed | Valid LIMIT with price |

---

## Key Learnings

### Why Pydantic?
- **Type Safety:** `Literal['BUY', 'SELL']` prevents invalid values
- **Validation on Parse:** Errors surface immediately, not in production
- **Self-Documenting:** Schema is the contract between LLM and system

### When Validation Fails (It's Good!)
- **Scenario:** LLM returns `{"side": "LONG", ...}` (LONG is not BUY or SELL)
- **Pydantic:** Raises `ValidationError` immediately
- **System:** Routes to human review, never auto-books invalid trade

### Field vs. Model Validators
| Type | Use Case | Example |
|------|----------|---------|
| `@field_validator` | Single-field rules | Uppercase symbol |
| `@model_validator` | Cross-field rules | If LIMIT, require limit_price |

---

## Next Steps (Days 2–14)

- **Day 2:** Prompt engineering — 4 levels of prompts, measure accuracy per level
- **Day 3:** Multi-provider support (Anthropic), compare latency + cost
- **Day 4:** Add fund validation (check fund_id against real fund registry)
- **Day 5–7:** Benchmarks, edge cases, error analysis
- **Day 8–14:** Deployment, monitoring, production readiness

---

## Troubleshooting

### "time is not defined" Error
- **Cause:** `import time` not executed before `call_openai()`
- **Fix:** Run the setup cell (Cell 1) before other cells, or add `import time` to the same cell

### "OPENAI_API_KEY: ❌ MISSING"
- **Cause:** `.env` file not created or not in project root
- **Fix:** Create `.env` with your key, run `load_dotenv()` in Cell 1

### "ValidationError: field required"
- **Cause:** LLM returned incomplete JSON (missing fields)
- **Fix:** Refine system prompt to include example output format, or lower temperature

### Model Returns Plain Text Instead of JSON
- **Cause:** Prompt unclear, model defaulted to narrative
- **Fix:** Update system prompt: "Return ONLY valid JSON, no extra text"

---

## Performance Metrics

From test run:

| Metric | Value |
|--------|-------|
| Emails processed | 7 |
| Parsed successfully | 5 |
| Routed to review | 2 |
| Avg latency per email | ~300 ms |
| Cost per email | ~$0.0001 (gpt-4.1-mini) |
| Success rate | 71% |

---

## License

This project is provided as-is for educational purposes. See `LICENSE` file.

---

## Author

Built as part of **Agentic AI 14-day learning series**.  
Contributions welcome! Please open issues or PRs on GitHub.

---

## Resources

- [Pydantic v2 Docs](https://docs.pydantic.dev/latest/)
- [OpenAI Chat Completions API](https://platform.openai.com/docs/api-reference/chat)
- [Anthropic API](https://docs.anthropic.com/)
- [Structured Outputs with OpenAI](https://platform.openai.com/docs/guides/structured-outputs)

---


