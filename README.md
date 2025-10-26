# Customer Support Agent for Product Returns

A **LangGraph-based intelligent customer support agent** designed to handle product return queries using the **ReAct (Reason + Act)** reasoning pattern with Large Language Models (LLMs).

---

## Overview

This agent assists customers with product return–related queries by:

- Understanding natural language questions about returns  
- Looking up order information from a mock database  
- Calculating refund amounts  
- Providing empathetic, context-aware responses  

---

## Architecture

### ReAct Pattern Implementation

The agent follows the **Reason + Act** paradigm:

1. **Reason:** The LLM interprets the user's query and determines what information is needed.  
2. **Act:** The agent calls appropriate tools to gather information.  
3. **Observe:** The agent receives results from the tools.  
4. **Respond:** The agent formulates a natural language response based on the gathered information.

---

## Available Tools

| Tool Name               | Description                                                |
|--------------------------|------------------------------------------------------------|
| **lookup_order**         | Retrieves order details by order ID                        |
| **product_return_policy**| Fetches the return policy (30-day window)                  |
| **calculate_refund**     | Calculates refund amount based on return reason            |

---

## Mock Data

The agent includes sample order data to simulate various return eligibility scenarios:

| Order ID | Product                      | Days Since Purchase | Eligible? |
|-----------|------------------------------|----------------------|------------|
| ORD-001   | Wireless Bluetooth Headphones | 10 days              | Yes     |
| ORD-002   | Smart Fitness Watch           | 35 days              | No      |
| ORD-003   | USB-C Charging Cable          | 5 days               | Yes     |
| ORD-004   | Portable Power Bank           | 45 days              | No      |
| ORD-005   | Laptop Sleeve Case            | 20 days              | Yes     |

---

## Setup and Run Instructions

### Prerequisites

- **Python 3.9 or higher**  
- **Anthropic Claude API key** 

---

### Step 1: Clone or Download the Repository

```bash
# If using Git
git clone https://github.com/vidhyakshayakannan/Customer-Support-Agent.git
cd customer-support-agent
````

---

### Step 2: Create a Virtual Environment (Recommended)

```bash
# Create a virtual environment
python -m venv venv

# Activate it
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

---

### Step 3: Install Dependencies

```bash
pip install langgraph langchain-anthropic langchain-core anthropic
```

Or use the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

**requirements.txt**

```
langgraph>=0.0.20
langchain-core>=0.1.0
langchain-anthropic>=0.1.0
anthropic>=0.18.0
```

---

### Step 4: Set the API Key

```
export ANTHROPIC_API_KEY="your-api-key-here"
```

---

### Step 5: Run the Agent

#### Run Example Scenarios

```bash
python customer.py
```

This runs predefined examples covering:

* Eligible return (recent order)
* Expired return (past 30 days)
* General return policy question

---

#### Run in Interactive Mode

Uncomment the final line in `customer.py`:

```python
if __name__ == "__main__":
    # ... existing code ...
    interactive_mode()  # Uncomment this line
```

Then run:

```bash
python customer.py
```

You can now chat interactively with the agent using any order ID (`ORD-001` through `ORD-005`).
