import os
from dotenv import load_dotenv
load_dotenv()
from datetime import datetime, timedelta
from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool

# purchase data

MOCK_PURCHASES = {
    "ORD-001": {
        "product_name": "Wireless Bluetooth Headphones",
        "purchase_date": (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d"),
        "order_id": "ORD-001",
        "price": 79.99,
        "customer_name": "John Smith"
    },
    "ORD-002": {
        "product_name": "Smart Fitness Watch",
        "purchase_date": (datetime.now() - timedelta(days=35)).strftime("%Y-%m-%d"),
        "order_id": "ORD-002",
        "price": 199.99,
        "customer_name": "Sarah Johnson"
    },
    "ORD-003": {
        "product_name": "USB-C Charging Cable",
        "purchase_date": (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d"),
        "order_id": "ORD-003",
        "price": 15.99,
        "customer_name": "Mike Davis"
    },
    "ORD-004": {
        "product_name": "Portable Power Bank",
        "purchase_date": (datetime.now() - timedelta(days=45)).strftime("%Y-%m-%d"),
        "order_id": "ORD-004",
        "price": 49.99,
        "customer_name": "Emily Chen"
    },
    "ORD-005": {
        "product_name": "Laptop Case",
        "purchase_date": (datetime.now() - timedelta(days=20)).strftime("%Y-%m-%d"),
        "order_id": "ORD-005",
        "price": 24.99,
        "customer_name": "Alex Rodriguez"
    }
}

# tools: react pattern implementation

@tool
def product_return_policy(product_category: str = "electronics") -> dict:
    """
    Get the return policy information including number of days for return eligibility.
    
    Args:
        product_category: The category of product (default: electronics)
    
    Returns:
        Dictionary with return policy details
    """
    policies = {
        "electronics": {
            "return_window_days": 15,
            "condition": "unopened or defective",
            "refund_type": "full refund or exchange"
        },
        "accessories": {
            "return_window_days": 45,
            "condition": "unused with original packaging",
            "refund_type": "full refund or store credit"
        },
        "default": {
            "return_window_days": 30,
            "condition": "unused and in original condition",
            "refund_type": "full refund"
        }
    }
    
    policy = policies.get(product_category.lower(), policies["default"])
    return {
        "return_window_days": policy["return_window_days"],
        "condition_required": policy["condition"],
        "refund_type": policy["refund_type"],
        "restocking_fee": "No restocking fee for defective items"
    }


@tool
def lookup_order(order_id: str) -> dict:
    """
    Look up order details by order ID.
    
    Args:
        order_id: The order ID to look up (e.g., ORD-001)
    
    Returns:
        Dictionary with order details or error message
    """
    order = MOCK_PURCHASES.get(order_id.upper())
    if order:
        return {
            "success": True,
            "order": order
        }
    return {
        "success": False,
        "message": f"Order {order_id} not found. Please check the order ID."
    }


@tool
def calculate_refund(order_id: str, return_reason: str = "general") -> dict:
    """
    Calculate refund amount for a return.
    
    Args:
        order_id: The order ID
        return_reason: Reason for return (defective, unwanted, wrong_item, etc.)
    
    Returns:
        Dictionary with refund calculation
    """
    order = MOCK_PURCHASES.get(order_id.upper())
    if not order:
        return {
            "success": False,
            "message": "Order not found"
        }
    
    price = order["price"]
    restocking_fee = 100
    
    # no restocking fee for defective items
    if return_reason.lower() not in ["defective", "wrong_item", "damaged"]:
        restocking_fee = 0  
    
    refund_amount = price - restocking_fee
    
    return {
        "success": True,
        "original_price": price,
        "restocking_fee": restocking_fee,
        "refund_amount": refund_amount,
        "refund_method": "Original payment method",
        "processing_time": "5-7 business days"
    }

TOOLS = {
    "lookup_order": lookup_order,
    "product_return_policy": product_return_policy,
    "calculate_refund": calculate_refund
}

# state definition

def add_messages(left: list, right: list) -> list:
    """Reducer function to append messages to the state"""
    return left + right


class AgentState(TypedDict):
    """State of the agent throughout the conversation"""
    messages: Annotated[list, add_messages]


# langgraph nodes 
def should_continue(state: AgentState) -> Literal["tools", "end"]:
    """Determine if we should continue to tools or end"""
    messages = state["messages"]
    last_message = messages[-1]
    
    # if the last message has tool calls, continue to tools
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "end"


def call_model(state: AgentState):
    """Call the LLM with current state"""
    messages = state["messages"]
    system_message = SystemMessage(content=f"""The date is {datetime.now()}. You are a helpful customer support agent specializing in product returns.

Your job is to:
1. Understand customer queries about returns
2. Use available tools to look up orders and check return eligibility
3. Provide clear, empathetic responses about return options

Available tools:
- lookup_order: Get order details by order ID
- product_return_policy: Get return policy information
- calculate_refund: Calculate refund amounts

Always:
- Be polite and empathetic
- Check eligibility before promising refunds
- Provide specific dates and timeframes
- Offer alternatives if returns aren't possible

Use the ReAct pattern: Reason about what information you need, then Act by using tools.""")
    
    
    llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0)
    llm_with_tools = llm.bind_tools([
        lookup_order,
        product_return_policy,
        calculate_refund
    ])
    response = llm_with_tools.invoke([system_message] + messages)
    return {"messages": [response]}


def execute_tools(state: AgentState):
    """Execute the tools that the LLM requested"""
    messages = state["messages"]
    last_message = messages[-1]
    
    # get the tool calls from the last message
    tool_calls = last_message.tool_calls
    
    # execute each tool and create response messages
    tool_messages = []
    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_id = tool_call["id"]
        
        # execute the tool
        tool_func = TOOLS[tool_name]
        try:
            if hasattr(tool_func, 'func'):
                result = tool_func.func(**tool_args)
            else:
                result = tool_func(**tool_args)
            
            tool_messages.append(
                ToolMessage(
                    content=str(result),
                    tool_call_id=tool_id,
                    name=tool_name
                )
            )
        except Exception as e:
            tool_messages.append(
                ToolMessage(
                    content=f"Error executing {tool_name}: {str(e)}",
                    tool_call_id=tool_id,
                    name=tool_name
                )
            )
    return {"messages": tool_messages}



####### build graph ######

def create_agent_graph():
    """Create the LangGraph workflow"""

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", execute_tools)
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "end": END
        }
    )
    workflow.add_edge("tools", "agent")
    return workflow.compile()

####### main ########
def run_agent(query: str, show_steps: bool = True):
    """
    Run the agent with a user query
    
    Args:
        query: User's question/request
        show_steps: Whether to show intermediate reasoning steps
    """
    app = create_agent_graph()
    initial_state = {
        "messages": [HumanMessage(content=query)]
    }
    
    print(f"\n{'='*60}")
    print(f"USER QUERY: {query}")
    print(f"{'='*60}\n")
    
    
    final_state = None
    step_count = 0
    
    for output in app.stream(initial_state):
        step_count += 1
        for key, value in output.items():
            final_state = value
            
            if show_steps:
                print(f"--- Step {step_count}: {key} ---")
                
                if "messages" in value:
                    for msg in value["messages"]:
                        # Show tool calls
                        if hasattr(msg, "tool_calls") and msg.tool_calls:
                            print("\n🔧 Tool Calls:")
                            for tool_call in msg.tool_calls:
                                print(f"  - {tool_call['name']}({tool_call['args']})")
                        
                        # tool results
                        elif isinstance(msg, ToolMessage):
                            print(f"\nTOOL_RESULT ({msg.name}):")
                            print(f"   {msg.content}")
                        
                        # agent thinking/response
                        elif hasattr(msg, "content") and isinstance(msg.content, str):
                            if not hasattr(msg, "tool_calls") or not msg.tool_calls:
                                print(f"\nAGENT_RESPONSE")
                                print(f"   {msg.content}")
                
                print()
    
    # final response
    if not show_steps and final_state:
        messages = final_state.get("messages", [])
        if messages:
            last_msg = messages[-1]
            if hasattr(last_msg, "content") and isinstance(last_msg.content, str):
                print(f"\n{last_msg.content}\n")
    
    print(f"{'='*60}\n")


def interactive_mode():
    """Run the agent in interactive chat mode"""
    print("\n" + "="*60)
    print("Customer Support Agent - Product Returns")
    print("="*60)
    print("\nAvailable Order IDs for testing:")
    for order_id, details in MOCK_PURCHASES.items():
        print(f"  - {order_id}: {details['product_name']} (purchased {details['purchase_date']})")
    print("\nType 'quit' to exit\n")
    
    # create the graph 
    app = create_agent_graph()
    
    # maintain conversation state
    conversation_messages = []
    
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\nThank you for using our support system! Goodbye! 👋\n")
            break
        
        if not user_input:
            continue
        
        # add user message
        conversation_messages.append(HumanMessage(content=user_input))
        
        # run agent with conversation history
        state = {"messages": conversation_messages}
        
        for output in app.stream(state):
            for key, value in output.items():
                if "messages" in value:
                    # add new messages to conversation
                    for msg in value["messages"]:
                        if msg not in conversation_messages:
                            conversation_messages.append(msg)
        
        # print the last message (agent's response)
        last_msg = conversation_messages[-1]
        if hasattr(last_msg, "content") and isinstance(last_msg.content, str):
            print(f"\nAgent: {last_msg.content}\n")



if __name__ == "__main__":
    os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("key not found")
        print()
    
    # example queries
    run_agent("I want to return my order ORD-001. Can I still return it?", show_steps=True)
    run_agent("Can I return order ORD-004? I'm not happy with it.", show_steps=True)
    run_agent("What is your return policy for electronics?", show_steps=True)
    # interactive_mode()