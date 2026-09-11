# Module 3 - AI Support Assistant

from typing import TypedDict
from langgraph.graph import StateGraph, START, END


class TicketState(TypedDict):
    ticket_text: str
    category: str
    response: str


BILLING_KEYWORDS = ["payment", "bill", "billing", "refund", "invoice"]
HR_KEYWORDS = ["salary", "leave", "employee", "hr", "payroll"]


# Classify the ticket
def classify(state: TicketState):
    text = state["ticket_text"].lower()

    if any(word in text for word in BILLING_KEYWORDS):
        return {"category": "billing"}

    elif any(word in text for word in HR_KEYWORDS):
        return {"category": "hr"}

    else:
        return {"category": "general"}


# Support handlers
def handle_billing(state: TicketState):
    return {"response": "Billing team will handle your request."}


def handle_hr(state: TicketState):
    return {"response": "HR team will handle your request."}


def handle_general(state: TicketState):
    return {"response": "General support team will handle your request."}


# Decide which handler to use
def route_ticket(state: TicketState):
    return state["category"]


# Build LangGraph
graph = StateGraph(TicketState)

graph.add_node("classify", classify)
graph.add_node("billing", handle_billing)
graph.add_node("hr", handle_hr)
graph.add_node("general", handle_general)

graph.add_edge(START, "classify")

graph.add_conditional_edges(
    "classify",
    route_ticket,
    {
        "billing": "billing",
        "hr": "hr",
        "general": "general"
    }
)

graph.add_edge("billing", END)
graph.add_edge("hr", END)
graph.add_edge("general", END)

# Compile the graph
app = graph.compile()

# Test all ticket categories

test_tickets = [
    "I need help with my payment",
    "I want to apply for leave",
    "I cannot login to my account"
]

for ticket in test_tickets:

    test_ticket = {
        "ticket_text": ticket,
        "category": "",
        "response": ""
    }

    result = app.invoke(test_ticket)

    print("\nTicket:", result["ticket_text"])
    print("Category:", result["category"])
    print("Response:", result["response"])
    # Save ticket results to log file

with open("support_assistant/docs/ticket_logs.txt", "w") as file:

    for ticket in test_tickets:

        test_ticket = {
            "ticket_text": ticket,
            "category": "",
            "response": ""
        }

        result = app.invoke(test_ticket)

        file.write(f"Ticket: {result['ticket_text']}\n")
        file.write(f"Category: {result['category']}\n")
        file.write(f"Response: {result['response']}\n")
        file.write("-" * 40 + "\n")

print("\nTicket logs saved successfully!")