# Module 3 - AI Support Assistant

This module implements a deterministic AI support ticket routing system using LangGraph. It classifies incoming support tickets and routes them to the appropriate support team without using any external LLM or API.

## Files

- `app.py` - Contains the LangGraph support ticket classification and routing workflow.
- `ticket_logs.txt` - Stores the results generated from sample support tickets.

## Ticket Categories

The assistant classifies tickets into three categories:

- Billing
- HR
- General

## Classification Logic

The system uses predefined keywords to classify support tickets.

### Billing Keywords

payment, bill, billing, refund, invoice

### HR Keywords

salary, leave, employee, hr, payroll

If no Billing or HR keyword is found, the ticket is classified as General.

## LangGraph Workflow

The workflow follows:

Ticket Input → Classification → Conditional Routing → Support Handler → Response

Three handler nodes are used:

- `handle_billing` - Handles billing-related tickets.
- `handle_hr` - Handles HR-related tickets.
- `handle_general` - Handles all other support tickets.

## Sample Results

### Billing Ticket

Ticket: I need help with my payment  
Category: billing  
Response: Billing team will handle your request.

### HR Ticket

Ticket: I want to apply for leave  
Category: hr  
Response: HR team will handle your request.

### General Ticket

Ticket: I cannot login to my account  
Category: general  
Response: General support team will handle your request.

## Logging

The processed ticket results are saved in:

`ticket_logs.txt`

Each log contains the ticket text, predicted category, and generated response.

## Conclusion

Module 3 successfully demonstrates deterministic support ticket classification and conditional routing using LangGraph without requiring an external LLM or API.