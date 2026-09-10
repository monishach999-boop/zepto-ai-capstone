# Zepto AI Capstone Project

This project is developed as part of the Zepto AI Capstone. It contains modules for data engineering, analytics, and an AI-based support assistant.

## Module 1 - Data Pipeline

The Data Pipeline module collects book product data from Books to Scrape and processes it into a clean and structured dataset.

### Data Source

Website used: Books to Scrape

The pipeline collects books from four categories:
- Travel
- Mystery
- Historical Fiction
- Sequential Art

The final dataset contains more than 60 books.

### Data Collected

For each book, the following information is collected:
- Title
- Price
- Rating
- Availability
- Category

### Data Cleaning

The scraped data is cleaned before storing:

- Book price is converted into a numeric GBP value.
- Star ratings such as One, Two, Three, Four, and Five are converted to integers from 1 to 5.
- Availability is converted into an `in_stock` boolean value.
- Missing or invalid numeric values are handled using median imputation.

### Currency Conversion

The project uses the fixed conversion rate required for the capstone:

**1 GBP = 105.50 INR**

The INR price is calculated as:

`price_inr = price_gbp × 105.50`

This is a fixed project-defined constant and does not use a live exchange-rate API.

### Database

The cleaned data is stored in a SQLite database using two normalized tables:

- `categories`
- `books`

The tables are connected using a primary key and foreign key relationship.

### SQL and Pandas

The project performs SQL queries demonstrating:
- SELECT
- WHERE
- ORDER BY
- LIMIT
- DISTINCT
- BETWEEN
- JOIN

SQL query results are also loaded into pandas DataFrames.

The SQL JOIN result is compared with a pandas `merge()` operation to verify that both approaches produce the same result.

## Installation

Create and activate a Python virtual environment, then install the required packages:

```bash
pip install -r requirements.txt