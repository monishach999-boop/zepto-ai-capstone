import requests
from bs4 import BeautifulSoup
import pandas as pd
import sqlite3
from pathlib import Path


# -------------------------------------------------
# 1. PROJECT PATH
# -------------------------------------------------

folder = Path("data_pipeline")
folder.mkdir(exist_ok=True)


# -------------------------------------------------
# 2. CATEGORIES TO SCRAPE
# -------------------------------------------------

categories = {
    "Travel": "travel_2",
    "Mystery": "mystery_3",
    "Historical Fiction": "historical-fiction_4",
    "Sequential Art": "sequential-art_5"
}


# Store all books here
all_books = []


# -------------------------------------------------
# 3. SCRAPE BOOK DATA
# -------------------------------------------------

print("Starting book scraping...\n")

for category, link in categories.items():

    url = (
        "https://books.toscrape.com/catalogue/"
        f"category/books/{link}/index.html"
    )

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        books = soup.find_all(
            "article",
            class_="product_pod"
        )

        print(category, "books found:", len(books))

        for book in books:

            title = book.h3.a["title"]

            price = book.find(
                "p",
                class_="price_color"
            ).text

            rating = book.find(
                "p",
                class_="star-rating"
            )["class"][1]

            availability = book.find(
                "p",
                class_="instock availability"
            ).text.strip()

            all_books.append({
                "title": title,
                "price": price,
                "star_rating": rating,
                "availability": availability,
                "category": category
            })

    except requests.RequestException as error:
        print("Could not load:", category)
        print("Reason:", error)


# -------------------------------------------------
# 4. CREATE DATAFRAME
# -------------------------------------------------

df = pd.DataFrame(all_books)

print("\nTotal books collected:", len(df))
print("Total categories:", df["category"].nunique())


# -------------------------------------------------
# 5. CLEAN PRICE
# -------------------------------------------------

# Remove £ and unwanted encoding characters
df["price"] = (
    df["price"]
    .str.replace("£", "", regex=False)
    .str.replace("Â", "", regex=False)
)

df["price_gbp"] = pd.to_numeric(
    df["price"],
    errors="coerce"
)

# If any price could not be parsed,
# fill it using the median price
df["price_gbp"] = df["price_gbp"].fillna(
    df["price_gbp"].median()
)


# -------------------------------------------------
# 6. CONVERT RATING TO INTEGER
# -------------------------------------------------

rating_map = {
    "One": 1,
    "Two": 2,
    "Three": 3,
    "Four": 4,
    "Five": 5
}

df["rating"] = df["star_rating"].map(rating_map)

# Handle unexpected missing rating
if df["rating"].isna().any():

    rating_median = df["rating"].median()

    df["rating"] = df["rating"].fillna(
        rating_median
    )

df["rating"] = df["rating"].astype(int)


# -------------------------------------------------
# 7. CONVERT AVAILABILITY TO BOOLEAN
# -------------------------------------------------

df["in_stock"] = (
    df["availability"]
    .str.lower()
    .str.contains("in stock")
)


# -------------------------------------------------
# 8. GBP TO INR
# -------------------------------------------------

# Fixed project-defined conversion rate
GBP_TO_INR = 105.50

df["price_inr"] = (
    df["price_gbp"] * GBP_TO_INR
).round(2)


# -------------------------------------------------
# 9. KEEP FINAL REQUIRED COLUMNS
# -------------------------------------------------

df = df[
    [
        "title",
        "price_gbp",
        "price_inr",
        "rating",
        "in_stock",
        "category"
    ]
]


# -------------------------------------------------
# 10. SAVE CLEAN CSV
# -------------------------------------------------

csv_path = folder / "books_clean.csv"

df.to_csv(
    csv_path,
    index=False
)

print("\nClean CSV saved:", csv_path)

print("\nFirst 5 cleaned rows:")
print(df.head())


# -------------------------------------------------
# 11. CREATE SQLITE DATABASE
# -------------------------------------------------

db_path = folder / "zepto_books.db"

# Delete old database so every run starts clean
if db_path.exists():
    db_path.unlink()

connection = sqlite3.connect(db_path)

cursor = connection.cursor()

# Enable foreign keys
cursor.execute("PRAGMA foreign_keys = ON")


# -------------------------------------------------
# 12. CREATE NORMALIZED TABLES
# -------------------------------------------------

cursor.execute("""
CREATE TABLE categories (
    category_id INTEGER PRIMARY KEY AUTOINCREMENT,
    category_name TEXT UNIQUE NOT NULL
)
""")


cursor.execute("""
CREATE TABLE books (
    book_id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    price_gbp REAL,
    price_inr REAL,
    rating INTEGER,
    in_stock INTEGER,
    category_id INTEGER,
    FOREIGN KEY (category_id)
        REFERENCES categories(category_id)
)
""")


# -------------------------------------------------
# 13. INSERT CATEGORIES
# -------------------------------------------------

unique_categories = df["category"].unique()

for category in unique_categories:

    cursor.execute(
        """
        INSERT OR IGNORE INTO categories(category_name)
        VALUES (?)
        """,
        (category,)
    )

connection.commit()


# -------------------------------------------------
# 14. GET CATEGORY IDs
# -------------------------------------------------

category_table = pd.read_sql_query(
    "SELECT * FROM categories",
    connection
)

category_id_map = dict(
    zip(
        category_table["category_name"],
        category_table["category_id"]
    )
)


# -------------------------------------------------
# 15. INSERT BOOKS
# -------------------------------------------------

for _, row in df.iterrows():

    cursor.execute(
        """
        INSERT INTO books
        (
            title,
            price_gbp,
            price_inr,
            rating,
            in_stock,
            category_id
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            row["title"],
            float(row["price_gbp"]),
            float(row["price_inr"]),
            int(row["rating"]),
            int(row["in_stock"]),
            category_id_map[row["category"]]
        )
    )

connection.commit()

print("\nSQLite database created:", db_path)


# -------------------------------------------------
# 16. SQL QUERIES
# -------------------------------------------------

queries = {

    "Query 1 - All books":
    """
    SELECT *
    FROM books
    """,

    "Query 2 - Five star books":
    """
    SELECT title, rating
    FROM books
    WHERE rating = 5
    """,

    "Query 3 - Top 10 expensive books":
    """
    SELECT title, price_inr
    FROM books
    ORDER BY price_inr DESC
    LIMIT 10
    """,

    "Query 4 - Distinct categories":
    """
    SELECT DISTINCT category_name
    FROM categories
    """,

    "Query 5 - Books with category":
    """
    SELECT
        b.title,
        b.price_gbp,
        b.price_inr,
        b.rating,
        b.in_stock,
        c.category_name
    FROM books b
    JOIN categories c
        ON b.category_id = c.category_id
    """,

    "Query 6 - INR price between range":
    """
    SELECT title, price_inr
    FROM books
    WHERE price_inr BETWEEN 2000 AND 5000
    ORDER BY price_inr DESC
    """
}


# -------------------------------------------------
# 17. RUN AND SAVE SQL QUERY OUTPUTS
# -------------------------------------------------

output_path = folder / "sql_query_outputs.txt"

with open(
    output_path,
    "w",
    encoding="utf-8"
) as file:

    for name, query in queries.items():

        result = pd.read_sql_query(
            query,
            connection
        )

        print("\n", name)
        print(result.head(10))

        file.write(
            "\n" + "=" * 70 + "\n"
        )

        file.write(name + "\n")

        file.write(
            "=" * 70 + "\n"
        )

        file.write(query.strip() + "\n\n")

        file.write(
            result.to_string(index=False)
        )

        file.write("\n")


print(
    "\nSQL queries and outputs saved:",
    output_path
)


# -------------------------------------------------
# 18. READ SQL RESULTS INTO PANDAS
# -------------------------------------------------

five_star_df = pd.read_sql_query(
    queries["Query 2 - Five star books"],
    connection
)

top_expensive_df = pd.read_sql_query(
    queries["Query 3 - Top 10 expensive books"],
    connection
)

print("\nFive-star books:")
print(five_star_df.head())

print("\nTop expensive books:")
print(top_expensive_df.head())


# -------------------------------------------------
# 19. SQL JOIN RESULT
# -------------------------------------------------

sql_join_df = pd.read_sql_query(
    queries["Query 5 - Books with category"],
    connection
)


# -------------------------------------------------
# 20. SAME JOIN USING pd.merge()
# -------------------------------------------------

books_table = pd.read_sql_query(
    "SELECT * FROM books",
    connection
)

categories_table = pd.read_sql_query(
    "SELECT * FROM categories",
    connection
)

pandas_join_df = pd.merge(
    books_table,
    categories_table,
    on="category_id"
)

pandas_join_df = pandas_join_df[
    [
        "title",
        "price_gbp",
        "price_inr",
        "rating",
        "in_stock",
        "category_name"
    ]
]


# -------------------------------------------------
# 21. VERIFY BOTH JOIN RESULTS
# -------------------------------------------------

sql_compare = (
    sql_join_df
    .sort_values("title")
    .reset_index(drop=True)
)

pandas_compare = (
    pandas_join_df
    .sort_values("title")
    .reset_index(drop=True)
)

same_result = sql_compare.equals(
    pandas_compare
)

print(
    "\nSQL JOIN and pandas merge match:",
    same_result
)


# -------------------------------------------------
# 22. FINAL SUMMARY
# -------------------------------------------------

print("\n-----------------------------")
print("MODULE 1 COMPLETE")
print("-----------------------------")

print(
    "Books collected:",
    len(df)
)

print(
    "Categories:",
    df["category"].nunique()
)

print(
    "Missing values:",
    df.isnull().sum().sum()
)

print(
    "Fixed GBP to INR rate:",
    GBP_TO_INR
)

print(
    "Database:",
    db_path
)

print(
    "SQL/Pandas JOIN Match:",
    same_result
)


connection.close()