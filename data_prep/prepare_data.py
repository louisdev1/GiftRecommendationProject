import pandas as pd
import os

# ----------- ABSOLUTE BASE PATH -----------
BASE_PATH = r"C:\Users\louis\OneDrive - Thomas More\Thomas-More\Semester_5\GiftRecommendationProject"

DATA_PATH = os.path.join(BASE_PATH, "data")
OUTPUT_PATH = os.path.join(BASE_PATH, "recommender", "Recommender")

# ----------- LOAD CSV FILES -----------
users = pd.read_csv(os.path.join(DATA_PATH, "users.csv"))
products = pd.read_csv(os.path.join(DATA_PATH, "products.csv"))
interactions = pd.read_csv(os.path.join(DATA_PATH, "interactions.csv"))

# ----------- MAP STRING IDS → INT IDS -----------
user_map = {u: i for i, u in enumerate(users["user_id"].unique())}
product_map = {p: i for i, p in enumerate(products["product_id"].unique())}

interactions["user_idx"] = interactions["user_id"].map(user_map)
interactions["product_idx"] = interactions["product_id"].map(product_map)

# ----------- EXPORT FOR C# -----------
os.makedirs(OUTPUT_PATH, exist_ok=True)

interactions[["user_idx", "product_idx", "rating"]].to_csv(
    os.path.join(OUTPUT_PATH, "ratings.csv"),
    index=False
)

pd.DataFrame.from_dict(user_map, orient="index", columns=["user_idx"]).to_csv(
    os.path.join(OUTPUT_PATH, "user_map.csv")
)

pd.DataFrame.from_dict(product_map, orient="index", columns=["product_idx"]).to_csv(
    os.path.join(OUTPUT_PATH, "product_map.csv")
)

print("✔ Data prepared successfully for C# Matrix Factorization")
