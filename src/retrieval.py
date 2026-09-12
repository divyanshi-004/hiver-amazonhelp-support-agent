from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "amazonhelp_direct_interactions.csv"
)


# ============================================================
# HISTORICAL RETRIEVER
# ============================================================

class AmazonHelpRetriever:
    """
    Retrieves historically similar AmazonHelp customer-support
    conversations using TF-IDF + cosine similarity.
    """

    def __init__(
        self,
        data_path=DATA_PATH,
        max_features=50000,
    ):
        self.data_path = Path(data_path)
        self.max_features = max_features

        self.data = None
        self.vectorizer = None
        self.matrix = None

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    def load_data(self):
        print("Loading AmazonHelp historical conversations...")

        self.data = pd.read_csv(
            self.data_path,
            usecols=[
                "customer_tweet_id",
                "customer_text",
                "amazon_reply",
            ],
        )

        self.data = self.data.dropna(
            subset=["customer_text", "amazon_reply"]
        )

        self.data["customer_text"] = (
            self.data["customer_text"]
            .astype(str)
            .str.strip()
        )

        self.data["amazon_reply"] = (
            self.data["amazon_reply"]
            .astype(str)
            .str.strip()
        )

        self.data = self.data[
            (self.data["customer_text"] != "")
            & (self.data["amazon_reply"] != "")
        ].reset_index(drop=True)

        print(
            f"Loaded {len(self.data):,} usable "
            "historical conversations."
        )

    # --------------------------------------------------------
    # BUILD INDEX
    # --------------------------------------------------------

    def build_index(self):
        if self.data is None:
            self.load_data()

        print("Building TF-IDF retrieval index...")

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            max_features=self.max_features,
            min_df=2,
        )

        self.matrix = self.vectorizer.fit_transform(
            self.data["customer_text"]
        )

        print(
            f"Index ready: {self.matrix.shape[0]:,} "
            f"documents × {self.matrix.shape[1]:,} features."
        )

    # --------------------------------------------------------
    # RETRIEVE SIMILAR CASES
    # --------------------------------------------------------

    def retrieve(
        self,
        query,
        top_k=5,
    ):
        if self.vectorizer is None or self.matrix is None:
            self.build_index()

        if not query or not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        query_vector = self.vectorizer.transform(
            [query]
        )

        similarities = cosine_similarity(
            query_vector,
            self.matrix,
        ).flatten()

        top_indices = similarities.argsort()[
            ::-1
        ][:top_k]

        results = []

        for index in top_indices:

            row = self.data.iloc[index]

            results.append(
                {
                    "customer_tweet_id": row[
                        "customer_tweet_id"
                    ],
                    "historical_customer": row[
                        "customer_text"
                    ],
                    "historical_amazon_reply": row[
                        "amazon_reply"
                    ],
                    "similarity": round(
                        float(similarities[index]),
                        4,
                    ),
                }
            )

        return results


# ============================================================
# CLI TEST
# ============================================================

def main():

    retriever = AmazonHelpRetriever()

    retriever.build_index()

    print("\n" + "=" * 70)
    print("AMAZONHELP HISTORICAL RETRIEVAL TEST")
    print("=" * 70)

    query = input(
        "\nEnter a customer message:\n> "
    ).strip()

    results = retriever.retrieve(
        query,
        top_k=5,
    )

    print("\n" + "=" * 70)
    print("TOP HISTORICAL MATCHES")
    print("=" * 70)

    for number, result in enumerate(
        results,
        start=1,
    ):

        print(
            f"\n[{number}] "
            f"Similarity: {result['similarity']}"
        )

        print(
            "\nHistorical customer:"
        )

        print(
            result["historical_customer"]
        )

        print(
            "\nHistorical AmazonHelp reply:"
        )

        print(
            result["historical_amazon_reply"]
        )

        print("-" * 70)


if __name__ == "__main__":
    main()