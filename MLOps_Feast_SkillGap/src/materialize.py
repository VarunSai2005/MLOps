
from datetime import datetime, timedelta, timezone

from feast import FeatureStore


def main():
    store = FeatureStore(repo_path="feature_repo")

    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=3650)  # cover the whole snapshot

    print(f"Materializing features from {start_date} to {end_date} ...")
    store.materialize(start_date=start_date, end_date=end_date)
    print("Materialization complete. Online store updated (feature_repo/data/online_store.db).")


if __name__ == "__main__":
    main()
