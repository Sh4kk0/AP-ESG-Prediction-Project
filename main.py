import sys
from src.data_loader import ESGDataLoader
from src.models import ESGModelExperiment


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()


def main():
    with open("results/metrics/metrics.txt", "w", encoding="utf-8") as f:
        original_stdout = sys.stdout
        sys.stdout = Tee(original_stdout, f)

        loader = ESGDataLoader(n_lags=6)
        df = loader.load_data()

        exp = ESGModelExperiment(chronological=False)
        exp.split(df)

        print("\n=== 4) Train ML Models ===")
        exp.train_all()

        print("\n=== 5) ESG Value Test (A/B - Global) ===")
        exp.ab_global()

        print("\n=== 6) ESG Value Test (A/B - By Sector) ===")
        exp.ab_by_sector(min_rows=200)

        sys.stdout = original_stdout  # restore


if __name__ == "__main__":
    main()
