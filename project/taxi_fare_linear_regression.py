from metaflow import FlowSpec, step, card, conda_base, current, Parameter, Flow, trigger, project
from metaflow.cards import Markdown, Table, Image, Artifact

URL = "https://outerbounds-datasets.s3.us-west-2.amazonaws.com/taxi/latest.parquet"
DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"


@project(name="markaward")
@trigger(events=["s3"])
@conda_base(
    libraries={
        "pandas": "1.4.2",
        "pyarrow": "11.0.0",
        "numpy": "1.21.2",
        "scikit-learn": "1.1.2",
    },
    python="3.9.18",
)
class TaxiFarePrediction(FlowSpec):
    data_url = Parameter("data_url", default=URL)

    def transform_features(self, df):
        import pandas as pd

        mask = (
            (df.fare_amount > 0)
            & (df.trip_distance > 0)
            & (df.trip_distance <= 100)
        )
        columns = ["trip_distance", "hour", "airport_fee", "congestion_surcharge"]
        dummy_cols = ["VendorID", "RatecodeID"]
        fill_values = {
            "airport_fee": 0.0,
            "congestion_surcharge": 0.0,
        }
        label = ["total_amount"]
        return pd.get_dummies(
            df.loc[mask, columns + dummy_cols + label].reset_index(drop=True).fillna(fill_values),
            columns=dummy_cols,
            drop_first=True,
            dummy_na=True,
        )

    @step
    def start(self):
        import pandas as pd
        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import train_test_split

        self.df = self.transform_features(pd.read_parquet(self.data_url))
        self.X = self.df.drop(columns="total_amount").values
        self.y = self.df["total_amount"].values

        self.model = LinearRegression()
        self.next(self.validate)

    def gather_sibling_flow_run_results(self):
        # storage to populate and feed to a Table in a Metaflow card
        rows = []

        # loop through runs of this flow
        for run in Flow(self.__class__.__name__):
            if run.id != current.run_id:
                if not run.finished:
                    continue
                if run.successful:
                    icon = "✅"
                    msg = "OK"
                    score = str(run.data.scores.mean())
                else:
                    icon = "❌"
                    msg = "Error"
                    score = "NA"
                    for step in run:
                        for task in step:
                            if not task.successful:
                                msg = task.stderr
                row = [
                    Markdown(icon),
                    Artifact(run.id),
                    Artifact(run.created_at.strftime(DATETIME_FORMAT)),
                    Artifact(score),
                    Markdown(msg),
                ]
                rows.append(row)
            else:
                rows.append(
                    [
                        Markdown("✅"),
                        Artifact(run.id),
                        Artifact(run.created_at.strftime(DATETIME_FORMAT)),
                        Artifact(str(self.scores.mean())),
                        Markdown("This run..."),
                    ]
                )
        return rows

    @card(type="corise")
    @step
    def validate(self):
        from sklearn.model_selection import cross_val_score

        self.scores = cross_val_score(self.model, self.X, self.y, cv=5)
        current.card.append(Markdown("# Taxi Fare Prediction Results"))
        current.card.append(
            Table(
                self.gather_sibling_flow_run_results(),
                headers=["Pass/fail", "Run ID", "Created At", "R^2 score", "Stderr"],
            )
        )
        self.next(self.end)

    @step
    def end(self):
        print("Success!")
        print(f"Scores mean: {self.scores.mean()}")


if __name__ == "__main__":
    TaxiFarePrediction()
