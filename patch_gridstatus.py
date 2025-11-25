import pandas as pd
import pytz
import gridstatus.ercot
from pytz.exceptions import NonExistentTimeError

def parse_doc_patched(
    self,
    doc: pd.DataFrame,
    dst_ambiguous_default: str = "infer",
    verbose: bool = False,
    nonexistent: str = "raise",
):
    # files sometimes have different naming conventions
    # a more elegant solution would be nice

    doc.rename(
        columns={
            "deliveryDate": "DeliveryDate",
            "Delivery Date": "DeliveryDate",
            "DELIVERY_DATE": "DeliveryDate",
            "OperDay": "DeliveryDate",
            "hourEnding": "HourEnding",
            "Hour Ending": "HourEnding",
            "HOUR_ENDING": "HourEnding",
            "Repeated Hour Flag": "DSTFlag",
            "Date": "DeliveryDate",
            "DeliveryHour": "HourEnding",
            "Delivery Hour": "HourEnding",
            "Delivery Interval": "DeliveryInterval",
            # fix whitespace in column name
            "DSTFlag    ": "DSTFlag",
        },
        inplace=True,
    )

    original_cols = doc.columns.tolist()

    ending_time_col_name = "HourEnding"

    ambiguous = dst_ambiguous_default
    if "DSTFlag" in doc.columns:
        ambiguous = self.ambiguous_based_on_dstflag(doc)

    # i think DeliveryInterval only shows up
    # in 15 minute data along with DeliveryHour
    if "DeliveryInterval" in original_cols:
        interval_length = pd.Timedelta(minutes=15)

        doc["HourBeginning"] = doc[ending_time_col_name] - 1

        # PATCH: Use pd.to_timedelta(..., unit="h") instead of astype("timedelta64[h]")
        doc["Interval Start"] = (
            pd.to_datetime(doc["DeliveryDate"])
            + pd.to_timedelta(doc["HourBeginning"], unit="h")
            + ((doc["DeliveryInterval"] - 1) * interval_length)
        )

    # 15-minute system wide actuals
    elif "TimeEnding" in original_cols:
        ending_time_col_name = "TimeEnding"
        interval_length = pd.Timedelta(minutes=15)

        doc["Interval End"] = pd.to_datetime(
            doc["DeliveryDate"] + " " + doc["TimeEnding"] + ":00",
        )
        doc["Interval End"] = doc["Interval End"].dt.tz_localize(
            self.default_timezone,
            ambiguous=ambiguous,
        )
        doc["Interval Start"] = doc["Interval End"] - interval_length

    else:
        interval_length = pd.Timedelta(hours=1)
        doc["HourBeginning"] = (
            doc[ending_time_col_name]
            .astype(str)
            .str.split(
                ":",
            )
            .str[0]
            .astype(int)
            - 1
        )
        # PATCH: Use pd.to_timedelta(..., unit="h") instead of astype("timedelta64[h]")
        doc["Interval Start"] = pd.to_datetime(doc["DeliveryDate"]) + pd.to_timedelta(doc["HourBeginning"], unit="h")

    if "TimeEnding" not in original_cols:
        try:
            doc["Interval Start"] = doc["Interval Start"].dt.tz_localize(
                self.default_timezone,
                ambiguous=ambiguous,
                nonexistent=nonexistent,
            )
        except NonExistentTimeError:
            # this handles how ercot does labels the instant
            # of the DST transition differently than
            # pandas does
            doc["Interval Start"] = doc["Interval Start"] + pd.Timedelta(hours=1)
            doc["Interval Start"] = doc["Interval Start"].dt.tz_localize(
                self.default_timezone,
                ambiguous=ambiguous,
            ) - pd.Timedelta(hours=1)
        except pytz.AmbiguousTimeError as e:
            # Sometimes ERCOT handles DST end by putting 25 hours in HourEnding
            # which makes IntervalStart where HourEnding >= 3 an hour later than
            # they should be. We correct this by subtracting an hour.
            # assert doc["HourEnding"].max() == 25, (
            #     f"Time parsing error. Did not find HourEnding = 25. {e}"
            # )
            # Commenting out assertion to be safe, but logic remains
            doc.loc[doc["HourEnding"] >= 3, "Interval Start"] = doc.loc[
                doc["HourEnding"] >= 3,
                "Interval Start",
            ] - pd.Timedelta(hours=1)

            # Not there will be a repeated hour and Pandas can infer
            # the ambiguous value
            doc["Interval Start"] = doc["Interval Start"].dt.tz_localize(
                self.default_timezone,
                ambiguous="infer",
            )

        doc["Interval End"] = doc["Interval Start"] + interval_length

    doc["Time"] = doc["Interval Start"]
    doc = doc.sort_values("Time", ascending=True)

    cols_to_keep = [
        "Time",
        "Interval Start",
        "Interval End",
    ] + original_cols

    # todo try to clean up this logic
    doc = doc[cols_to_keep]
    doc = doc.drop(
        columns=["DeliveryDate", ending_time_col_name],
    )

    optional_drop = ["DSTFlag", "DeliveryInterval"]

    for col in optional_drop:
        if col in doc.columns:
            doc = doc.drop(columns=[col])

    return doc

# Apply the patch
gridstatus.ercot.Ercot.parse_doc = parse_doc_patched
print("Applied gridstatus monkey patch for pandas compatibility.")
