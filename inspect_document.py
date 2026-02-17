import gridstatus
import pandas as pd

iso = gridstatus.Ercot()
docs = iso._get_documents(
    report_type_id=12301,
    published_after=pd.Timestamp.now(tz="US/Central") - pd.Timedelta(days=5)
)

if docs:
    d = docs[0]
    print("Document attributes:")
    print(dir(d))
    print(f"\nSample doc: {d}")
else:
    print("No docs found to inspect")
