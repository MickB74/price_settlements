import gridstatus
import inspect

iso = gridstatus.Ercot()
print("Signature of _get_documents:")
try:
    print(inspect.signature(iso._get_documents))
except Exception as e:
    print(e)
