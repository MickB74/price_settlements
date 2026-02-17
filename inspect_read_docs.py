import gridstatus
import inspect

iso = gridstatus.Ercot()
print("Signature of read_docs:")
try:
    print(inspect.signature(iso.read_docs))
except Exception as e:
    print(e)
