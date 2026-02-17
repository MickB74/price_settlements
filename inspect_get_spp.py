import gridstatus
import inspect

iso = gridstatus.Ercot()
print("Signature of get_spp:")
try:
    print(inspect.signature(iso.get_spp))
except Exception as e:
    print(e)
