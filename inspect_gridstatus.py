import gridstatus
import inspect

iso = gridstatus.Ercot()
print("Signature of get_rtm_spp:")
try:
    print(inspect.signature(iso.get_rtm_spp))
except Exception as e:
    print(e)

print("\nHelp on get_rtm_spp:")
help(iso.get_rtm_spp)
