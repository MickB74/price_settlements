import gridstatus.ercot
import inspect

try:
    print(inspect.getsource(gridstatus.ercot.Ercot.parse_doc))
except Exception as e:
    print(e)
