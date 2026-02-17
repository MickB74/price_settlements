import gridstatus
iso = gridstatus.Ercot()
print("Methods related to 'doc':")
print([m for m in dir(iso) if 'doc' in m.lower()])
