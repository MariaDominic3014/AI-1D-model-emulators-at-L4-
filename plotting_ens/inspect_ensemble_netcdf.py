from netCDF4 import Dataset

ds = Dataset("data/processed_ensemble.nc")
#print(ds)

# List dimensions
print(ds.dimensions.keys())

print("\nVariables:")
for name in ds.variables:
    print(name, ds.variables[name].shape)
    print(ds.variables[name][:3])



# List variable names
#print(ds.variables.keys())

# Look at one variable
#print(ds.variables["temp"])

# Look at the shape
#print(ds.variables["temp"].shape)

# Read a few values
#print(ds.variables["temp"][:5])