using Pkg

# Activate the current directory
Pkg.activate(".")

# Turn the current directory into a Julia package
Pkg.generate("BNPStep")

# Add required dependencies
Pkg.add(["DataFrames", "Plots", "CUDA", "Distributions"])

# Instantiate the environment to ensure all dependencies are installed
Pkg.instantiate()