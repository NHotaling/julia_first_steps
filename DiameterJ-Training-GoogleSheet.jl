#=
Pkg.add("GoogleSheetsCSVExporter")
Pkg.add("DataFrames")
Pkg.add("CSV")
=#

# for dataframe shortcuts - https://jcharistech.wordpress.com/julia-dataframes-cheat-sheets/

using GoogleSheetsCSVExporter, CSV, DataFrames

# Define sheet we want to access
url = ""

# Export the CSV and create a dataframe of the sheet
all_unique_logins_file = GoogleSheetsCSVExporter.fromURI(url) |> CSV.File |> DataFrame

head(all_unique_logins_file)

# Find number of rows and the number of unique IDs
nrow(all_unique_logins_file)
length(unique(all_unique_logins_file[:ID]))

