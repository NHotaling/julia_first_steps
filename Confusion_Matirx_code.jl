#=
Pkg.add("VegaLite")
Pkg.add("DataFrames")
Pkg.add("Query")
Pkg.add("VegaDatasets")
Pkg.add("DataVoyager")
Pkg.add("Statistics")
=#

# install.packages("clues")
# Confusion Matrix Metrics
#       Actual Classes
#       P   N
#   P   TP  FP
#   N   FN  TN

pred_vals = ["A","B","B","A","A","A","A","A","A","A","A","A","B","B","A","B","A","A","B","B","B"]
real_vals = ["A","A","A","B","A","A","A","B","B","A","B","B","A","B","B","B","A","A","A","B","A"]

# Function to define a confusion matrix given two binary arrays
# The first binary array should be the "ground truth" or the non-model predicted values
# The second binary array should be the predicted values 
# The two arrays should be of equal length and only contain 2 or less classes

function conf_matrix(actuals, preds)
    if length(actuals) != length(preds) || length(unique(actuals)) < 3 || length(unique(preds)) < 3 # A few checks to ensure plugin runs smooth
        println("Arrays are not equal in length or more than binary classes")
    else # if above passes, we will sum values into a confusion matrix
        uniq_acts = unique(actuals) # Unique values for actual class (i.e. non-pred class)    
        TP = 0
        FN = 0
        TN = 0
        FP = 0
        
        # For each entry in the the actual vector check to see which values match the pred value 
        # at the same index. If first unique value of actual matches the value of the pred array
        # at the same index its a true positive (TP), if not its a false negative (FN).  
        for i in 1:length(actuals)   
            if actuals[i] == uniq_acts[1]
                if actuals[i] == preds[i] 
                    TP = TP+1
                else
                    FN = FN+1
                end
            else # if a value at an index in the actual array does not equal the first unique value of 
                 # the actuals array then its the actuals second class. If pred array at the same index
                 # equals this value then its a true negative (TN) other wise its a False Positive (FP)
                if actuals[i] == preds[i]
                    TN = TN +1
                else
                    FP = FP + 1
                end
            end
        end
        # Return an array of the four values summed above
        return [TP, FN, FP, TN]

    end
end

using Random

# Let's check performance of our code, let's see how long our confusion matrix calculation 
# would take to compare two 16MP images to eachother

# Set seed for a random initialization of two variables that will be 16M parameter arrays
Random.seed!(9);
act_vals = rand(0:1,(4096*4096))

Random.seed!(11);
mdl_vals = rand(0:1,(4096*4096))

# create a variable that will store the confusion matrix returned for use later
df = conf_matrix(act_vals, mdl_vals)

# Look at total time to execute
@time conf_matrix(act_vals, mdl_vals)

    