# Features to add
# Multithreading for confusion matrix function
# Conditional multithreading only for large arrays
# Multiple dispactch for different 
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

# Function to define a confusion matrix given two binary arrays. Binary is defined as 
# consisting of only two classes, i.e. could be 1/0, A/B, foo/bar, etc. 
# The first binary array should be the "ground truth" or the non-model predicted values
# The second binary array should be the predicted values 
# The two arrays should be of equal length and only contain 2 or less classes

# Inputs: 
# Two arrays with float64 data type

# Outputs: 
# 4x1 array representing a confusion matrix where 
# Value 1 is represents true positives
# value 2 represents false negative values
# Value 3 represents false positive values
# value 4 represents true negative values. 

function conf_matrix(actuals, preds)
    if length(actuals) != length(preds)  # A few checks to ensure plugin runs smooth
        println("Arrays are not equal in length or more than binary classes")
    elseif length(unique(actuals)) > 2  # A few checks to ensure plugin runs smooth
        println("Actual Value Array has more than binary classes")
    elseif length(unique(preds)) > 2  # A few checks to ensure plugin runs smooth
        println("Prediction Value Array has more than binary classes")
    else # if above passes, we will sum values into a confusion matrix
        uniq_acts = unique(actuals) # Unique values for actual class (i.e. non-pred class)    
        tp = Float64(0)
        fn = Float64(0)
        tn = Float64(0)
        fp = Float64(0)
        
        # For each entry in the the actual vector check to see which values match the pred value 
        # at the same index. If first unique value of actual matches the value of the pred array
        # at the same index its a true positive (TP), if not its a false negative (FN).  
        for i in 1:length(actuals)   
            if actuals[i] == uniq_acts[1]
                if actuals[i] == preds[i] 
                    tp = tp+1
                else
                    fn = fn+1
                end
            else # if a value at an index in the actual array does not equal the first unique value of 
                 # the actual array, then its the actual array's second class. If pred array at the same index
                 # equals this value then its a true negative (TN) other wise its a False Positive (FP)
                if actuals[i] == preds[i]
                    tn = tn +1
                else
                    fp = fp + 1
                end
            end
        end
        # Return an array of the four values summed above
        return [tp, fn, fp, tn]
    end
end

cwr = conf_matrix(real_vals, pred_vals)
sum(cwr)

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

typeof(df)

# Look at total time to execute
@time conf_matrix(act_vals, mdl_vals)




# Function to define a the sensitivity, recall, hit rate, or true positive rate 
# based on the definitions of the array positions from the confusion matrix function. 
# Sensitivity measures how many of class 1 an algorithm correctly predicted relative to the 
# number of class 1s there were in total in actuality.

# Inputs: 
# Two arrays with float64 data type

# Outputs: 
# a single float64 value representing the sensitivity 

function sensitivity(acts, prds)
    if length(acts) != length(prds)  # A few checks to ensure plugin runs smooth
        println("Arrays are not equal in length or more than binary classes")
    elseif length(unique(acts)) > 2  # A few checks to ensure plugin runs smooth
        println("Actual Value Array has more than binary classes")
    elseif length(unique(prds)) > 2  # A few checks to ensure plugin runs smooth
        println("Prediction Value Array has more than binary classes")
    else # if above passes, we will sum values into a confusion matrix
        cfm = conf_matrix(acts, prds)
        tp = cfm[1]
        fn = cfm[2]
        fp = cfm[3]
        tn = cfm[4]

        tpr = tp/(tp+fn)

        return(tpr)
    end
end


# Function to define a the precision or positive predictive value
# based on the definitions of the array positions from the confusion matrix function. 
# precision measures how many of class 1 an algorithm correctly predicted relative to the 
# number of class 1s there were in the total predicted. 

# Inputs: 
# Two arrays with float64 data type

# Outputs: 
# a single float64 value representing the precision 

function precision(acts, prds)
    if length(acts) != length(prds)  # A few checks to ensure plugin runs smooth
        println("Arrays are not equal in length or more than binary classes")
    elseif length(unique(acts)) > 2  # A few checks to ensure plugin runs smooth
        println("Actual Value Array has more than binary classes")
    elseif length(unique(prds)) > 2  # A few checks to ensure plugin runs smooth
        println("Prediction Value Array has more than binary classes")
    else # if above passes, we will sum values into a confusion matrix 

        cfm = conf_matrix(acts, prds)
        tp = cfm[1]
        fn = cfm[2]
        fp = cfm[3]
        tn = cfm[4]

        ppv = tp/(tp+fp)

        return(ppv)
    end
end


# Function to define a the specificity, selectivity or true negative rate
# based on the definitions of the array positions from the confusion matrix function. 
# specificity measures how many of class 2 an algorithm correctly predicted relative to the 
# number of class 2s there were in the total actually measured. 

# Inputs: 
# Two arrays with float64 data type

# Outputs: 
# a single float64 value representing the specificity 

function specificity(acts, prds)
    if length(acts) != length(prds)  # A few checks to ensure plugin runs smooth
        println("Arrays are not equal in length or more than binary classes")
    elseif length(unique(acts)) > 2  # A few checks to ensure plugin runs smooth
        println("Actual Value Array has more than binary classes")
    elseif length(unique(prds)) > 2  # A few checks to ensure plugin runs smooth
        println("Prediction Value Array has more than binary classes")
    else # if above passes, we will sum values into a confusion matrix
 
        cfm = conf_matrix(acts, prds)
        tp = cfm[1]
        fn = cfm[2]
        fp = cfm[3]
        tn = cfm[4]

        tnr = tn/(tn+fp)

        return(tnr)
    end
end


# Function to define a the negative predictive value (NPV)
# based on the definitions of the array positions from the confusion matrix function. 
# NPV measures how many of class 2 an algorithm correctly predicted relative to the 
# number of class 2s there were in the total predicted. 

# Inputs: 
# Two arrays with float64 data type

# Outputs: 
# a single float64 value representing the NPV

function neg_pred_val(acts, prds)
    if length(acts) != length(prds)  # A few checks to ensure plugin runs smooth
        println("Arrays are not equal in length or more than binary classes")
    elseif length(unique(acts)) > 2  # A few checks to ensure plugin runs smooth
        println("Actual Value Array has more than binary classes")
    elseif length(unique(prds)) > 2  # A few checks to ensure plugin runs smooth
        println("Prediction Value Array has more than binary classes")
    else # if above passes, we will sum values into a confusion matrix
 
        cfm = conf_matrix(acts, prds)
        tp = cfm[1]
        fn = cfm[2]
        fp = cfm[3]
        tn = cfm[4]

        npv = tn/(tn+fn)

        return(npv)
    end
end


# Function to define the miss rate or false negative rate (FNR)
# based on the definitions of the array positions from the confusion matrix function. 
# FNR measures how many of class 2 an algorithm incorrectly predicted relative to the 
# number of class 1s there were in the total actual. 

# Inputs: 
# Two arrays with float64 data type

# Outputs: 
# a single float64 value representing the FNR

function false_neg_rate(acts, prds)
    if length(acts) != length(prds)  # A few checks to ensure plugin runs smooth
        println("Arrays are not equal in length or more than binary classes")
    elseif length(unique(acts)) > 2  # A few checks to ensure plugin runs smooth
        println("Actual Value Array has more than binary classes")
    elseif length(unique(prds)) > 2  # A few checks to ensure plugin runs smooth
        println("Prediction Value Array has more than binary classes")
    else # if above passes, we will sum values into a confusion matrix
 
        cfm = conf_matrix(acts, prds)
        tp = cfm[1]
        fn = cfm[2]
        fp = cfm[3]
        tn = cfm[4]

        fnr = fn/(tp+fn)

        return(fnr)
    end
end


# Function to define the fallout or false positive rate (FPR)
# based on the definitions of the array positions from the confusion matrix function. 
# FPR measures how many of class 1 an algorithm incorrectly predicted relative to the 
# number of class 2s there were in the total actual. 

# Inputs: 
# Two arrays with float64 data type

# Outputs: 
# a single float64 value representing the FPR

function false_pos_rate(acts, prds)
    if length(acts) != length(prds)  # A few checks to ensure plugin runs smooth
        println("Arrays are not equal in length or more than binary classes")
    elseif length(unique(acts)) > 2  # A few checks to ensure plugin runs smooth
        println("Actual Value Array has more than binary classes")
    elseif length(unique(prds)) > 2  # A few checks to ensure plugin runs smooth
        println("Prediction Value Array has more than binary classes")
    else # if above passes, we will sum values into a confusion matrix
 
        cfm = conf_matrix(acts, prds)
        tp = cfm[1]
        fn = cfm[2]
        fp = cfm[3]
        tn = cfm[4]

        fpr = fp/(fp+tn)

        return(fpr)
    end
end


# Function to define the false discovery rate (FDR)
# based on the definitions of the array positions from the confusion matrix function. 
# FDR measures how many of class 1 an algorithm incorrectly predicted relative to the 
# number of class 1s there were in the total predicted count. 

# Inputs: 
# Two arrays with float64 data type

# Outputs: 
# a single float64 value representing the FDR

function false_disc_rate(acts, prds)
    if length(acts) != length(prds)  # A few checks to ensure plugin runs smooth
        println("Arrays are not equal in length or more than binary classes")
    elseif length(unique(acts)) > 2  # A few checks to ensure plugin runs smooth
        println("Actual Value Array has more than binary classes")
    elseif length(unique(prds)) > 2  # A few checks to ensure plugin runs smooth
        println("Prediction Value Array has more than binary classes")
    else # if above passes, we will sum values into a confusion matrix
 
        cfm = conf_matrix(acts, prds)
        tp = cfm[1]
        fn = cfm[2]
        fp = cfm[3]
        tn = cfm[4]

        fdr = fp/(fp+tp)

        return(fdr)
    end
end


# Function to define the false omission rate (FR)
# based on the definitions of the array positions from the confusion matrix function. 
# FR measures how many of class 2s an algorithm incorrectly predicted relative to the 
# number of class 2s there were in the total predicted count. 

# Inputs: 
# Two arrays with float64 data type

# Outputs: 
# a single float64 value representing the FR

function false_omis_rate(acts, prds)
    if length(acts) != length(prds)  # A few checks to ensure plugin runs smooth
        println("Arrays are not equal in length or more than binary classes")
    elseif length(unique(acts)) > 2  # A few checks to ensure plugin runs smooth
        println("Actual Value Array has more than binary classes")
    elseif length(unique(prds)) > 2  # A few checks to ensure plugin runs smooth
        println("Prediction Value Array has more than binary classes")
    else # if above passes, we will sum values into a confusion matrix
 
        cfm = conf_matrix(acts, prds)
        tp = cfm[1]
        fn = cfm[2]
        fp = cfm[3]
        tn = cfm[4]

        fr = fn/(fn+tn)

        return(fr)
    end
end


# Function to define the prevalence (prev)
# based on the definitions of the array positions from the confusion matrix function. 
# prev measures how many of class 2s an algorithm incorrectly predicted relative to the 
# number of class 2s there were in the total predicted count. 

# Inputs: 
# Two arrays with float64 data type

# Outputs: 
# a single float64 value representing the prev

function prevalence(acts, prds)
    if length(acts) != length(prds)  # A few checks to ensure plugin runs smooth
        println("Arrays are not equal in length or more than binary classes")
    elseif length(unique(acts)) > 2  # A few checks to ensure plugin runs smooth
        println("Actual Value Array has more than binary classes")
    elseif length(unique(prds)) > 2  # A few checks to ensure plugin runs smooth
        println("Prediction Value Array has more than binary classes")
    else # if above passes, we will sum values into a confusion matrix
 
        cfm = conf_matrix(acts, prds)
        tp = cfm[1]
        fn = cfm[2]
        fp = cfm[3]
        tn = cfm[4]
        ntot = sum(cfm)

        prev = (tp+fn)/ntot

        return(prev)
    end
end


# Function to define the accuracy or rand index (RI)
# based on the definitions of the array positions from the confusion matrix function. 
# RI measures how many of class 1 or class 2s an algorithm correctly predicted relative to the 
# total number of all measures. 

# Inputs: 
# Two arrays with float64 data type

# Outputs: 
# a single float64 value representing the RI

function accuracy(acts, prds)
    if length(acts) != length(prds)  # A few checks to ensure plugin runs smooth
        println("Arrays are not equal in length or more than binary classes")
    elseif length(unique(acts)) > 2  # A few checks to ensure plugin runs smooth
        println("Actual Value Array has more than binary classes")
    elseif length(unique(prds)) > 2  # A few checks to ensure plugin runs smooth
        println("Prediction Value Array has more than binary classes")
    else # if above passes, we will sum values into a confusion matrix
 
        cfm = conf_matrix(acts, prds)
        tp = cfm[1]
        fn = cfm[2]
        fp = cfm[3]
        tn = cfm[4]
        ntot = sum(cfm)

        ri = (tp+tn)/ntot

        return(ri)
    end
end


# Function to define the Jaccard Index, Threat Score, Critical Success Index, Intersection over Union (IoU)
# based on the definitions of the array positions from the confusion matrix function. 
# IoU measures how many of class 1s an algorithm correctly predicted relative to the 
# sum of correct class 1s, incorrect class 1s, and incorrect class 2s. 

# Inputs: 
# Two arrays with float64 data type

# Outputs: 
# a single float64 value representing the IoU

function jaccard(acts, prds)
    if length(acts) != length(prds)  # A few checks to ensure plugin runs smooth
        println("Arrays are not equal in length or more than binary classes")
    elseif length(unique(acts)) > 2  # A few checks to ensure plugin runs smooth
        println("Actual Value Array has more than binary classes")
    elseif length(unique(prds)) > 2  # A few checks to ensure plugin runs smooth
        println("Prediction Value Array has more than binary classes")
    else # if above passes, we will sum values into a confusion matrix
 
        cfm = conf_matrix(acts, prds)
        tp = cfm[1]
        fn = cfm[2]
        fp = cfm[3]
        tn = cfm[4]

        iou = tp/(tp+fp+fn)

        return(iou)
    end
end


# Function to define the Balanced Accuracy (BA)
# based on the definitions of the array positions from the confusion matrix function. 
# BA is the average of the ratio of true positives to total class 1s and true 
# negatives with total class 2s. 

# Inputs: 
# Two arrays with float64 data type

# Outputs: 
# a single float64 value representing the BA

function balanced_acc(acts, prds)
    if length(acts) != length(prds)  # A few checks to ensure plugin runs smooth
        println("Arrays are not equal in length or more than binary classes")
    elseif length(unique(acts)) > 2  # A few checks to ensure plugin runs smooth
        println("Actual Value Array has more than binary classes")
    elseif length(unique(prds)) > 2  # A few checks to ensure plugin runs smooth
        println("Prediction Value Array has more than binary classes")
    else # if above passes, we will sum values into a confusion matrix
 
        cfm = conf_matrix(acts, prds)
        tp = cfm[1]
        fn = cfm[2]
        fp = cfm[3]
        tn = cfm[4]

        ba = (0.5*tp/(tp+fn)) + (0.5*tn/(tn+fp))

        return(ba)
    end
end


# Function to define the general form of F-Scores
# based on the definitions of the array positions from the confusion matrix function. 
# FScore is a weighted form of IoU where we take the weighted ratio of true poitives   
# over the weighted ratio of true positives + false negatives + false positives

# Inputs: 
# Two arrays with float64 data type and one float64 number

# Outputs: 
# a single float64 value representing the FScore at the fval level

function fscore(acts, prds, fval)
    if length(acts) != length(prds)  # A few checks to ensure plugin runs smooth
        println("Arrays are not equal in length or more than binary classes")
    elseif length(unique(acts)) > 2  # A few checks to ensure plugin runs smooth
        println("Actual Value Array has more than binary classes")
    elseif length(unique(prds)) > 2  # A few checks to ensure plugin runs smooth
        println("Prediction Value Array has more than binary classes")
    else # if above passes, we will sum values into a confusion matrix
 
        cfm = conf_matrix(acts, prds)
        tp = cfm[1]
        fn = cfm[2]
        fp = cfm[3]
        tn = cfm[4]

        fscore = ((1+fval^2)*tp)/((1+fval^2)*tp + (fval^2)*fn + fp)

        return(fscore)
    end
end


# Function to define the F1-Score or dice index (DI)
# based on the definitions of the array positions from the confusion matrix function. 
# DI is a weighted form of IoU where we take the weighted ratio of true poitives   
# over the weighted ratio of true positives + false negatives + false positives

# Inputs: 
# Two arrays with float64 data type

# Outputs: 
# a single float64 value representing the DI

function dice_index(acts, prds)
    if length(acts) != length(prds)  # A few checks to ensure plugin runs smooth
        println("Arrays are not equal in length or more than binary classes")
    elseif length(unique(acts)) > 2  # A few checks to ensure plugin runs smooth
        println("Actual Value Array has more than binary classes")
    elseif length(unique(prds)) > 2  # A few checks to ensure plugin runs smooth
        println("Prediction Value Array has more than binary classes")
    else # if above passes, we will sum values into a confusion matrix
 
        cfm = conf_matrix(acts, prds)
        tp = cfm[1]
        fn = cfm[2]
        fp = cfm[3]
        tn = cfm[4]
        fval = 1

        di = ((1+fval^2)*tp)/((1+fval^2)*tp + (fval^2)*fn + fp)

        return(di)
    end
end


# Function to define the prevalence threshold (PT)
# based on the definitions of the array positions from the confusion matrix function. 
# PT is a weighted form of IoU where we take the weighted ratio of true poitives   
# over the weighted ratio of true positives + false negatives + false positives

# Inputs: 
# Two arrays with float64 data type

# Outputs: 
# a single float64 value representing the PT

function prev_thresh(acts, prds)
    if length(acts) != length(prds)  # A few checks to ensure plugin runs smooth
        println("Arrays are not equal in length or more than binary classes")
    elseif length(unique(acts)) > 2  # A few checks to ensure plugin runs smooth
        println("Actual Value Array has more than binary classes")
    elseif length(unique(prds)) > 2  # A few checks to ensure plugin runs smooth
        println("Prediction Value Array has more than binary classes")
    else # if above passes, we will sum values into a confusion matrix
 
        cfm = conf_matrix(acts, prds)
        tp = cfm[1]
        fn = cfm[2]
        fp = cfm[3]
        tn = cfm[4]

        tnr = tn/(tn+fp)
        tpr = tp/(tp+fn)

        pt = ((tnr-1)+sqrt(tpr*(1-tnr)))/(tpr+tnr-1)

        return(pt)
    end
end


# Function to define the Pearson's phi coefficient, Yule phi coefficient, or Matthews Correlation Coefficient (MCC)
# based on the definitions of the array positions from the confusion matrix function. 
# MCC is a weighted form of IoU where we take the weighted ratio of true poitives   
# over the weighted ratio of true positives + false negatives + false positives

# Inputs: 
# Two arrays with float64 data type

# Outputs: 
# a single float64 value representing the MCC

function mathews(acts, prds)
    if length(acts) != length(prds)  # A few checks to ensure plugin runs smooth
        println("Arrays are not equal in length or more than binary classes")
    elseif length(unique(acts)) > 2  # A few checks to ensure plugin runs smooth
        println("Actual Value Array has more than binary classes")
    elseif length(unique(prds)) > 2  # A few checks to ensure plugin runs smooth
        println("Prediction Value Array has more than binary classes")
    else # if above passes, we will sum values into a confusion matrix
 
        cfm = conf_matrix(acts, prds)
        tp = cfm[1]
        fn = cfm[2]
        fp = cfm[3]
        tn = cfm[4]

        mcc = ((tp*tn)-(fp*fn))/(√((tp+fp)*(tp+fn)*(tn+fn)*(tn+fp)))
        
        return(mcc)
    end
end


# Function to define the Fowlkes-Mallows Index (FMI)
# based on the definitions of the array positions from the confusion matrix function. 
# FMI is the square root of the precision times the  sensitivity  
# 

# Inputs: 
# Two arrays with float64 data type

# Outputs: 
# a single float64 value representing the FMI

function fowlkes(acts, prds)
    if length(acts) != length(prds)  # A few checks to ensure plugin runs smooth
        println("Arrays are not equal in length or more than binary classes")
    elseif length(unique(acts)) > 2  # A few checks to ensure plugin runs smooth
        println("Actual Value Array has more than binary classes")
    elseif length(unique(prds)) > 2  # A few checks to ensure plugin runs smooth
        println("Prediction Value Array has more than binary classes")
    else # if above passes, we will sum values into a confusion matrix
 
        cfm = conf_matrix(acts, prds)
        tp = cfm[1]
        fn = cfm[2]
        fp = cfm[3]
        tn = cfm[4]

        fmi = √((tp/(tp+fp))*(tp/(tp+fn)))
        
        return(fmi)
    end
end


# Function to define the Informedness, Bookermaker Informedness (BI)
# based on the definitions of the array positions from the confusion matrix function. 
# BI is the sensitivity plus the specificity minus 1  
# 

# Inputs: 
# Two arrays with float64 data type

# Outputs: 
# a single float64 value representing the BI

function informedness(acts, prds)
    if length(acts) != length(prds)  # A few checks to ensure plugin runs smooth
        println("Arrays are not equal in length or more than binary classes")
    elseif length(unique(acts)) > 2  # A few checks to ensure plugin runs smooth
        println("Actual Value Array has more than binary classes")
    elseif length(unique(prds)) > 2  # A few checks to ensure plugin runs smooth
        println("Prediction Value Array has more than binary classes")
    else # if above passes, we will sum values into a confusion matrix
 
        cfm = conf_matrix(acts, prds)
        tp = cfm[1]
        fn = cfm[2]
        fp = cfm[3]
        tn = cfm[4]

        bi = (tp/(tp+fn))+(tn/(tn+fp))-1
        
        return(bi)
    end
end


# Function to define the Markedness (MKN)
# based on the definitions of the array positions from the confusion matrix function. 
# MKN is the Precision plus the Negative Predictive Value minus 1  
# 

# Inputs: 
# Two arrays with float64 data type

# Outputs: 
# a single float64 value representing the MKN

function markedness(acts, prds)
    if length(acts) != length(prds)  # A few checks to ensure plugin runs smooth
        println("Arrays are not equal in length or more than binary classes")
    elseif length(unique(acts)) > 2  # A few checks to ensure plugin runs smooth
        println("Actual Value Array has more than binary classes")
    elseif length(unique(prds)) > 2  # A few checks to ensure plugin runs smooth
        println("Prediction Value Array has more than binary classes")
    else # if above passes, we will sum values into a confusion matrix
 
        cfm = conf_matrix(acts, prds)
        tp = cfm[1]
        fn = cfm[2]
        fp = cfm[3]
        tn = cfm[4]

        mkn = (tp/(tp+fp))+(tn/(tn+fn))-1
        
        return(mkn)
    end
end


# Function to define the Cohen's Kappa (CK)
# based on the definitions of the array positions from the confusion matrix function. 
# CK is the is a statistic that is used to measure inter-rater reliability 
# (and also intra-rater reliability) for qualitative (categorical) items

# Inputs: 
# Two arrays with float64 data type

# Outputs: 
# a single float64 value representing the CK

function cohenk(acts, prds)
    if length(acts) != length(prds)  # A few checks to ensure plugin runs smooth
        println("Arrays are not equal in length or more than binary classes")
    elseif length(unique(acts)) > 2  # A few checks to ensure plugin runs smooth
        println("Actual Value Array has more than binary classes")
    elseif length(unique(prds)) > 2  # A few checks to ensure plugin runs smooth
        println("Prediction Value Array has more than binary classes")
    else # if above passes, we will sum values into a confusion matrix
 
        cfm = conf_matrix(acts, prds)
        tp = cfm[1]
        fn = cfm[2]
        fp = cfm[3]
        tn = cfm[4]
        ntot = sum(cfm)
        ri = (tp+tn)/ntot
        norm_prd_pos = (tp+fp)/ntot
        norm_act_pos = (tp+fn)/ntot
        norm_prd_neg = (tn+fn)/ntot
        norm_act_neg = (fp+tn)/ntot
        ck_num = ri-(norm_prd_pos*norm_act_pos + norm_prd_neg*norm_act_neg)
        ck_denom = 1-(norm_prd_pos*norm_act_pos + norm_prd_neg*norm_act_neg)

        ck = ck_num/ck_denom
        
        return(ck)
    end
end


# Function to define the Mirkin Metric (MM)
# based on the definitions of the array positions from the confusion matrix function. 
# MM is an adujested rand index accounting for total N but not random chance

# Inputs: 
# Two arrays with float64 data type

# Outputs: 
# a single float64 value representing the MM

function mirkin(acts, prds)
    if length(acts) != length(prds)  # A few checks to ensure plugin runs smooth
        println("Arrays are not equal in length or more than binary classes")
    elseif length(unique(acts)) > 2  # A few checks to ensure plugin runs smooth
        println("Actual Value Array has more than binary classes")
    elseif length(unique(prds)) > 2  # A few checks to ensure plugin runs smooth
        println("Prediction Value Array has more than binary classes")
    else # if above passes, we will sum values into a confusion matrix
 
        cfm = conf_matrix(acts, prds)
        tp = cfm[1]
        fn = cfm[2]
        fp = cfm[3]
        tn = cfm[4]
        ntot = sum(cfm)
        ri = (tp+tn)/ntot

        mm = ntot*(ntot-1)*(1-ri)
        
        return(mm)
    end
end


# Function to define the Adujested Mirkin Metric (AMM)
# based on the definitions of the array positions from the confusion matrix function. 
# AMM is the mirkin metric divided by the square of the total count. 

# Inputs: 
# Two arrays with float64 data type

# Outputs: 
# a single float64 value representing the AMM

function mirkin_adj(acts, prds)
    if length(acts) != length(prds)  # A few checks to ensure plugin runs smooth
        println("Arrays are not equal in length or more than binary classes")
    elseif length(unique(acts)) > 2  # A few checks to ensure plugin runs smooth
        println("Actual Value Array has more than binary classes")
    elseif length(unique(prds)) > 2  # A few checks to ensure plugin runs smooth
        println("Prediction Value Array has more than binary classes")
    else # if above passes, we will sum values into a confusion matrix
 
        cfm = conf_matrix(acts, prds)
        tp = cfm[1]
        fn = cfm[2]
        fp = cfm[3]
        tn = cfm[4]
        ntot = sum(cfm)
        ri = (tp+tn)/ntot

        amm = (ntot*(ntot-1)*(1-ri))/(ntot*ntot)
        
        return(amm)
    end
end


# Function to define the Adujested rand index (ARI)
# based on the definitions of the array positions from the confusion matrix function. 
# ARI is the rand index adjusted for chance across all class combinations

# Inputs: 
# Two arrays with float64 data type

# Outputs: 
# a single float64 value representing the ARI

function rand_adj(acts, prds)
    if length(acts) != length(prds)  # A few checks to ensure plugin runs smooth
        println("Arrays are not equal in length or more than binary classes")
    elseif length(unique(acts)) > 2  # A few checks to ensure plugin runs smooth
        println("Actual Value Array has more than binary classes")
    elseif length(unique(prds)) > 2  # A few checks to ensure plugin runs smooth
        println("Prediction Value Array has more than binary classes")
    else # if above passes, we will sum values into a confusion matrix
 
        cfm = conf_matrix(acts, prds)
        tp = BigInt(cfm[1])
        fn = BigInt(cfm[2])
        fp = BigInt(cfm[3])
        tn = BigInt(cfm[4])
        ntot = BigInt(sum(cfm))

        ari = ((binomial(tp, 2) + binomial(tn, 2) + binomial(fp, 2) + binomial(fn, 2)) - 
        (((binomial((tp+fp), 2)+binomial((fn+tn), 2))*(binomial((tp+fn), 2)+binomial((fp+tn), 2)))/
        (binomial(ntot, 2))))/
        (0.5*((binomial((tp+fp), 2)+binomial((fn+tn), 2))+(binomial((tp+fn), 2)+binomial((fp+tn), 2))) - 
        ((binomial((tp+fp), 2)+binomial((fn+tn), 2))*(binomial((tp+fn), 2)+binomial((fp+tn), 2)))/
        binomial(ntot, 2))
        
        ari= convert(Float64, ari)

        return(ari)
    end
end

# Function to define an array of all function values above into a single function call
# this will be called, all confusion matrix metrics (ACMM). These will all be calculated based 
# on the definitions of the array positions from the confusion matrix function. 
# ARI is the rand index adjusted for chance across all class combinations

# Inputs: 
# Two arrays with float64 data type

# Outputs: 
# a single float64 value representing the ARI

function all_cmm(acts, prds) 
    if length(acts) != length(prds)  # A few checks to ensure plugin runs smooth
        println("Arrays are not equal in length or more than binary classes")
    elseif length(unique(acts)) > 2  # A few checks to ensure plugin runs smooth
        println("Actual Value Array has more than binary classes")
    elseif length(unique(prds)) > 2  # A few checks to ensure plugin runs smooth
        println("Prediction Value Array has more than binary classes")
    else # if above passes, we will sum values into a confusion matrix
 
        cfm = conf_matrix(acts, prds)
        tp = BigInt(cfm[1])
        fn = BigInt(cfm[2])
        fp = BigInt(cfm[3])
        tn = BigInt(cfm[4])
        ntot = BigInt(sum(cfm))
        ri = (tp+tn)/ntot

        norm_prd_pos = (tp+fp)/ntot
        norm_act_pos = (tp+fn)/ntot
        norm_prd_neg = (tn+fn)/ntot
        norm_act_neg = (fp+tn)/ntot
        ck_num = ri-(norm_prd_pos*norm_act_pos + norm_prd_neg*norm_act_neg)
        ck_denom = 1-(norm_prd_pos*norm_act_pos + norm_prd_neg*norm_act_neg)

        tpr = tp/(tp+fn)
        ppv = tp/(tp+fp)
        tnr = tn/(tn+fp)
        npv = tn/(tn+fn)
        fnr = fn/(tp+fn)
        fpr = fp/(fp+tn)
        fdr = fp/(fp+tp)
        fr = fn/(fn+tn)
        prev = (tp+fn)/ntot
        iou = tp/(tp+fp+fn)
        ba = (0.5*tp/(tp+fn)) + (0.5*tn/(tn+fp))
        f05score = ((1+0.5^2)*tp)/((1+0.5^2)*tp + (0.5^2)*fn + fp)
        di = ((1+1^2)*tp)/((1+1^2)*tp + (1^2)*fn + fp)
        f2score = ((1+2^2)*tp)/((1+2^2)*tp + (2^2)*fn + fp)
        pt = ((tnr-1)+√(tpr*(1-tnr)))/(tpr+tnr-1)
        mcc = ((tp*tn)-(fp*fn))/(√((tp+fp)*(tp+fn)*(tn+fn)*(tn+fp)))
        fmi = √((tp/(tp+fp))*(tp/(tp+fn)))
        bi = (tp/(tp+fn))+(tn/(tn+fp))-1
        mkn = (tp/(tp+fp))+(tn/(tn+fn))-1
        ck = ck_num/ck_denom
        mm = ntot*(ntot-1)*(1-ri)
        amm = (ntot*(ntot-1)*(1-ri))/(ntot*ntot)
        ari = ((binomial(tp, 2) + binomial(tn, 2) + binomial(fp, 2) + binomial(fn, 2)) - 
                (((binomial((tp+fp), 2)+binomial((fn+tn), 2))*(binomial((tp+fn), 2)+binomial((fp+tn), 2)))/
                (binomial(ntot, 2))))/
                (0.5*((binomial((tp+fp), 2)+binomial((fn+tn), 2))+(binomial((tp+fn), 2)+binomial((fp+tn), 2))) - 
                ((binomial((tp+fp), 2)+binomial((fn+tn), 2))*(binomial((tp+fn), 2)+binomial((fp+tn), 2)))/
                binomial(ntot, 2))


        acmm = [tpr, ppv, tnr, npv, fnr, fpr, fdr, fr, prev, ri, iou, ba, f05score, di, f2score, pt, mcc, fmi, bi, mkn, ck, mm, amm, ari]
        acmm = convert(Array{Float64,1}, acmm)
        return(acmm)
    end
end


# Look at total time to execute
println(@time conf_matrix(act_vals, mdl_vals))

println(@time all_cmm(act_vals, mdl_vals))

println(@time rand_adj(act_vals, mdl_vals))

println(@time accuracy(act_vals, mdl_vals))