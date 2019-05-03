## the dataset can be found online at:
## https://www.cdc.gov/vaccines/imz-managers/nis/datasets-teen.html
## we used the 2017 data, the latest avaliable

# import packages
using DataFrames
using CSV
using DecisionTree
using MLDataUtils

# import our data
nis_2017 = CSV.read("/Users/Will/Downloads/NIS_teen_data.csv") |> DataFrame

# clean data to only include rows with YES/NO answer to HPVI_ANY
nis_2017_clean = nis_2017[.&(nis_2017[Symbol("HPVI_ANY")] .!= "DON'T KNOW", nis_2017[Symbol("HPVI_ANY")] .!= "REFUSED"),:] |> DataFrame

# define test and train data
train, test = splitobs(nis_2017_clean, at = 0.7);

# define HPVI_ANY as label
# levels: ["YES", "NO"]
labels = string.(train[Symbol("HPVI_ANY")])

# define features
features = convert(Array, train[:, [
                Symbol("RACE_K"),
                Symbol("SEX"),
                Symbol("INCPOV1"),
                Symbol("RENT_OWN"),
                Symbol("VISITS"),
                Symbol("AGE"),
                Symbol("EDUC1"),
                Symbol("INCPORAR"),
                Symbol("MARITAL2"),
                Symbol("INS_BREAK_I"),
                Symbol("INS_STAT2_I"),
                Symbol("CKUP_AGE"),
                Symbol("HPVI_RECOM") #fair predictor?
                ]])

# set of classification parameters and respective default values
# pruning_purity: purity threshold used for post-pruning (default: 1.0, no pruning)
# max_depth: maximum depth of the decision tree (default: -1, no maximum)
# min_samples_leaf: the minimum number of samples each leaf needs to have (default: 1)
# min_samples_split: the minimum number of samples in needed for a split (default: 2)
# min_purity_increase: minimum purity needed for a split (default: 0.0)
# n_subfeatures: number of features to select at random (default: 0, keep all)
n_subfeatures=0; max_depth=5; min_samples_leaf=10; min_samples_split=2
min_purity_increase=0.13; pruning_purity = 3.0

model    =   build_tree(labels, features,
                        n_subfeatures,
                        max_depth,
                        min_samples_leaf,
                        min_samples_split,
                        min_purity_increase)

accuracy = nfoldCV_tree(labels, features,
                        n_folds,
                        pruning_purity,
                        max_depth,
                        min_samples_leaf,
                        min_samples_split,
                        min_purity_increase)

test_labels = string.(test[Symbol("HPVI_ANY")])

# define features
test_features = convert(Array, test[:, [
                Symbol("RACE_K"),
                Symbol("SEX"),
                Symbol("INCPOV1"),
                Symbol("RENT_OWN"),
                Symbol("VISITS"),
                Symbol("AGE"),
                Symbol("EDUC1"),
                Symbol("INCPORAR"),
                Symbol("MARITAL2"),
                Symbol("INS_BREAK_I"),
                Symbol("INS_STAT2_I"),
                Symbol("CKUP_AGE"),
                Symbol("HPVI_RECOM") #fair predictor?
                ]])

apply_tree(model, test_features)

function accur(lom, lot)
    num_total = convert(Float64, length(lot))
    num_correct = length(filter(x -> x[1] == x[2], collect(zip(lom, lot))))
    println("Model is $((num_correct / num_total) * 100)% accurate on testing data")
end

accur(apply_tree(model, test_features), test_labels)

# Subsetting Data for Rhode Island Prediction
nis_2017_ri = nis_2017[(nis_2017[Symbol("STATE")] .== "RHODE ISLAND"),:] |> DataFrame

labels_ri = string.(nis_2017_ri[Symbol("HPVI_ANY")])
features_ri = convert(Array, nis_2017_ri[:, [
                Symbol("RACE_K"),
                Symbol("SEX"),
                Symbol("INCPOV1"),
                Symbol("RENT_OWN"),
                Symbol("VISITS"),
                Symbol("AGE"),
                Symbol("EDUC1"),
                Symbol("INCPORAR"),
                Symbol("MARITAL2"),
                Symbol("INS_BREAK_I"),
                Symbol("INS_STAT2_I"),
                Symbol("CKUP_AGE")#,
                #Symbol("HPVI_RECOM") #fair predictor?
                ]])

# train full-tree classifier
model_ri = build_tree(labels_ri, features_ri)

# prune tree: merge leaves having >= 90% combined purity (default: 100%)
model_ri = prune_tree(model_ri, 0.9)

# pretty print of the tree, to a depth of 5 nodes (optional)
print_tree(model_ri, 5)

# run 3-fold cross validation of pruned tree,
n_folds_ri=5
accuracy_ri = nfoldCV_tree(labels_ri, features_ri, n_folds_ri)

#principle components analysis
#random forest

# train random forest classifier
# predictors: determine who is UTD on HPV vaccinations
labels = string.(nis_2017_clean[Symbol("HPVI_ANY")])
features = convert(Array, nis_2017_clean[:, [
                Symbol("RACE_K"),
                Symbol("SEX"),
                Symbol("INCPOV1"),
                Symbol("RENT_OWN"),
                Symbol("VISITS"),
                Symbol("AGE"),
                Symbol("EDUC1"),
                Symbol("INCPORAR"),
                Symbol("MARITAL2"),
                Symbol("INS_BREAK_I"),
                Symbol("INS_STAT2_I"),
                Symbol("CKUP_AGE"),
                Symbol("HPVI_RECOM") #fair predictor?
                ]])
# using 2 random features, 10 trees, 0.5 portion of samples per tree, and a maximum tree depth of 6
model = build_forest(labels, features, 2, 10, 0.5, 6)
# apply learned model
apply_forest(model, [5.9,3.0,5.1,1.9])
# get the probability of each label
apply_forest_proba(model, [5.9,3.0,5.1,1.9], ["Iris-setosa", "Iris-versicolor", "Iris-virginica"])
# run 3-fold cross validation for forests, using 2 random features per split
n_folds=3; n_subfeatures=2
accuracy = nfoldCV_forest(labels, features, n_folds, n_subfeatures)

# Decision tree model cleaned to
# predictors: determine who is UTD on HPV vaccinations
labels = string.(nis_2017_clean[Symbol("HPVI_ANY")])
features = convert(Array, nis_2017_clean[:, [
                Symbol("RACE_K"),
                Symbol("SEX"),
                Symbol("INCPOV1"),
                Symbol("RENT_OWN"),
                Symbol("VISITS"),
                Symbol("AGE"),
                Symbol("EDUC1"),
                Symbol("INCPORAR"),
                Symbol("MARITAL2"),
                Symbol("INS_BREAK_I"),
                Symbol("INS_STAT2_I"),
                Symbol("CKUP_AGE"),
                Symbol("HPVI_RECOM") #fair predictor?
                ]])

# train full-tree classifier
model = build_tree(labels, features)

# prune tree: merge leaves having >= 90% combined purity (default: 100%)
model = prune_tree(model, 0.9)

# pretty print of the tree, to a depth of 5 nodes (optional)
#print_tree(model, 5)

# run 3-fold cross validation of pruned tree,
n_folds=10
accuracy = nfoldCV_tree(labels, features, n_folds)
