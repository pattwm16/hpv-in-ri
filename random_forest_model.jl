# visualizing our models!
# use SciKitLearn and GraphViz
using CSV
using DataFrames
using ScikitLearn, Random, Statistics
@sk_import tree: (DecisionTreeClassifier, export_graphviz)
@sk_import ensemble: RandomForestClassifier
using PyPlot
using PyCall
@sk_import preprocessing: (LabelBinarizer, StandardScaler)
using Impute
using MLDataUtils
using PyPlot

# load in our dataset
nis_2017 = CSV.read("/Users/Will/Downloads/NIS_teen_data.csv") |> DataFrame
nis_2017_clean = nis_2017[.&(nis_2017[Symbol("HPVI_ANY")] .!= "DON'T KNOW",
                nis_2017[Symbol("HPVI_ANY")] .!= "REFUSED"),:] |> DataFrame

# split data into test and train sets
train, test = splitobs(nis_2017_clean, at = 0.7);

# predictors: determine who is UTD on HPV vaccinations
# define train labels and train features from split above
train_labels = string.(train[Symbol("HPVI_ANY")])
train_features = convert(DataFrame, train[:, [
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

# define testing labels and testing features from split above
test_labels = string.(test[Symbol("HPVI_ANY")])
test_features = convert(DataFrame, test[:, [
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

### Imputation ---
# recoding function below cannot deal with NA values in INCPORAR column
# we will use imputation to fill in those values
to_impute = Array(train_features.INCPORAR)

# define function replace_with_missing
# INPUT: an interatable data structure with NA values
# OUTPUT: an array of type Float64 with NA values replaced with missing
# NOTE: mutable push! is okay b/c out is redefined as empty at each call
function replace_with_missing(a)
    out = []
    for elem in a
        if elem != "NA"
            push!(out, parse(Float64, elem))
        else
            push!(out, missing)
        end
    end
    out
end

# after imputing, replace column in test and train features with imputed vals
train_features.INCPORAR = impute(replace_with_missing(Array(train_features.INCPORAR)), :fill; limit=0.2)
test_features.INCPORAR = impute(replace_with_missing(Array(test_features.INCPORAR)), :fill; limit=0.2)

### Flattening ---
# create the DataFrameMapper to reshape data
# I had difficulty here with the INCPORAR variable, see above for data cleaning
mapper = DataFrameMapper([(:RACE_K, LabelBinarizer()),
                          (:SEX, LabelBinarizer()),
                          (:INCPOV1, LabelBinarizer()),
                          (:RENT_OWN, LabelBinarizer()),
                          (:VISITS, LabelBinarizer()),
                          ([:AGE], StandardScaler()),
                          (:EDUC1, LabelBinarizer()),
                          ([:INCPORAR], StandardScaler()), # bad egg
                          (:MARITAL2, LabelBinarizer()),
                          (:INS_BREAK_I, LabelBinarizer()),
                          (:INS_STAT2_I, LabelBinarizer()),
                          ([:CKUP_AGE], StandardScaler()),
                          (:HPVI_RECOM, LabelBinarizer())]);

# apply the DataFrameMapper to training and testing features to flatten
train_dat_feat = fit_transform!(mapper, copy(train_features))
test_dat_feat = fit_transform!(mapper, copy(test_features))
convert(DataFrame, train_dat_feat)

# using these CSV files to try to count which features correspond to which
# numbers in output of DataFrameMapper
# CSV.write("First_elem.csv",  first(convert(DataFrame, train_dat_feat), 1), writeheader=false)
# CSV.write("First_feat.csv",  first(convert(DataFrame, train_features), 1), writeheader=false)

### Model Training ---
# reset the pygui
pygui(false)
pygui(true)

# using a Grid Search Algorithm to determine the best hyperparameters for
# our decision tree model
# we import the GridSearchCV module from SciKitLearn
using ScikitLearn.GridSearch: GridSearchCV

### apply GridSearchCV to our DecisionTreeClassifier ###

# GridSearch for best value for max_depth
# scan from 1-30 to prevent overfitting and print best param
gridsearch = GridSearchCV(DecisionTreeClassifier(max_features=12), Dict(:max_depth => 1:1:30))
fit!(gridsearch, train_dat_feat, train_labels)
println("Best parameters: $(gridsearch.best_params_)")

# GridSearch for best value for max_features
# scan from 1-13 (13 = total number of predictors) and print best value
gridsearch2 = GridSearchCV(DecisionTreeClassifier(max_depth=4), Dict(:max_features => 1:1:13))
fit!(gridsearch2, train_dat_feat, train_labels)
println("Best parameters: $(gridsearch2.best_params_)")

### apply GridSearchCV to our RandomForestClassifier ###

# GridSearch for best value for max_depth
# scan from 1-30 to prevent overfitting and print best param
gridsearch3 = GridSearchCV(RandomForestClassifier(max_features=12), Dict(:max_depth => 1:1:30))
fit!(gridsearch3, train_dat_feat, train_labels)
println("Best parameters: $(gridsearch3.best_params_)")

# GridSearch for best value for max_features
# scan from 1-13 (13 = total number of predictors) and print best value
gridsearch4 = GridSearchCV(RandomForestClassifier(max_depth=7), Dict(:max_features => 1:1:13))
fit!(gridsearch4, train_dat_feat, train_labels)
println("Best parameters: $(gridsearch4.best_params_)")

# create a plot of the cross validated score against tested values above
subplots_adjust(hspace=0.5)

# plot max_depth Decision Tree Classifier
subplot(221)
title("Decision Tree Classifier")
plot([cv_res.parameters[:max_depth] for cv_res in gridsearch.grid_scores_],
     [mean(cv_res.cv_validation_scores) for cv_res in gridsearch.grid_scores_])
xlabel("Max depth of tree")
ylabel("Cross Validated Accuracy")

# plot max_features Decision Tree Classifier
subplot(222)
title("Decision Tree Classifier")
plot([cv_res.parameters[:max_features] for cv_res in gridsearch2.grid_scores_],
         [mean(cv_res.cv_validation_scores) for cv_res in gridsearch2.grid_scores_])
xlabel("Max number of features")
ylabel("Cross Validated Accuracy")
suptitle("Using Grid Search Algorithm to Tune Hyperparameters")

# plot max_depth Random Forest Classifier
subplot(223)
title("Random Forest Classifier")
plot([cv_res.parameters[:max_depth] for cv_res in gridsearch3.grid_scores_],
     [mean(cv_res.cv_validation_scores) for cv_res in gridsearch3.grid_scores_])
xlabel("Max depth of tree")
ylabel("Cross Validated Accuracy")

# plot max_features Random Forest Classifier
subplot(224)
title("Random Forest Classifier")
plot([cv_res.parameters[:max_features] for cv_res in gridsearch4.grid_scores_],
         [mean(cv_res.cv_validation_scores) for cv_res in gridsearch4.grid_scores_])
xlabel("Max number of features")
ylabel("Cross Validated Accuracy")
suptitle("Using Grid Search Algorithm to Tune Hyperparameters")


# using best params from above, start cross validation of model
using ScikitLearn.CrossValidation: cross_val_score
using ScikitLearn.CrossValidation: cross_val_predict

# Fit classification model from above params
# max_depth = 3, max_features = 12
dectree = DecisionTreeClassifier(max_depth=3, max_features=12)
model = fit!(dectree, train_dat_feat, train_labels)
cross_val_score(model, train_dat_feat, train_labels; cv=10)

# random forest classifier
# max_depth = 5, max_features = 8, n_estimators=100
randforest = RandomForestClassifier(max_depth=5, max_features=8, n_estimators=100)
model2 = fit!(randforest, train_dat_feat, train_labels)
cross_val_score(model2, train_dat_feat, train_labels; cv=10)


# 10 fold cross validated accuracy for both models
predicted = cross_val_predict(model, test_dat_feat, test_labels, cv=10)
predicted2 = cross_val_predict(model2, test_dat_feat, test_labels, cv=10)

# import accuracy_score and apply to both models
# random forest won when I ran it (77.4% to 74.1%)
@pyimport sklearn.metrics as metrics
metrics.accuracy_score(test_labels, predicted) #74.1%
metrics.accuracy_score(test_labels, predicted2) # 77.4%

# get estimators from random forest model
estimator = model2.estimators_[5]

# export the estimators as a .dot file and vizualize
# could not get `using GraphViz` to work, had to vizualize using:
# https://dreampuf.github.io/GraphvizOnline/
export_graphviz(estimator, out_file="tree2.dot",
                #feature_names = mapper.features,
                class_names = ["Has recieved HPV shot", "Has NOT recieved HPV shot"],
                rounded = true, proportion = false,
                precision = 2, filled = true)

# select top 10 most important features from the model
importances = collect(enumerate(model2.feature_importances_))
top10importances = collect(Iterators.take(sort!(importances, by = x -> x[2], rev=true), 7))
imp_feat_str = Array([string(elem) for elem in [x[1] for x in top10importances]])

# reset the pygui
pygui(false)
pygui(true)

# plot the top 10 importances from most to least important
# TODO: figure out which feature numbers yield which features
grid("on")
bar(1:7, [y[2] for y in top10importances], width=0.8)
xticks(1:7, imp_feat_str)
axis("tight")
#title("Feature Importances")
xlabel("Feature")
ylabel("Importance")
