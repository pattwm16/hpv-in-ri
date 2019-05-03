<img src="http://blackstonevalleyprep.org/wp-content/uploads/2016/04/RI-map.png" align="right" height=200/>

# HPV Vaccination Access in Rhode Island

## Creating a Classification Model
> What are the best predictors to help a model classify an individual as having started the HPV vaccination schedule or not?
> What is the best type of classification model to perform this analysis?

We tested a **decision tree classification model** and **random forest classification model** to parse our data. We ultimately decided on our random forest classification model, as it had the highest cross validated accuracy score in predicting our data. The final model can be found in [`random_forest_model.jl`](random_forest_model.jl).

##### To run it yourself:

1. :floppy_disk:  - Download the data from [here](https://www.cdc.gov/vaccines/imz-managers/nis/datasets-teen.html "NIS Teen Datasets")
2. :open_file_folder: - Run [`NIS_input_statements.R`](Data/NIS_input_statements.R) and export as CSV
3. :pencil2: - Change the path in line 16 of [`random_forest_model.jl`](random_forest_model.jl) to the path of the CSV from above
4. :running: - Run the code!

### Data Source

Our data came from the [National Immunization Survey](https://www.cdc.gov/vaccines/imz-managers/nis/datasets-teen.html "NIS Teen Datasets") conducted by the CDC. We used the latest data from 2017 in our model - though further extensions of this project could include other previous years to see which predictors remain salient.

The CDC provided a script to load labelled data into R which was modified to fit our needs ([`NIS_input_statements.R`](Data/NIS_input_statements.R)). The loaded data were then exported in CSV format which could be accessed in Julia.

### Label
Our label was `HPVI_ANY` - "Has the teen ever received any HPV shots"? Though there exist other variables that parse the specific stage in HPV immunization, the brand of immunization received, or the valency of the shot (trivalent vs. nonavalent), we decided to first determine predictors that are associated with non-entry into the HPV vaccination cascade. All `NA`, `Don't Know`, and `Refused` responses were omitted from variable classification.


### Predictors
After a literature review, we decided on thirteen predictors to include in our model. These are:

* `RACE_K`: Race of teen with multi-race category
* `SEX`: Sex of teen
* `INCPOV1`: Poverty status
    * This measure was based on the 2016 Census poverty thresholds.
* `RENT_OWN`: Is home owned/being bought, rented, or occupied by some other arrangement?
* `VISITS`: In the past 12 months, what is the number of times the teen has seen a doctor or other health care professional?
* `AGE`: Age of teen
* `EDUC1`: Education level of mother
* `INCPORAR`: Income to poverty ratio
    * This measure was based on the 2016 Census poverty thresholds.
* `MARITAL2`: Marital status of mother
* `INS_BREAK_I`: Continuity of insurance coverage since age 11
* `INS_STAT2_I`: Insurance status (Private, Medicaid, Other, Uninsured)
* `CKUP_AGE`: Age in years at last check-up
* `HPVI_RECOM`: Had or has a doctor or other health care professional ever recommended that the teen receive HPV shots?

### Data Preparation

#### Imputation
Additionally, the `INCPORAR` data contained a large number of `NA` data values (N = 3475/39237) and numerical values as coded as strings. `replace_with_missing()` was created to transform those values into `missing` type.

```julia
# INPUT: an iterable data structure with NA values
# OUTPUT: an array of type Float64 with NA values replaced with missing
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
```
The resulting array contained a number of NA values. Using `Impute.jl`, I performed mean imputation on the missing data.

#### Flattening
In order to use the categorical data in a classification analysis, each variable was flattened. For instance, `RACE_K` has three possible values: `WHITE ONLY`, `BLACK ONLY`, or `OTHER + MIXED RACE`. Each of these options was coded separately as a dichotomous variable (e.g. `RACE_K_WHITE_ONLY`, `RACE_K_BLACK_ONLY`, `RACE_K_OTHER_+_MIXED_RACE`). This was achieved using the `DataFrameMapper` function on [line 88](https://github.com/bcbi-edu-methods/vax-access/blob/8b1fe3f7bdd75e0fdbc01cf28b4e2703a06aaaef/random_forest_model.jl#L88).

### Next Steps

- [x] Downloaded and labelled data from NIS website
- [x] Cleaned data
- [x] Implemented decision tree classifier using `DecisionTree.jl`
- [x] Implemented decision tree classifier using `ScikitLearn.jl`
- [x] Implemented random forest classifier using `ScikitLearn.jl`
- [x] Used Grid Search algorithm to tune hyperparameters for both models
- [x] Cross validated (k=10) and tested models
- [x] Visualized final decision tree using GraphViz
- [x] Identified ten most important factors for prediction
- [ ] Consider more variables in our random forest model
- [ ] Perform principle component analysis on variable set to reduce dimensionality (using `MultivariateStats.jl`)
- [ ] Consider a clustering analysis or latent class analysis to break down subgroups of those who haven not been vaccinated to highlight important factors
- [ ] Using this framework, determine important factors between people who have initiated the HPV vaccination and those who have completed the vaccination schedule

### Known Bugs and Limitations
#### Imputation
`INCPORAR` had a number of missing values (N = 3,475/39,237). I performed mean imputation using `Impute.jl` under the assumption that the data are missing at random. This has some drawbacks - particularly biasing the standard error of the value.

#### Unable to parse transformed features
`DataFrameMapper` was used to flatten non-dichotomous variables. In the python implementation of `DataFrameMapper`, the object has an attribute `.transformed_names_` with the transformed feature names (see [here](https://github.com/scikit-learn-contrib/sklearn-pandas#output-features-names)). As far as I can tell `ScikitLearn.jl` provides `DataFrameMapper`, but does not yet have this field attribute.

The current fields can be queried as follows:
```julia
using ScikitLearn
fieldnames(DataFrameMapper)
```
This returns `(:features, :sparse, :missing2NaN, :output_type)`, which is currently missing a `.transformed_names_` field.

We performed a feature importance analysis of our random forest model.
