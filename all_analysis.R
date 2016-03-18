library(ggplot2)
library(data.table)
library(dplyr)
library(caret)
library(boot)
library(glmnet) #glmnet
library(klaR) # Naive bayes
library(kernlab) # SVM
library(randomForest)
library(ROCR)

# Set theme
theme_set(theme_gray(base_size = 18))


######################  OUTLINE ##########################

# 1. DATA LOAD AND TYPE MANAGEMENT
# 2. BASIC EDA
# 3. MORE EDA
# 4. REDUCE FRAME AND PREPARE FOR ML
# 5. CLASSIFICATION ML


#################### 1. DATA LOAD AND TYPE MANAGEMENT ##########################

location = '/Users/nicholasculver/repos/Udacity/R/final'
setwd(location)

chosen_columns = c('ListingKey', 'LoanStatus','Term','ProsperScore','EmploymentStatus',
                   'CreditScoreRangeLower','MonthlyLoanPayment',
                   'BorrowerRate','BorrowerAPR','LoanOriginalAmount','EstimatedLoss',
                   'BorrowerState','IsBorrowerHomeowner','LoanOriginationDate',
                   'CurrentCreditLines','IncomeRange','StatedMonthlyIncome',
                   'DebtToIncomeRatio','LoanMonthsSinceOrigination')

# Bring in data and brose columns
core <- fread('prosper.csv',stringsAs = FALSE, data.table=TRUE,
              select=chosen_columns)

# Datatype adjustments
core$LoanStatus <- as.factor(core$LoanStatus)
core$Term <- as.factor(core$Term)
core$BorrowerAPR <- as.numeric(core$BorrowerAPR)
core$BorrowerRate <- as.numeric(core$BorrowerRate)
core$ProsperScore <- as.factor(core$ProsperScore)
core$LoanOriginalAmount <- as.numeric(core$LoanOriginalAmount)
core$MonthlyLoanPayment <- as.numeric(core$MonthlyLoanPayment)
core$EstimatedLoss <- as.numeric(core$EstimatedLoss) # This is presumably a percentage
core$IsBorrowerHomeowner <- as.logical(core$IsBorrowerHomeowner)
core$EmploymentStatus <- as.factor(core$EmploymentStatus)
core$CreditScoreRangeLower <- as.numeric(core$CreditScoreRangeLower) 
core$DebtToIncomeRatio <- as.numeric(core$DebtToIncomeRatio)
core$LoanMonthsSinceOrigination <- as.numeric(core$LoanMonthsSinceOrigination)
core$LoanOriginationDate <- as.Date(core$LoanOriginationDate,
                                    format = '%Y-%m-%d %H:%M:%S')
core$CurrentCreditLines <- as.numeric(core$CurrentCreditLines)
core$IncomeRange <- as.factor(core$IncomeRange)
core$StatedMonthlyIncome <- as.numeric(core$StatedMonthlyIncome)

# Check out NAs for each column
NA_counts <- apply(core, 2, function(x) sum(is.na(x)))
NA_counts

# Create a new yes/no/current default variable
core$Default <- 'Current'
core$Default[core$LoanStatus %in% c('Completed','FinalPaymentInProgress')] <- 'No'
core$Default[core$LoanStatus %in% c('Chargedoff','Defaulted')] <- 'Yes'
table(core$Default)




################### 2. BASIC EDA ###########################

# Most common monthly payment is around 175 dollars
ggplot(data=core,
       aes(x=MonthlyLoanPayment)) +
  geom_histogram(binwidth=25) +
  scale_x_continuous(limits = c(0,1000), 
                     breaks=seq(0,1000,100))

# Some common loan totals but overal quite wide range
ggplot(data=core,
       aes(x=LoanOriginalAmount)) +
  geom_histogram(binwidth=500) +
  scale_x_continuous(limits = c(0,30000), 
                     breaks=seq(0,30000,3000))

# Jesus these are high loan rates - 
ggplot(data=core,
       aes(x=BorrowerRate)) +
  geom_histogram(binwidth=.01) +
  scale_x_continuous(limits = c(0,0.4), 
                     breaks=seq(0,0.4,.025))

# Credit score - 
ggplot(data=core,
       aes(x=CreditScoreRangeLower)) +
  geom_histogram(binwidth = 25) +
  scale_x_continuous(limits = c(400,900), 
                     breaks=seq(400,900,50))

# Almost all employed, so not much useful there. could restrict to just employed.
ggplot(data=core,
       aes(x=EmploymentStatus)) +
  geom_bar()

# Very few 12 month loans, could cut those out
ggplot(data=core,
       aes(x=Term)) +
  geom_bar()

# Most borrowers were not given prosper scores - they are blank rather than NA
ggplot(data=core,
       aes(x=ProsperScore)) +
  geom_bar()
NA_counts['ProsperScore']

# Critical column here
ggplot(data=core,
       aes(x=LoanStatus)) +
  geom_bar()

# Total credit lines - very similar distribution as open and current credit lines
ggplot(data=core,
       aes(x=TotalCreditLinespast7years)) +
  geom_histogram(binwidth=1)
table(core$TotalCreditLinespast7years)
NA_counts['TotalCreditLinespast7years']

# There are five major income tiers - others can be dropped
ggplot(data=core,
       aes(x=IncomeRange)) +
  geom_bar()

# There was a gap around late 2008 to mid 2009 when no loans were given out
ggplot(data=core,
       aes(x=LoanOriginationDate, y=StatedMonthlyIncome)) +
  geom_point(alpha=0.2) +
  ylim(0,25000) +
  geom_smooth()

# We've have strict credit score criteria, which have tightned over the years
ggplot(data= subset(core, CreditScoreRangeLower >0),
       aes(x=LoanOriginationDate, y=CreditScoreRangeLower)) +
  geom_jitter(alpha=0.1)

# 8% seems to be standard loss, not much variation across states
mean(core$EstimatedLoss[!is.na(core$EstimatedLoss)])
loss_by_state <- core[!is.na(EstimatedLoss),mean(EstimatedLoss),by=BorrowerState]




############### 3. MORE EDA ######################################

# This explains "not displayed" values in income range - it was before 
# income levels were bothered with
ggplot(data=core,
       aes(x=LoanOriginationDate, y=StatedMonthlyIncome, group=IncomeRange, 
           col=IncomeRange)) +
  geom_point(alpha=0.2) +
  ylim(0,25000) +
  geom_smooth()

# Homeowners make more money - got it
ggplot(data=core,
       aes(x=LoanOriginationDate, y=StatedMonthlyIncome, group=IsBorrowerHomeowner, 
           col=IsBorrowerHomeowner)) +
  geom_point(alpha=0.2) +
  ylim(0,25000) +
  geom_smooth()

# Term 60 is a new thing for company, introduced in 2011
# Also, loan rates have been set differently since 2011 - more standardized
ggplot(data=core[Term!=12],
       aes(x=LoanOriginationDate, y=BorrowerRate, group=Term, 
           col=Term)) +
  geom_point(alpha=0.2) +
  ylim(0,0.4) +
  geom_smooth()

# Homeowners get better loan rates - got it
ggplot(data=core,
       aes(x=LoanOriginationDate, y=BorrowerRate, group=IsBorrowerHomeowner, 
           col=IsBorrowerHomeowner)) +
  geom_point(alpha=0.2) +
  ylim(0,0.4) +
  geom_smooth()

# Let's graph borrower rate against income over time
# First we need to re-arrange the factors so our legend is coherent
old_levels <- levels(core$IncomeRange)
new_levels <- c(old_levels[2],old_levels[4],old_levels[5],old_levels[6],
                old_levels[3],old_levels[1],old_levels[7],old_levels[8])
core$IncomeRange <- factor(core$IncomeRange, levels=new_levels)

# Wow - utterly clear that higher income means lower borrow rate
ggplot(data=core[!IncomeRange %in% c('$0','Not displayed','Not employed')],
       aes(x=LoanOriginationDate, y=BorrowerRate, group=IncomeRange, 
           col=IncomeRange)) +
  geom_point(alpha=0.2) +
  ylim(0,0.4) +
  geom_smooth()

# We've have strict credit score criteria, which have tightned over the years
ggplot(data= subset(core, CreditScoreRangeLower >0),
       aes(x=LoanOriginationDate, y=CreditScoreRangeLower)) +
  geom_jitter(alpha=0.1)

# Prosper score sorta fits with credit score
ggplot(data= subset(core, ProsperScore != ""),
       aes(x=LoanOriginationDate, y=CreditScoreRangeLower, group=ProsperScore,
           col=ProsperScore)) +
  geom_jitter(alpha=0.3)

# Total loans - company is really ramping up loans over 2013
by_day_sum <- core[,list(counted=.N),by=LoanOriginationDate]
by_day_sum <- setorder(by_day_sum, LoanOriginationDate)
ggplot(by_day_sum,
       aes(x=LoanOriginationDate, y=counted)) +
  geom_point() +
  geom_smooth()

# Graph defaults over time - recent customers haven't had time to default
by_day_defaults <- core[Default=='Yes',list(counted=.N),by=LoanOriginationDate]
by_day_defaults <- setorder(by_day_defaults, LoanOriginationDate)
ggplot(by_day_defaults,
       aes(x=LoanOriginationDate, y=counted)) +
  geom_point() +
  geom_smooth()



###################### 4. REDUCE FRAME AND PREPARE FOR ML ###############

# Create a smaller frame 
learn_columns = c('Default','CreditScoreRangeLower',
                  'MonthlyLoanPayment','BorrowerRate','IsBorrowerHomeowner',
                  'LoanOriginationDate',
                  'StatedMonthlyIncome','IncomeRange')
learn <- core[,learn_columns,with=FALSE]

# Remove NAs
NA_learn_counts <- apply(learn, 2, function(x) sum(is.na(x)))
NA_learn_counts 
learn$Default <- as.factor(learn$Default)
learn <- learn[!is.na(learn$CreditScoreRangeLower)]
learn <- learn[CreditScoreRangeLower>0]

# Look at correlations
cor_matrix <- as.matrix(cor(learn[,-c('IsBorrowerHomeowner','Default',                                    'LoanOriginationDate','IncomeRange'),with=FALSE]))
cor_melted <- arrange(melt(cor_matrix), -abs(value))
cor_melted

# Remove selected variables
learn$LoanOriginationDate <- NULL
learn$IncomeRange <-NULL

# Aggressive and somewhat arbitrary elimination of outliers
ggplot(data=learn, aes(x=StatedMonthlyIncome)) + geom_histogram(binwidth=500)
table(learn$StatedMonthlyIncome > 25000)
table(learn$StatedMonthlyIncome < 1000)
learn <- learn[!learn$StatedMonthlyIncome > 25000]
learn <- learn[!learn$StatedMonthlyIncome < 1000]

ggplot(data=learn, aes(x=MonthlyLoanPayment)) + geom_histogram(binwidth=10)
table(learn$MonthlyLoanPayment > 1000)
table(learn$MonthlyLoanPayment < 10)
learn <- learn[!learn$MonthlyLoanPayment > 1000]
learn <- learn[!learn$MonthlyLoanPayment <10]

# And the big one - throw out current loans (ie. more than half the rows)
learn <- learn[!Default=='Current']
learn$Default <- droplevels(learn$Default)

# Pre-process:center and scale
PreProc <- preProcess(learn, method = c('center','scale'))
learn <- predict(PreProc, learn)
learn$IsBorrowerHomeowner <- as.numeric(learn$IsBorrowerHomeowner)

# Preprocessing checks
bad_vars <-nearZeroVar(learn, saveMetrics=TRUE)
bad_vars




###################### 5. CLASSIFICATION ML #########################

# split data
set.seed(42)
train_index <- createDataPartition(learn$Default, p=0.8, list=F)
main <- learn[train_index]
test <- learn[-train_index]

# Final prep
trControl <- trainControl(method="cv", number=5, repeats=1, 
                          summaryFunction=twoClassSummary, classProbs=TRUE)
metric <- "ROC"

# Make fits
set.seed(42)
lda_fit <- train(Default ~ ., data=main,
                 method='lda', metric=metric, trControl=trControl)
set.seed(42)
knn_fit <- train(Default ~ ., data=main, 
                 method='knn', metric=metric, trControl=trControl)
set.seed(42)
nb_fit <- train(Default ~ ., data=main, 
                method='nb', metric=metric, trControl=trControl)
set.seed(42)
glm_fit <- train(Default ~ ., data=main,
                 method='glmnet', metric=metric, trControl=trControl)
set.seed(42)
rf_fit <- train(Default ~ ., data=main,
                method='rf', metric=metric, trControl=trControl, ntree=100,
                nodesize=30, cutoff=c(0.55,0.45)) 
set.seed(42)
svm_fit <- train(Default ~ ., data=main,
                 method='svmRadial', metric=metric, trControl=trControl, 
                 class.weights=c('Yes'=2, 'No'=1))

# Summarize models
collection <- resamples(list(LDA=lda_fit,KNN=knn_fit, NB=nb_fit, GLMNET=glm_fit, RF=rf_fit))
summary(collection)
dotplot(collection)

# RF test set
rf_test_preds <- predict(rf_fit, test)
confusionMatrix(rf_test_preds, test$Default, positive='Yes')

# SVM  on test set
svm_test_preds <- predict(svm_fit3, test)
confusionMatrix(svm_test_preds, test$Default, positive='Yes')

# Try oversampling the small class - downsample with random forest
nmin <- sum(main$Default == 'Yes')
set.seed(42)
rf_fit_equal <- train(Default ~ ., data=main, strata=main$Default, sampsize = rep(nmin, 2),
                      method='rf', metric=metric, trControl=trControl,
                      ntree=100, nodesize=5)

# RF on test set
rf_test_preds_equal <- predict(rf_fit_equal, test)
confusionMatrix(rf_test_preds_equal, test$Default, positive='Yes')
