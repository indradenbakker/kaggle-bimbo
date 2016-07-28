setwd("~/Bimbo")

library(readr)
library(xgboost)


train <- read_csv("train.csv")

dev <- train[train$Semana <= 8,]
val <- train[train$Semana == 9,]

dev$Semana <- as.numeric(dev$Semana)
val$Semana <- as.numeric(val$Semana)

dev$Demanda_uni_equil <- log1p(dev$Demanda_uni_equil)
val$Demanda_uni_equil <- log1p(val$Demanda_uni_equil)

xgtrain = xgb.DMatrix(as.matrix(dev[,-c(which(colnames(dev) %in% c("Venta_uni_hoy", "Venta_hoy", "Dev_uni_proxima", "Dev_proxima", "Demanda_uni_equil")))]), label = as.numeric(dev$Demanda_uni_equil))
xgval = xgb.DMatrix(as.matrix(val[,-c(which(colnames(val) %in% c("Venta_uni_hoy", "Venta_hoy", "Dev_uni_proxima", "Dev_proxima", "Demanda_uni_equil")))]), label = as.numeric(dev$Demanda_uni_equil))

param0 <- list(
  # some generic, non specific params
  "objective"  = "reg:linear"
  , "eval_metric" = "rmse"
  , "eta" = 0.2
  , "subsample" = 0.6
  , "colsample_bytree" = 0.8
  # , "min_child_weight" = 1
  , "max_depth" = 40
)

model_cv = xgb.cv(
  params = param0
  , nrounds = 500
  , nfold = 5
  , data = xgtrain
  , early.stop.round = 10
  , print.every.n = 5
  , maximize = FALSE
  , nthread = 28
)
gc()
best <- min(model_cv$test.logloss.mean)
bestIter <- which(model_cv$test.logloss.mean==best)-1

cat("\n",best, bestIter,"\n")
print(model_cv[bestIter])


watchlist <- list('train' = xgtrain)
set.seed(2016)
model = xgb.train(
  nrounds = 500
  , params = param0
  , data = xgtrain
  , watchlist = watchlist
  , print.every.n = 5
  , nthread = 18
)


[0]	train-rmse:1.079122
[5]	train-rmse:0.650779
[10]	train-rmse:0.625654
[15]	train-rmse:0.618966
[20]	train-rmse:0.615783
[25]	train-rmse:0.613114
[30]	train-rmse:0.611435
[35]	train-rmse:0.610070
[40]	train-rmse:0.608811
