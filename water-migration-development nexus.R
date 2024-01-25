### Water Stress, International Migration and Development Nexus Using Machine Learning Approach

## Loading the Necessary Libraries

library(readxl)
library(dplyr)
library(tidyr)
library(tidyverse)
library(ggplot2)
library(caret)
library(vip)
library(pdp)
library(iml)
library(patchwork)
library(corrplot)
library(ggpubr)
library(Hmisc)


## Loading the Dataset

dataset_wmd <- read_excel("Final Dataset.xlsx", sheet = "Sheet2", col_types = c("text", "text", "text", "text", "numeric", "numeric", "numeric", "numeric", "numeric", "numeric", "numeric",
                                                                              "text", "numeric", "numeric", "numeric", "numeric"))
view(dataset_wmd)

## Calculating Relative change in target variables (NI, NE, IMS_BS) 
## (Relative Change will be calculated based on the value of 1990-1994 for each country)

# RC_NI

baseline_data_NI <- dataset_wmd %>%
  dplyr::group_by(Countries) %>%
  dplyr::filter(Year == "1990-1994") %>%
  dplyr::select(Countries, baseline_value = NI)

dataset_wmd_RC <- dataset_wmd %>%
  dplyr::left_join(baseline_data_NI, by="Countries") %>%
  dplyr::mutate(RC_NI = (NI - baseline_value) / baseline_value) %>%
  dplyr::select(-baseline_value)

view(dataset_wmd_RC)

# RC_NE

baseline_data_NE <- dataset_wmd_RC %>%
  dplyr::group_by(Countries) %>%
  dplyr::filter(Year == "1990-1994") %>%
  dplyr::select(Countries, baseline_value = NE)

dataset_wmd_RC <- dataset_wmd_RC %>%
  dplyr::left_join(baseline_data_NE, by="Countries") %>%
  dplyr::mutate(RC_NE = (NE - baseline_value) / baseline_value) %>%
  dplyr::select(-baseline_value)

view(dataset_wmd_RC)

# RC_IMS

baseline_data_IMS <- dataset_wmd_RC %>%
  dplyr::group_by(Countries) %>%
  dplyr::filter(Year == "1990-1994") %>%
  dplyr::select(Countries, baseline_value = IMS_BS)

dataset_wmd_RC <- dataset_wmd_RC %>%
  dplyr::left_join(baseline_data_IMS, by="Countries") %>%
  dplyr::mutate(RC_IMS = (IMS_BS - baseline_value) / baseline_value) %>%
  dplyr::select(-baseline_value)

view(dataset_wmd_RC)

## Now excluding the rows of base values (Year == 1990-1994)

dataset_wmd_RC_1 <- dataset_wmd_RC[dataset_wmd_RC$Year != '1990-1994', ]
view(dataset_wmd_RC_1)

## Excluding the unnecessary columns of the dataset

dataset_wmd_RC_final <- dataset_wmd_RC_1[, -c(1:2, 4:7, 16)]
View(dataset_wmd_RC_final)

## Encoding the categorical variables by performing one hot encoding

dummy_dataset <- dummyVars(~ ., data=dataset_wmd_RC_final)
encoded_dataset <- data.frame(predict(dummy_dataset, newdata=dataset_wmd_RC_final))
view(encoded_dataset)

sum(is.na(encoded_dataset))

## Splitting the dataset for machine learning

set.seed(100)

train_test_dataset <- sample(c(1:nrow(encoded_dataset)), 0.8*nrow(encoded_dataset), replace=F)
data_train <- encoded_dataset[train_test_dataset, ]
data_test <- setdiff(encoded_dataset, data_train)

### Applying Machine Learning Algorithms

## Specifying the Method of Model Validation

modelcontrol <- trainControl(method = "cv", number = 10)

## Regression of NE (Number of Emigrants)

set.seed(123)
model_NE_lm <- caret::train(RC_NE ~ DSLeast.Developed + DSLess.Developed + DSMore.Developed + PD + GDPPC + HDI + WS + WSLCritical + WSLHigh.Stress + WSLLow.Stress + WSLMedium.Stress + WSLNo.Stress + WUE + NRI + TPASDW, 
                                   data=data_train, method="glmStepAIC", trControl=modelcontrol)

set.seed(123)
model_NE_cart <- caret::train(RC_NE ~ DSLeast.Developed + DSLess.Developed + DSMore.Developed + PD + GDPPC + HDI + WS + WSLCritical + WSLHigh.Stress + WSLLow.Stress + WSLMedium.Stress + WSLNo.Stress + WUE + NRI + TPASDW, 
                                     data=data_train, method="ctree", trControl=modelcontrol)

set.seed(123)
model_NE_rf <- caret::train(RC_NE ~ DSLeast.Developed + DSLess.Developed + DSMore.Developed + PD + GDPPC + HDI + WS + WSLCritical + WSLHigh.Stress + WSLLow.Stress + WSLMedium.Stress + WSLNo.Stress + WUE + NRI + TPASDW, 
                                   data=data_train, method="rf", trControl=modelcontrol)
set.seed(123)
model_NE_gbm <- caret::train(RC_NE ~ DSLeast.Developed + DSLess.Developed + DSMore.Developed + PD + GDPPC + HDI + WS + WSLCritical + WSLHigh.Stress + WSLLow.Stress + WSLMedium.Stress + WSLNo.Stress + WUE + NRI + TPASDW, 
                                    data=data_train, method="gbm", trControl=modelcontrol)

## Performance Evaluation of the models


pred_model_NE_lm <- predict(model_NE_lm, data_test)
pred_model_NE_cart <- predict(model_NE_cart, data_test)
pred_model_NE_rf <- predict(model_NE_rf, data_test)
pred_model_NE_gbm <- predict(model_NE_gbm, data_test)

# Calculating R-Squared

R2_model_NE_lm <- R2(pred_model_NE_lm, data_test$RC_NE)
R2_model_NE_cart <- R2(pred_model_NE_cart, data_test$RC_NE)
R2_model_NE_rf <- R2(pred_model_NE_rf, data_test$RC_NE)
R2_model_NE_gbm <- R2(pred_model_NE_gbm, data_test$RC_NE)

# Calculating RMSE

RMSE_model_NE_lm <- RMSE(pred_model_NE_lm, data_test$RC_NE)
RMSE_model_NE_cart <- RMSE(pred_model_NE_cart, data_test$RC_NE)
RMSE_model_NE_rf <- RMSE(pred_model_NE_rf, data_test$RC_NE)
RMSE_model_NE_gbm <- RMSE(pred_model_NE_gbm, data_test$RC_NE)

# Creating dataset for plotting R2 and RMSE

R2_NE <- data.frame(r2 = c(R2_model_NE_lm,R2_model_NE_cart,R2_model_NE_rf,R2_model_NE_gbm), 
                        algorithm = c("Linear model","Decision Tree","Random Forests","Gradient Boosting") %>% factor(.,levels=c("Linear model","Decision Tree","Random Forests","Gradient Boosting")))

R2_NE
Fig1.1a <-
  ggplot(R2_NE, aes(x=algorithm, y=r2, fill=algorithm)) + 
  geom_bar(stat="identity") + 
  ylab("R-squared") +
  scale_fill_manual(values=c("Linear model"="grey",
                             "Decision Tree"="orange",
                             "Random Forests"="darkgreen",
                             "Gradient Boosting"="darkblue")) +
  theme_bw() +
  theme(legend.position = "none")

Fig1.1a

RMSE_NE <- data.frame(rmse = c(RMSE_model_NE_lm,RMSE_model_NE_cart,RMSE_model_NE_rf,RMSE_model_NE_gbm), 
                    algorithm = c("Linear model","Decision Tree","Random Forests","Gradient Boosting") %>% factor(.,levels=c("Linear model","Decision Tree","Random Forests","Gradient Boosting")))

RMSE_NE

Fig1.1b <-
  ggplot(RMSE_NE, aes(x=algorithm, y=rmse, fill=algorithm)) + 
  geom_bar(stat="identity") + 
  ylab("RMSE") +
  scale_fill_manual(values=c("Linear model"="grey",
                             "Decision Tree"="orange",
                             "Random Forests"="darkgreen",
                             "Gradient Boosting"="darkblue")) +
  theme_bw() +
  theme(legend.position = "none")

Fig1.1b

Fig1.1 <- Fig1.1a + Fig1.1b + 
  plot_annotation(tag_levels = "a") + plot_layout(ncol = 1)

Fig1.1

# Variable Importance of the NE models
# Permutation-Based Feature Importance

set.seed(123)
vi_model_NE_lm <- vip(model_NE_lm, method="permute", train=data_train, target="RC_NE", metric="rsquared", 
                      pred_wrapper=predict, nsim=30, geom="boxplot",aesthetics = list(fill = "grey", color="black")) + labs(title="Linear Model") +theme_bw()
vi_model_NE_lm

set.seed(123)
vi_model_NE_cart <- vip(model_NE_cart, method="permute", train=data_train, target="RC_NE", metric="rsquared", 
                      pred_wrapper=predict, nsim=30, geom="boxplot",aesthetics = list(fill = "orange", color="black")) + labs(title="Decision Tree") +theme_bw()
vi_model_NE_cart

set.seed(123)
vi_model_NE_rf <- vip(model_NE_rf, method="permute", train=data_train, target="RC_NE", metric="rsquared", 
                        pred_wrapper=predict, nsim=30, geom="boxplot",aesthetics = list(fill = "darkgreen", color="black")) + labs(title="Random Forest") +theme_bw()
vi_model_NE_rf


set.seed(123)
vi_model_NE_gbm <- vip(model_NE_gbm, method="permute", train=data_train, target="RC_NE", metric="rsquared", 
                      pred_wrapper=predict, nsim=30, geom="boxplot",aesthetics = list(fill = "darkblue", color="black")) + labs(title="Gradient Boosting") +theme_bw()
vi_model_NE_gbm

Fig1.2 <-
  vi_model_NE_lm + vi_model_NE_cart + vi_model_NE_rf + vi_model_NE_gbm +
  plot_annotation(tag_levels = 'a')

Fig1.2

# Friedman's H-index
# Pairwise Interaction Statistics

int_model_NE_lm <-  vint(
  object = model_NE_lm,                    # fitted model object
  feature_names = c("DSLeast.Developed", "DSLess.Developed", "DSMore.Developed", "PD", "GDPPC", "HDI", "WS", "WSLCritical", "WSLHigh.Stress", "WSLLow.Stress", "WSLMedium.Stress", "WSLNo.Stress", "WUE", "NRI", "TPASDW"),  # features for which to compute pairwise interactions statistics
  parallel = TRUE
)
int_model_NE_lm

plot_int_model_NE_lm <- ggplot(int_model_NE_lm[1:10, ], aes(reorder(Variables, Interaction), Interaction)) +
  geom_bar(stat="identity", fill="grey") +
  labs(x = "", y = "Interaction strength") +
  theme(axis.text.x = element_text(angle = 45, vjust = 1)) +
  coord_flip()  +
  theme_bw() + labs(title="Linear Model")

int_model_NE_cart <- vint(
  object = model_NE_cart,                    # fitted model object
  feature_names = c("DSLeast.Developed", "DSLess.Developed", "DSMore.Developed", "PD", "GDPPC", "HDI", "WS", "WSLCritical", "WSLHigh.Stress", "WSLLow.Stress", "WSLMedium.Stress", "WSLNo.Stress", "WUE", "NRI", "TPASDW"),  # features for which to compute pairwise interactions statistics
  parallel = TRUE
)
int_model_NE_cart

plot_int_model_NE_cart <- ggplot(int_model_NE_cart[1:10, ], aes(reorder(Variables, Interaction), Interaction)) +
  geom_bar(stat="identity", fill="orange") +
  labs(x = "", y = "Interaction strength") +
  theme(axis.text.x = element_text(angle = 45, vjust = 1)) +
  coord_flip()  +
  theme_bw() + labs(title="Decision Tree")

int_model_NE_rf <- vint(
  object = model_NE_rf,                    # fitted model object
  feature_names = c("DSLeast.Developed", "DSLess.Developed", "DSMore.Developed", "PD", "GDPPC", "HDI", "WS", "WSLCritical", "WSLHigh.Stress", "WSLLow.Stress", "WSLMedium.Stress", "WSLNo.Stress", "WUE", "NRI", "TPASDW"),  # features for which to compute pairwise interactions statistics
  parallel = TRUE
)
int_model_NE_rf

plot_int_model_NE_rf <- ggplot(int_model_NE_rf[1:10, ], aes(reorder(Variables, Interaction), Interaction)) +
  geom_bar(stat="identity", fill="darkgreen") +
  labs(x = "", y = "Interaction strength") +
  theme(axis.text.x = element_text(angle = 45, vjust = 1)) +
  coord_flip()  +
  theme_bw() + labs(title="Random Forest")

plot_int_model_NE_rf

int_model_NE_gbm <- vint(
  object = model_NE_gbm,                    # fitted model object
  feature_names = c("DSLeast.Developed", "DSLess.Developed", "DSMore.Developed", "PD", "GDPPC", "HDI", "WS", "WSLCritical", "WSLHigh.Stress", "WSLLow.Stress", "WSLMedium.Stress", "WSLNo.Stress", "WUE", "NRI", "TPASDW"),  # features for which to compute pairwise interactions statistics
  parallel = TRUE
)
int_model_NE_gbm

plot_int_model_NE_gbm <- ggplot(int_model_NE_gbm[1:10, ], aes(reorder(Variables, Interaction), Interaction)) +
  geom_bar(stat="identity", fill="darkblue") +
  labs(x = "", y = "Interaction strength") +
  theme(axis.text.x = element_text(angle = 45, vjust = 1)) +
  coord_flip()  +
  theme_bw() + labs(title="Gradient Boosting")

plot_int_model_NE_gbm

Fig1.3 <- plot_int_model_NE_lm + plot_int_model_NE_cart + plot_int_model_NE_rf + plot_int_model_NE_gbm +
  plot_annotation(tag_levels = 'a')

Fig1.3


## Partial Dependence Plot

# NRI: The most important variables

pdp_NRI   <- rbind(
  model_NE_lm %>%  partial(pred.var=c("NRI")) %>% cbind(., algorithm = "Linear model"),
  model_NE_cart %>%  partial(pred.var=c("NRI"), approx=T) %>% cbind(., algorithm = "Decision Tree"),
  model_NE_rf %>%  partial(pred.var=c("NRI"), approx=T) %>% cbind(., algorithm = "Random Forest"),
  model_NE_gbm %>%  partial(pred.var=c("NRI"), approx=T)  %>% cbind(., algorithm = "Gradient Boosting")
) 
pdp_NRI$algorithm <- factor(pdp_NRI$algorithm, levels=c("Linear model","Decision Tree","Random Forest","Gradient Boosting"))

# WS: The second most important variables

pdp_WS   <- rbind(
  model_NE_lm %>%  partial(pred.var=c("WS")) %>% cbind(., algorithm = "Linear model"),
  model_NE_cart %>%  partial(pred.var=c("WS"), approx=T) %>% cbind(., algorithm = "Decision Tree"),
  model_NE_rf %>%  partial(pred.var=c("WS"), approx=T) %>% cbind(., algorithm = "Random Forest"),
  model_NE_gbm %>%  partial(pred.var=c("WS"), approx=T)  %>% cbind(., algorithm = "Gradient Boosting")
) 
pdp_WS$algorithm <- factor(pdp_WS$algorithm, levels=c("Linear model","Decision Tree","Random Forest","Gradient Boosting"))


Fig1.4a <-
  ggplot(pdp_NRI, aes(x=NRI, y=yhat, color=algorithm)) +
  geom_line(size=1) +
  scale_color_manual(values=c("Linear model"="darkgrey",
                              "Decision Tree"="orange",
                              "Random Forest"="darkgreen",
                              "Gradient Boosting"="darkblue"))  +
  ylab("Partial dependence") +
  theme_bw() +
  theme(
        axis.title.x = element_blank(),
        axis.text.x  = element_blank())
Fig1.4a

Fig1.4b <-
  ggplot(pdp_WS, aes(x=WS, y=yhat, color=algorithm)) +
  geom_line(size=1) +
  scale_color_manual(values=c("Linear model"="darkgrey",
                              "Decision Tree"="orange",
                              "Random Forest"="darkgreen",
                              "Gradient Boosting"="darkblue"))  +
  ylab("Partial dependence") +
  theme_bw() +
  theme(legend.position="none", 
        axis.title.x = element_blank(),
        axis.text.x  = element_blank())
Fig1.4b

Fig1.4c <- ggplot(data_train, aes(x=NRI)) + geom_histogram() + theme_bw()
Fig1.4d <- ggplot(data_train, aes(x=WS)) + geom_histogram() + theme_bw()

Fig1.4 <- Fig1.4a + Fig1.4b + 
  Fig1.4c + Fig1.4d +
  plot_annotation(tag_levels = "a") +
  plot_layout(heights=c(9,1))

Fig1.4

# Two-way Interactions

pdp_NE_lm <- model_NE_lm %>%  partial(pred.var=c("NRI","WS")) %>% autoplot + labs(title="Linear model")
pdp_NE_cart <- model_NE_cart %>%  partial(pred.var=c("NRI","WS"), approx=T) %>% autoplot + labs(title="Decision Tree")
pdp_NE_rf <- model_NE_rf %>%  partial(pred.var=c("NRI","WS"), approx=T) %>% autoplot + labs(title="Random Forest")
pdp_NE_gbm <- model_NE_gbm %>%  partial(pred.var=c("NRI","WS"), approx=T) %>% autoplot + labs(title="Gradient Boosting")

Fig1.5 <-
  pdp_NE_lm + pdp_NE_cart + pdp_NE_rf + pdp_NE_gbm

Fig1.5

# NRI*WS

pdp_NRI_WS   <- rbind(
  model_NE_lm %>%  partial(pred.var=c("NRI","WS")) %>% cbind(., algorithm = "Linear model"),
  model_NE_cart %>%  partial(pred.var=c("NRI","WS"), approx=T) %>% cbind(., algorithm = "Decision Tree"),
  model_NE_rf %>%  partial(pred.var=c("NRI","WS"), approx=T) %>% cbind(., algorithm = "Random Forest"),
  model_NE_gbm %>%  partial(pred.var=c("NRI","WS"), approx=T, n.trees=500)  %>% cbind(., algorithm = "Gradient Boosting")
) 

pdp_NRI_WS$algorithm <- factor(pdp_NRI_WS$algorithm, levels=c("Linear model","Decision Tree","Random Forest","Gradient Boosting"))
pdp_NRI_WS <- 
  pdp_NRI_WS  %>% 
  mutate(WS_class = case_when(
    WS > 50 ~ "WS > 75%",
    WS < 50 ~ "WS < 75%"
  ))

Fig1.6 <-
  ggplot(pdp_NRI_WS, aes(x=NRI, y=yhat, 
                            color=algorithm,
                            linetype=WS_class)) +
  facet_wrap(vars(algorithm)) +
  geom_line(stat="summary", fun=mean, size=1) +
  ylab("Partial dependence") +
  theme_bw()  +
  scale_color_manual(values=c("Linear model"="darkgrey",
                              "Decision Tree"="orange",
                              "Random Forest"="darkgreen",
                              "Gradient Boosting"="darkblue"))  

Fig1.6

# WSLCritical * HDI

# WSLCritical

pdp_WSLCritical   <- rbind(
  model_NE_lm %>%  partial(pred.var=c("WSLCritical")) %>% cbind(., algorithm = "Linear model"),
  model_NE_cart %>%  partial(pred.var=c("WSLCritical"), approx=T) %>% cbind(., algorithm = "Decision Tree"),
  model_NE_rf %>%  partial(pred.var=c("WSLCritical"), approx=T) %>% cbind(., algorithm = "Random Forest"),
  model_NE_gbm %>%  partial(pred.var=c("WSLCritical"), approx=T)  %>% cbind(., algorithm = "Gradient Boosting")
) 
pdp_WSLCritical$algorithm <- factor(pdp_WSLCritical$algorithm, levels=c("Linear model","Decision Tree","Random Forest","Gradient Boosting"))


# HDI

pdp_HDI   <- rbind(
  model_NE_lm %>%  partial(pred.var=c("HDI")) %>% cbind(., algorithm = "Linear model"),
  model_NE_cart %>%  partial(pred.var=c("HDI"), approx=T) %>% cbind(., algorithm = "Decision Tree"),
  model_NE_rf %>%  partial(pred.var=c("HDI"), approx=T) %>% cbind(., algorithm = "Random Forest"),
  model_NE_gbm %>%  partial(pred.var=c("HDI"), approx=T)  %>% cbind(., algorithm = "Gradient Boosting")
) 
pdp_HDI$algorithm <- factor(pdp_HDI$algorithm, levels=c("Linear model","Decision Tree","Random Forest","Gradient Boosting"))

Fig1.7a <-
  ggplot(pdp_WSLCritical, aes(x=WSLCritical, y=yhat, color=algorithm)) +
  geom_line(size=1) +
  scale_color_manual(values=c("Linear model"="darkgrey",
                              "Decision Tree"="orange",
                              "Random Forest"="darkgreen",
                              "Gradient Boosting"="darkblue"))  +
  ylab("Partial dependence") +
  theme_bw() +
  theme(
    axis.title.x = element_blank(),
    axis.text.x  = element_blank())
Fig1.7a

Fig1.7b <-
  ggplot(pdp_HDI, aes(x=HDI, y=yhat, color=algorithm)) +
  geom_line(size=1) +
  scale_color_manual(values=c("Linear model"="darkgrey",
                              "Decision Tree"="orange",
                              "Random Forest"="darkgreen",
                              "Gradient Boosting"="darkblue"))  +
  ylab("Partial dependence") +
  theme_bw() +
  theme(legend.position="none", 
        axis.title.x = element_blank(),
        axis.text.x  = element_blank())
Fig1.7b

Fig1.7c <- ggplot(data_train, aes(x=WSLCritical)) + geom_histogram() + theme_bw()
Fig1.7d <- ggplot(data_train, aes(x=HDI)) + geom_histogram() + theme_bw()

Fig1.7 <- Fig1.7a + Fig1.7b + 
  Fig1.7c + Fig1.7d +
  plot_annotation(tag_levels = "a") +
  plot_layout(heights=c(9,1))

Fig1.7

# Two-way Interactions

pdp_NE_lm_1 <- model_NE_lm %>%  partial(pred.var=c("WSLCritical","HDI")) %>% autoplot + labs(title="Linear model")
pdp_NE_cart_1 <- model_NE_cart %>%  partial(pred.var=c("WSLCritical","HDI"), approx=T) %>% autoplot + labs(title="Decision Tree")
pdp_NE_rf_1 <- model_NE_rf %>%  partial(pred.var=c("WSLCritical","HDI"), approx=T) %>% autoplot + labs(title="Random Forest")
pdp_NE_gbm_1 <- model_NE_gbm %>%  partial(pred.var=c("WSLCritical","HDI"), approx=T) %>% autoplot + labs(title="Gradient Boosting")

Fig1.8 <-
  pdp_NE_lm_1 + pdp_NE_cart_1 + pdp_NE_rf_1 + pdp_NE_gbm_1

Fig1.8

# WSLCritical * HDI

pdp_WSLCritical_HDI   <- rbind(
  model_NE_lm %>%  partial(pred.var=c("WSLCritical","HDI")) %>% cbind(., algorithm = "Linear model"),
  model_NE_cart %>%  partial(pred.var=c("WSLCritical","HDI"), approx=T) %>% cbind(., algorithm = "Decision Tree"),
  model_NE_rf %>%  partial(pred.var=c("WSLCritical","HDI"), approx=T) %>% cbind(., algorithm = "Random Forest"),
  model_NE_gbm %>%  partial(pred.var=c("WSLCritical","HDI"), approx=T, n.trees=500)  %>% cbind(., algorithm = "Gradient Boosting")
) 

pdp_WSLCritical_HDI$algorithm <- factor(pdp_WSLCritical_HDI$algorithm, levels=c("Linear model","Decision Tree","Random Forest","Gradient Boosting"))
pdp_WSLCritical_HDI <- 
  pdp_WSLCritical_HDI  %>% 
  mutate(HDI_class = case_when(
    HDI > 0.55 ~ "HDI > 0.55",
    HDI < 0.55 ~ "HDI < 0.55"
  ))

Fig1.9 <-
  ggplot(pdp_WSLCritical_HDI, aes(x=WSLCritical, y=yhat, 
                         color=algorithm,
                         linetype=HDI_class)) +
  facet_wrap(vars(algorithm)) +
  geom_line(stat="summary", fun=mean, size=1) +
  ylab("Partial dependence") +
  theme_bw()  +
  scale_color_manual(values=c("Linear model"="darkgrey",
                              "Decision Tree"="orange",
                              "Random Forest"="darkgreen",
                              "Gradient Boosting"="darkblue"))  

Fig1.9

## Regression of NI (Number of Immigrants)

set.seed(123)
model_NI_lm <- caret::train(RC_NI ~ DSLeast.Developed + DSLess.Developed + DSMore.Developed + PD + GDPPC + HDI + WS + WSLCritical + WSLHigh.Stress + WSLLow.Stress + WSLMedium.Stress + WSLNo.Stress + WUE + NRI + TPASDW, 
                            data=data_train, method="glmStepAIC", trControl=modelcontrol)

set.seed(123)
model_NI_cart <- caret::train(RC_NI ~ DSLeast.Developed + DSLess.Developed + DSMore.Developed + PD + GDPPC + HDI + WS + WSLCritical + WSLHigh.Stress + WSLLow.Stress + WSLMedium.Stress + WSLNo.Stress + WUE + NRI + TPASDW, 
                              data=data_train, method="ctree", trControl=modelcontrol)

set.seed(123)
model_NI_rf <- caret::train(RC_NI ~ DSLeast.Developed + DSLess.Developed + DSMore.Developed + PD + GDPPC + HDI + WS + WSLCritical + WSLHigh.Stress + WSLLow.Stress + WSLMedium.Stress + WSLNo.Stress + WUE + NRI + TPASDW, 
                            data=data_train, method="rf", trControl=modelcontrol)
set.seed(123)
model_NI_gbm <- caret::train(RC_NI ~ DSLeast.Developed + DSLess.Developed + DSMore.Developed + PD + GDPPC + HDI + WS + WSLCritical + WSLHigh.Stress + WSLLow.Stress + WSLMedium.Stress + WSLNo.Stress + WUE + NRI + TPASDW, 
                             data=data_train, method="gbm", trControl=modelcontrol)


## Performance Evaluation of the models


pred_model_NI_lm <- predict(model_NI_lm, data_test)
pred_model_NI_cart <- predict(model_NI_cart, data_test)
pred_model_NI_rf <- predict(model_NI_rf, data_test)
pred_model_NI_gbm <- predict(model_NI_gbm, data_test)

# Calculating R-Squared

R2_model_NI_lm <- R2(pred_model_NI_lm, data_test$RC_NI)
R2_model_NI_cart <- R2(pred_model_NI_cart, data_test$RC_NI)
R2_model_NI_rf <- R2(pred_model_NI_rf, data_test$RC_NI)
R2_model_NI_gbm <- R2(pred_model_NI_gbm, data_test$RC_NI)

# Calculating RMSE

RMSE_model_NI_lm <- RMSE(pred_model_NI_lm, data_test$RC_NI)
RMSE_model_NI_cart <- RMSE(pred_model_NI_cart, data_test$RC_NI)
RMSE_model_NI_rf <- RMSE(pred_model_NI_rf, data_test$RC_NI)
RMSE_model_NI_gbm <- RMSE(pred_model_NI_gbm, data_test$RC_NI)

# Creating dataset for plotting R2 and RMSE

R2_NI <- data.frame(r2 = c(R2_model_NI_lm,R2_model_NI_cart,R2_model_NI_rf,R2_model_NI_gbm), 
                    algorithm = c("Linear model","Decision Tree","Random Forests","Gradient Boosting") %>% factor(.,levels=c("Linear model","Decision Tree","Random Forests","Gradient Boosting")))

R2_NI
Fig2.1a <-
  ggplot(R2_NI, aes(x=algorithm, y=r2, fill=algorithm)) + 
  geom_bar(stat="identity") + 
  ylab("R-squared") +
  scale_fill_manual(values=c("Linear model"="grey",
                             "Decision Tree"="orange",
                             "Random Forests"="darkgreen",
                             "Gradient Boosting"="darkblue")) +
  theme_bw() +
  theme(legend.position = "none")

Fig2.1a

RMSE_NI <- data.frame(rmse = c(RMSE_model_NI_lm,RMSE_model_NI_cart,RMSE_model_NI_rf,RMSE_model_NI_gbm), 
                      algorithm = c("Linear model","Decision Tree","Random Forests","Gradient Boosting") %>% factor(.,levels=c("Linear model","Decision Tree","Random Forests","Gradient Boosting")))

RMSE_NI

Fig2.1b <-
  ggplot(RMSE_NI, aes(x=algorithm, y=rmse, fill=algorithm)) + 
  geom_bar(stat="identity") + 
  ylab("RMSE") +
  scale_fill_manual(values=c("Linear model"="grey",
                             "Decision Tree"="orange",
                             "Random Forests"="darkgreen",
                             "Gradient Boosting"="darkblue")) +
  theme_bw() +
  theme(legend.position = "none")

Fig2.1b

Fig2.1 <- Fig2.1a + Fig2.1b + 
  plot_annotation(tag_levels = "a") + plot_layout(ncol = 1)

Fig2.1

# Variable Importance of the NI models
# Permutation-Based Feature Importance

set.seed(123)
vi_model_NI_lm <- vip(model_NI_lm, method="permute", train=data_train, target="RC_NI", metric="rsquared", 
                      pred_wrapper=predict, nsim=30, geom="boxplot",aesthetics = list(fill = "grey", color="black")) + labs(title="Linear Model") +theme_bw()
vi_model_NI_lm

set.seed(123)
vi_model_NI_cart <- vip(model_NI_cart, method="permute", train=data_train, target="RC_NI", metric="rsquared", 
                        pred_wrapper=predict, nsim=30, geom="boxplot",aesthetics = list(fill = "orange", color="black")) + labs(title="Decision Tree") +theme_bw()
vi_model_NI_cart

set.seed(123)
vi_model_NI_rf <- vip(model_NI_rf, method="permute", train=data_train, target="RC_NI", metric="rsquared", 
                      pred_wrapper=predict, nsim=30, geom="boxplot",aesthetics = list(fill = "darkgreen", color="black")) + labs(title="Random Forest") +theme_bw()
vi_model_NI_rf


set.seed(123)
vi_model_NI_gbm <- vip(model_NI_gbm, method="permute", train=data_train, target="RC_NI", metric="rsquared", 
                       pred_wrapper=predict, nsim=30, geom="boxplot",aesthetics = list(fill = "darkblue", color="black")) + labs(title="Gradient Boosting") +theme_bw()
vi_model_NI_gbm

Fig2.2 <-
  vi_model_NI_lm + vi_model_NI_cart + vi_model_NI_rf + vi_model_NI_gbm +
  plot_annotation(tag_levels = 'a')

Fig2.2

# Friedman's H-index
# Pairwise Interaction Statistics

int_model_NI_lm <-  vint(
  object = model_NI_lm,                    # fitted model object
  feature_names = c("DSLeast.Developed", "DSLess.Developed", "DSMore.Developed", "PD", "GDPPC", "HDI", "WS", "WSLCritical", "WSLHigh.Stress", "WSLLow.Stress", "WSLMedium.Stress", "WSLNo.Stress", "WUE", "NRI", "TPASDW"),  # features for which to compute pairwise interactions statistics
  parallel = TRUE
)
int_model_NI_lm

plot_int_model_NI_lm <- ggplot(int_model_NI_lm[1:10, ], aes(reorder(Variables, Interaction), Interaction)) +
  geom_bar(stat="identity", fill="grey") +
  labs(x = "", y = "Interaction strength") +
  theme(axis.text.x = element_text(angle = 45, vjust = 1)) +
  coord_flip()  +
  theme_bw() + labs(title="Linear Model")

plot_int_model_NI_lm

int_model_NI_cart <- vint(
  object = model_NI_cart,                    # fitted model object
  feature_names = c("DSLeast.Developed", "DSLess.Developed", "DSMore.Developed", "PD", "GDPPC", "HDI", "WS", "WSLCritical", "WSLHigh.Stress", "WSLLow.Stress", "WSLMedium.Stress", "WSLNo.Stress", "WUE", "NRI", "TPASDW"),  # features for which to compute pairwise interactions statistics
  parallel = TRUE
)
int_model_NI_cart

plot_int_model_NI_cart <- ggplot(int_model_NI_cart[1:10, ], aes(reorder(Variables, Interaction), Interaction)) +
  geom_bar(stat="identity", fill="orange") +
  labs(x = "", y = "Interaction strength") +
  theme(axis.text.x = element_text(angle = 45, vjust = 1)) +
  coord_flip()  +
  theme_bw() + labs(title="Decision Tree")

int_model_NI_rf <- vint(
  object = model_NI_rf,                    # fitted model object
  feature_names = c("DSLeast.Developed", "DSLess.Developed", "DSMore.Developed", "PD", "GDPPC", "HDI", "WS", "WSLCritical", "WSLHigh.Stress", "WSLLow.Stress", "WSLMedium.Stress", "WSLNo.Stress", "WUE", "NRI", "TPASDW"),  # features for which to compute pairwise interactions statistics
  parallel = TRUE
)
int_model_NI_rf

plot_int_model_NI_rf <- ggplot(int_model_NI_rf[1:10, ], aes(reorder(Variables, Interaction), Interaction)) +
  geom_bar(stat="identity", fill="darkgreen") +
  labs(x = "", y = "Interaction strength") +
  theme(axis.text.x = element_text(angle = 45, vjust = 1)) +
  coord_flip()  +
  theme_bw() + labs(title="Random Forest")

plot_int_model_NI_rf

int_model_NI_gbm <- vint(
  object = model_NI_gbm,                    # fitted model object
  feature_names = c("DSLeast.Developed", "DSLess.Developed", "DSMore.Developed", "PD", "GDPPC", "HDI", "WS", "WSLCritical", "WSLHigh.Stress", "WSLLow.Stress", "WSLMedium.Stress", "WSLNo.Stress", "WUE", "NRI", "TPASDW"),  # features for which to compute pairwise interactions statistics
  parallel = TRUE
)
int_model_NI_gbm

plot_int_model_NI_gbm <- ggplot(int_model_NI_gbm[1:10, ], aes(reorder(Variables, Interaction), Interaction)) +
  geom_bar(stat="identity", fill="darkblue") +
  labs(x = "", y = "Interaction strength") +
  theme(axis.text.x = element_text(angle = 45, vjust = 1)) +
  coord_flip()  +
  theme_bw() + labs(title="Gradient Boosting")

plot_int_model_NI_gbm

Fig2.3 <- plot_int_model_NI_lm + plot_int_model_NI_cart + plot_int_model_NI_rf + plot_int_model_NI_gbm +
  plot_annotation(tag_levels = 'a')

Fig2.3

## Partial Dependence Plot

# GDPPC: The most important variables

pdp_GDPPC   <- rbind(
  model_NI_lm %>%  partial(pred.var=c("GDPPC")) %>% cbind(., algorithm = "Linear model"),
  model_NI_cart %>%  partial(pred.var=c("GDPPC"), approx=T) %>% cbind(., algorithm = "Decision Tree"),
  model_NI_rf %>%  partial(pred.var=c("GDPPC"), approx=T) %>% cbind(., algorithm = "Random Forest"),
  model_NI_gbm %>%  partial(pred.var=c("GDPPC"), approx=T)  %>% cbind(., algorithm = "Gradient Boosting")
) 
pdp_GDPPC$algorithm <- factor(pdp_GDPPC$algorithm, levels=c("Linear model","Decision Tree","Random Forest","Gradient Boosting"))

# NRI: The second most important variables

pdp_NRI_NI_1   <- rbind(
  model_NI_lm %>%  partial(pred.var=c("NRI")) %>% cbind(., algorithm = "Linear model"),
  model_NI_cart %>%  partial(pred.var=c("NRI"), approx=T) %>% cbind(., algorithm = "Decision Tree"),
  model_NI_rf %>%  partial(pred.var=c("NRI"), approx=T) %>% cbind(., algorithm = "Random Forest"),
  model_NI_gbm %>%  partial(pred.var=c("NRI"), approx=T)  %>% cbind(., algorithm = "Gradient Boosting")
) 
pdp_NRI_NI_1$algorithm <- factor(pdp_NRI_NI_1$algorithm, levels=c("Linear model","Decision Tree","Random Forest","Gradient Boosting"))


Fig2.4a <-
  ggplot(pdp_GDPPC, aes(x=GDPPC, y=yhat, color=algorithm)) +
  geom_line(size=1) +
  scale_color_manual(values=c("Linear model"="darkgrey",
                              "Decision Tree"="orange",
                              "Random Forest"="darkgreen",
                              "Gradient Boosting"="darkblue"))  +
  ylab("Partial dependence") +
  theme_bw() +
  theme(
    axis.title.x = element_blank(),
    axis.text.x  = element_blank())
Fig2.4a

Fig2.4b <-
  ggplot(pdp_NRI_NI_1, aes(x=NRI, y=yhat, color=algorithm)) +
  geom_line(size=1) +
  scale_color_manual(values=c("Linear model"="darkgrey",
                              "Decision Tree"="orange",
                              "Random Forest"="darkgreen",
                              "Gradient Boosting"="darkblue"))  +
  ylab("Partial dependence") +
  theme_bw() +
  theme(legend.position="none", 
        axis.title.x = element_blank(),
        axis.text.x  = element_blank())
Fig2.4b

Fig2.4c <- ggplot(data_train, aes(x=GDPPC)) + geom_histogram() + theme_bw()
Fig2.4d <- ggplot(data_train, aes(x=NRI)) + geom_histogram() + theme_bw()

Fig2.4 <- Fig2.4a + Fig2.4b + 
  Fig2.4c + Fig2.4d +
  plot_annotation(tag_levels = "a") +
  plot_layout(heights=c(9,1))

Fig2.4

# Two-way Interactions

pdp_NI_lm <- model_NI_lm %>%  partial(pred.var=c("NRI","GDPPC")) %>% autoplot + labs(title="Linear model")
pdp_NI_cart <- model_NI_cart %>%  partial(pred.var=c("NRI","GDPPC"), approx=T) %>% autoplot + labs(title="Decision Tree")
pdp_NI_rf <- model_NI_rf %>%  partial(pred.var=c("NRI","GDPPC"), approx=T) %>% autoplot + labs(title="Random Forest")
pdp_NI_gbm <- model_NI_gbm %>%  partial(pred.var=c("NRI","GDPPC"), approx=T) %>% autoplot + labs(title="Gradient Boosting")

Fig2.5 <-
  pdp_NI_lm + pdp_NI_cart + pdp_NI_rf + pdp_NI_gbm

Fig2.5

# GDPPC*NRI

pdp_GDPPC_NRI   <- rbind(
  model_NI_lm %>%  partial(pred.var=c("NRI","GDPPC")) %>% cbind(., algorithm = "Linear model"),
  model_NI_cart %>%  partial(pred.var=c("NRI","GDPPC"), approx=T) %>% cbind(., algorithm = "Decision Tree"),
  model_NI_rf %>%  partial(pred.var=c("NRI","GDPPC"), approx=T) %>% cbind(., algorithm = "Random Forest"),
  model_NI_gbm %>%  partial(pred.var=c("NRI","GDPPC"), approx=T, n.trees=500)  %>% cbind(., algorithm = "Gradient Boosting")
) 

pdp_GDPPC_NRI$algorithm <- factor(pdp_GDPPC_NRI$algorithm, levels=c("Linear model","Decision Tree","Random Forest","Gradient Boosting"))
pdp_GDPPC_NRI <- 
  pdp_GDPPC_NRI  %>% 
  mutate(GDPPC_class = case_when(
    GDPPC > 30000 ~ "GDPPC > 30000",
    GDPPC < 30000 ~ "GDPPC < 30000"
  ))

Fig2.6 <-
  ggplot(pdp_GDPPC_NRI, aes(x=NRI, y=yhat, 
                         color=algorithm,
                         linetype=GDPPC_class)) +
  facet_wrap(vars(algorithm)) +
  geom_line(stat="summary", fun=mean, size=1) +
  ylab("Partial dependence") +
  theme_bw()  +
  scale_color_manual(values=c("Linear model"="darkgrey",
                              "Decision Tree"="orange",
                              "Random Forest"="darkgreen",
                              "Gradient Boosting"="darkblue"))  

Fig2.6

# Partial Depedence Plot of NRI*WS

# NRI: The most important variables

pdp_NRI_NI_2   <- rbind(
  model_NI_lm %>%  partial(pred.var=c("NRI")) %>% cbind(., algorithm = "Linear model"),
  model_NI_cart %>%  partial(pred.var=c("NRI"), approx=T) %>% cbind(., algorithm = "Decision Tree"),
  model_NI_rf %>%  partial(pred.var=c("NRI"), approx=T) %>% cbind(., algorithm = "Random Forest"),
  model_NI_gbm %>%  partial(pred.var=c("NRI"), approx=T)  %>% cbind(., algorithm = "Gradient Boosting")
) 
pdp_NRI_NI_2$algorithm <- factor(pdp_NRI_NI_2$algorithm, levels=c("Linear model","Decision Tree","Random Forest","Gradient Boosting"))

# WS: The second most important variables

pdp_WS_NI   <- rbind(
  model_NI_lm %>%  partial(pred.var=c("WS")) %>% cbind(., algorithm = "Linear model"),
  model_NI_cart %>%  partial(pred.var=c("WS"), approx=T) %>% cbind(., algorithm = "Decision Tree"),
  model_NI_rf %>%  partial(pred.var=c("WS"), approx=T) %>% cbind(., algorithm = "Random Forest"),
  model_NI_gbm %>%  partial(pred.var=c("WS"), approx=T)  %>% cbind(., algorithm = "Gradient Boosting")
) 
pdp_WS_NI$algorithm <- factor(pdp_WS_NI$algorithm, levels=c("Linear model","Decision Tree","Random Forest","Gradient Boosting"))


Fig2.5a <-
  ggplot(pdp_NRI_NI_2, aes(x=NRI, y=yhat, color=algorithm)) +
  geom_line(size=1) +
  scale_color_manual(values=c("Linear model"="darkgrey",
                              "Decision Tree"="orange",
                              "Random Forest"="darkgreen",
                              "Gradient Boosting"="darkblue"))  +
  ylab("Partial dependence") +
  theme_bw() +
  theme(
    axis.title.x = element_blank(),
    axis.text.x  = element_blank())
Fig2.5a

Fig2.5b <-
  ggplot(pdp_WS_NI, aes(x=WS, y=yhat, color=algorithm)) +
  geom_line(size=1) +
  scale_color_manual(values=c("Linear model"="darkgrey",
                              "Decision Tree"="orange",
                              "Random Forest"="darkgreen",
                              "Gradient Boosting"="darkblue"))  +
  ylab("Partial dependence") +
  theme_bw() +
  theme(legend.position="none", 
        axis.title.x = element_blank(),
        axis.text.x  = element_blank())
Fig2.5b

Fig2.5c <- ggplot(data_train, aes(x=NRI)) + geom_histogram() + theme_bw()
Fig2.5d <- ggplot(data_train, aes(x=WS)) + geom_histogram() + theme_bw()

Fig2.5 <- Fig2.5a + Fig2.5b + 
  Fig2.5c + Fig2.5d +
  plot_annotation(tag_levels = "a") +
  plot_layout(heights=c(9,1))

Fig2.5

# Two-way Interactions

pdp_NI_lm_1 <- model_NI_lm %>%  partial(pred.var=c("NRI","WS")) %>% autoplot + labs(title="Linear model")
pdp_NI_cart_1 <- model_NI_cart %>%  partial(pred.var=c("NRI","WS"), approx=T) %>% autoplot + labs(title="Decision Tree")
pdp_NI_rf_1 <- model_NI_rf %>%  partial(pred.var=c("NRI","WS"), approx=T) %>% autoplot + labs(title="Random Forest")
pdp_NI_gbm_1 <- model_NI_gbm %>%  partial(pred.var=c("NRI","WS"), approx=T) %>% autoplot + labs(title="Gradient Boosting")

Fig2.6 <-
  pdp_NI_lm_1 + pdp_NI_cart_1 + pdp_NI_rf_1 + pdp_NI_gbm_1

Fig2.6

# NRI*WS

pdp_NRI_WS_NI   <- rbind(
  model_NI_lm %>%  partial(pred.var=c("NRI","WS")) %>% cbind(., algorithm = "Linear model"),
  model_NI_cart %>%  partial(pred.var=c("NRI","WS"), approx=T) %>% cbind(., algorithm = "Decision Tree"),
  model_NI_rf %>%  partial(pred.var=c("NRI","WS"), approx=T) %>% cbind(., algorithm = "Random Forest"),
  model_NI_gbm %>%  partial(pred.var=c("NRI","WS"), approx=T, n.trees=500)  %>% cbind(., algorithm = "Gradient Boosting")
) 

pdp_NRI_WS_NI$algorithm <- factor(pdp_NRI_WS_NI$algorithm, levels=c("Linear model","Decision Tree","Random Forest","Gradient Boosting"))
pdp_NRI_WS_NI <- 
  pdp_NRI_WS_NI  %>% 
  mutate(WS_class = case_when(
    WS > 50 ~ "WS > 50%",
    WS < 50 ~ "WS < 50%"
  ))

Fig2.7 <-
  ggplot(pdp_NRI_WS_NI, aes(x=NRI, y=yhat, 
                            color=algorithm,
                            linetype=WS_class)) +
  facet_wrap(vars(algorithm)) +
  geom_line(stat="summary", fun=mean, size=1) +
  ylab("Partial dependence") +
  theme_bw()  +
  scale_color_manual(values=c("Linear model"="darkgrey",
                              "Decision Tree"="orange",
                              "Random Forest"="darkgreen",
                              "Gradient Boosting"="darkblue"))  

Fig2.7


### Exploratory Analysis

## Creating dataset for exploratory analysis

dataset_wmd_RC_final_ex <- dataset_wmd_RC_final[, -c(12)]
View(dataset_wmd_RC_final_ex)

## Correlation Analysis

# Creating dataset for correlation excluding categorical variables

dataset_wmd_RC_final_ex_cor <- dataset_wmd_RC_final_ex[, -c(1, 6)]
View(dataset_wmd_RC_final_ex_cor)

# Creating function for correlogram

corrplot2 <- function(data,
                      method = "pearson",
                      sig.level = 0.05,
                      order = "original",
                      diag = FALSE,
                      type = "upper",
                      tl.srt = 90,
                      number.font = 1,
                      number.cex = 1,
                      mar = c(0, 0, 0, 0)) {
  library(corrplot)
  data_incomplete <- data
  data <- data[complete.cases(data), ]
  mat <- cor(data, method = method)
  cor.mtest <- function(mat, method) {
    mat <- as.matrix(mat)
    n <- ncol(mat)
    p.mat <- matrix(NA, n, n)
    diag(p.mat) <- 0
    for (i in 1:(n - 1)) {
      for (j in (i + 1):n) {
        tmp <- cor.test(mat[, i], mat[, j], method = method)
        p.mat[i, j] <- p.mat[j, i] <- tmp$p.value
      }
    }
    colnames(p.mat) <- rownames(p.mat) <- colnames(mat)
    p.mat
  }
  p.mat <- cor.mtest(data, method = method)
  col <- colorRampPalette(c("#BB4444", "#EE9988", "#FFFFFF", "#77AADD", "#4477AA"))
  corrplot(mat,
           method = "color", col = col(200), number.font = number.font,
           mar = mar, number.cex = number.cex,
           type = type, order = order,
           addCoef.col = "black", # add correlation coefficient
           tl.col = "black", tl.srt = tl.srt, # rotation of text labels
           # combine with significance level
           p.mat = p.mat, sig.level = sig.level, insig = "blank",
           # hide correlation coefficients on the diagonal
           diag = diag
  )
}

# Creating Correlogram

corrplot2(
  data = dataset_wmd_RC_final_ex_cor,
  method = "pearson",
  sig.level = 0.05,
  order = "original",
  diag = FALSE,
  type = "upper",
  tl.srt = 75
)

## Barplot to explore the data based on the control variables

##RC_NE

# Least Developed

NE_Least <- dplyr::select(filter(dataset_wmd_RC_final_ex, DS == "Least Developed"), c(RC_NE, DS, WSL))
View(NE_Least)

NE_Least_summary <- NE_Least %>% # the names of the new data frame and the data frame to be summarised
  group_by(WSL) %>%   # the grouping variable
  summarise(mean_NE_Least = mean(RC_NE),  # calculates the mean of each group
            sd_NE_Least = sd(RC_NE), # calculates the standard deviation of each group
            n_NE = n(),  # calculates the sample size per group
            SE_NE_Least = sd(RC_NE)/sqrt(n())) # calculates the standard error of each group

View(NE_Least_summary)

NE_Least_plot <- ggplot(data=NE_Least_summary, aes(y= mean_NE_Least, x = reorder(WSL, mean_NE_Least))) +
  geom_bar(stat="identity", position = "dodge", width = 0.5, fill = c("orchid4", "red4", "blue4")) +
  geom_text(aes(label= round(mean_NE_Least, 2)), position=position_dodge(width=0.9), vjust=-0.5, hjust = -0.25) +
  scale_x_discrete(expand = waiver(),position = "bottom") + labs (x = "", y = "Relative change of number of emigrants", title = "Least Developed") + 
  theme(axis.text.x = element_text(angle = 40, hjust = 1.0, vjust = 1.0), legend.position = "bottom", plot.title = element_text(hjust = 0.5)) +
  geom_errorbar(aes(ymin = mean_NE_Least - sd_NE_Least, ymax = mean_NE_Least + sd_NE_Least), width=0.2)

NE_Least_plot

# Less Developed

NE_Less <- dplyr::select(filter(dataset_wmd_RC_final_ex, DS == "Less Developed"), c(RC_NE, DS, WSL))
View(NE_Less)

NE_Less_summary <- NE_Less %>% # the names of the new data frame and the data frame to be summarised
  group_by(WSL) %>%   # the grouping variable
  summarise(mean_NE_Less = mean(RC_NE),  # calculates the mean of each group
            sd_NE_Less = sd(RC_NE), # calculates the standard deviation of each group
            n_NE = n(),  # calculates the sample size per group
            SE_NE_Less = sd(RC_NE)/sqrt(n())) # calculates the standard error of each group

View(NE_Less_summary)

NE_Less_plot <-  ggplot(data=NE_Less_summary, aes(y= mean_NE_Less, x = reorder(WSL, mean_NE_Less))) +
  geom_bar(stat="identity", position = "dodge", width = 0.5, fill = c("orchid4", "red4", "blue4", "olivedrab", "cyan4")) +
  geom_text(aes(label= round(mean_NE_Less, 2)), position=position_dodge(width=0.9), vjust=-0.5, hjust = -0.25) +
  scale_x_discrete(expand = waiver(),position = "bottom") + labs (x = "", y = "", title = "Less Developed") + 
  theme(axis.text.x = element_text(angle = 40, hjust = 1.0, vjust = 1.0), legend.position = "bottom", plot.title = element_text(hjust = 0.5)) +
  geom_errorbar(aes(ymin = mean_NE_Less - sd_NE_Less, ymax = mean_NE_Less + sd_NE_Less), width=0.2)

plot(NE_Less_plot)

# More Developed

NE_More <- dplyr::select(filter(dataset_wmd_RC_final_ex, DS == "More Developed"), c(RC_NE, DS, WSL))
View(NE_More)

NE_More_summary <- NE_More %>% # the names of the new data frame and the data frame to be summarised
  group_by(WSL) %>%   # the grouping variable
  summarise(mean_NE_More = mean(RC_NE),  # calculates the mean of each group
            sd_NE_More = sd(RC_NE), # calculates the standard deviation of each group
            n_NE = n(),  # calculates the sample size per group
            SE_NE_More = sd(RC_NE)/sqrt(n())) # calculates the standard error of each group

View(NE_More_summary)

NE_More_plot <-  ggplot(data=NE_More_summary, aes(y= mean_NE_More, x = reorder(WSL, mean_NE_More))) +
  geom_bar(stat="identity", position = "dodge", width = 0.5, fill = c("orchid4", "red4", "blue4")) +
  geom_text(aes(label= round(mean_NE_More, 2)), position=position_dodge(width=0.9), vjust=-0.7, hjust = -0.25) +
  scale_x_discrete(expand = waiver(),position = "bottom") + labs (x = "", y = "", title = "More Developed") + 
  theme(axis.text.x = element_text(angle = 40, hjust = 1.0, vjust = 1.0), legend.position = "bottom", plot.title = element_text(hjust = 0.5)) +
  geom_errorbar(aes(ymin = mean_NE_More - sd_NE_More, ymax = mean_NE_More + sd_NE_More), width=0.2)

plot(NE_More_plot)

Fig3.2 <- NE_Least_plot + NE_Less_plot + NE_More_plot +
  plot_annotation(tag_levels = "a")

Fig3.2

##RC_NI

# Least Developed

NI_Least <- dplyr::select(filter(dataset_wmd_RC_final_ex, DS == "Least Developed"), c(RC_NI, DS, WSL))
View(NI_Least)

NI_Least_summary <- NI_Least %>% # the names of the new data frame and the data frame to be summarised
  group_by(WSL) %>%   # the grouping variable
  summarise(mean_NI_Least = mean(RC_NI),  # calculates the mean of each group
            sd_NI_Least = sd(RC_NI), # calculates the standard deviation of each group
            n_NI = n(),  # calculates the sample size per group
            SE_NI_Least = sd(RC_NI)/sqrt(n())) # calculates the standard error of each group

View(NI_Least_summary)

NI_Least_plot <- ggplot(data=NI_Least_summary, aes(y= mean_NI_Least, x = reorder(WSL, mean_NI_Least))) +
  geom_bar(stat="identity", position = "dodge", width = 0.5, fill = c("orchid4", "red4", "blue4")) +
  geom_text(aes(label= round(mean_NI_Least, 2)), position=position_dodge(width=0.9), vjust=-0.5, hjust = -0.25) +
  scale_x_discrete(expand = waiver(),position = "bottom") + labs (x = "", y = "Relative change of number of immigrants", title = "Least Developed") + 
  theme(axis.text.x = element_text(angle = 40, hjust = 1.0, vjust = 1.0), legend.position = "bottom", plot.title = element_text(hjust = 0.5)) +
  geom_errorbar(aes(ymin = mean_NI_Least - sd_NI_Least, ymax = mean_NI_Least + sd_NI_Least), width=0.2)

NI_Least_plot

# Less Developed

NI_Less <- dplyr::select(filter(dataset_wmd_RC_final_ex, DS == "Less Developed"), c(RC_NI, DS, WSL))
View(NI_Less)

NI_Less_summary <- NI_Less %>% # the names of the new data frame and the data frame to be summarised
  group_by(WSL) %>%   # the grouping variable
  summarise(mean_NI_Less = mean(RC_NI),  # calculates the mean of each group
            sd_NI_Less = sd(RC_NI), # calculates the standard deviation of each group
            n_NI = n(),  # calculates the sample size per group
            SE_NI_Less = sd(RC_NI)/sqrt(n())) # calculates the standard error of each group

View(NI_Less_summary)

NI_Less_plot <-  ggplot(data=NI_Less_summary, aes(y= mean_NI_Less, x = reorder(WSL, mean_NI_Less))) +
  geom_bar(stat="identity", position = "dodge", width = 0.5, fill = c("orchid4", "red4", "blue4", "olivedrab", "cyan4")) +
  geom_text(aes(label= round(mean_NI_Less, 2)), position=position_dodge(width=0.9), vjust=-0.5, hjust = -0.25) +
  scale_x_discrete(expand = waiver(),position = "bottom") + labs (x = "", y = "", title = "Less Developed") + 
  theme(axis.text.x = element_text(angle = 40, hjust = 1.0, vjust = 1.0), legend.position = "bottom", plot.title = element_text(hjust = 0.5)) +
  geom_errorbar(aes(ymin = mean_NI_Less - sd_NI_Less, ymax = mean_NI_Less + sd_NI_Less), width=0.2)

NI_Less_plot

# More Developed

NI_More <- dplyr::select(filter(dataset_wmd_RC_final_ex, DS == "More Developed"), c(RC_NI, DS, WSL))
View(NI_More)

NI_More_summary <- NI_More %>% # the names of the new data frame and the data frame to be summarised
  group_by(WSL) %>%   # the grouping variable
  summarise(mean_NI_More = mean(RC_NI),  # calculates the mean of each group
            sd_NI_More = sd(RC_NI), # calculates the standard deviation of each group
            n_NI = n(),  # calculates the sample size per group
            SE_NI_More = sd(RC_NI)/sqrt(n())) # calculates the standard error of each group

View(NI_More_summary)

NI_More_plot <-  ggplot(data=NI_More_summary, aes(y= mean_NI_More, x = reorder(WSL, mean_NI_More))) +
  geom_bar(stat="identity", position = "dodge", width = 0.5, fill = c("orchid4", "red4", "blue4")) +
  geom_text(aes(label= round(mean_NI_More, 2)), position=position_dodge(width=0.9), vjust=-0.7, hjust = -0.25) +
  scale_x_discrete(expand = waiver(),position = "bottom") + labs (x = "", y = "", title = "More Developed") + 
  theme(axis.text.x = element_text(angle = 40, hjust = 1.0, vjust = 1.0), legend.position = "bottom", plot.title = element_text(hjust = 0.5)) +
  geom_errorbar(aes(ymin = mean_NI_More - sd_NI_More, ymax = mean_NI_More + sd_NI_More), width=0.2)

plot(NI_More_plot)

Fig3.3 <- NI_Least_plot + NI_Less_plot + NI_More_plot +
  plot_annotation(tag_levels = "a")

Fig3.3

## Descriptive Statistics

dataset_wmd_RC_final_ex

mean <- aggregate(dataset_wmd_RC_final_ex, by= list (DS=dataset_wmd_RC_final_ex$DS), mean, na.rm=TRUE)
max <- aggregate(dataset_wmd_RC_final_ex, by= list (DS=dataset_wmd_RC_final_ex$DS), max, na.rm=TRUE)
min <- aggregate(dataset_wmd_RC_final_ex, by= list (DS=dataset_wmd_RC_final_ex$DS), min, na.rm=TRUE)
sd <- aggregate(dataset_wmd_RC_final_ex, by= list (DS=dataset_wmd_RC_final_ex$DS), sd, na.rm=TRUE)



ds <- cbind (mean, max, min, sd)

view(ds)

write_csv(ds, file ="Descriptive Statistics.csv")


n_ds <- dataset_wmd_RC_final_ex %>% # the names of the new data frame and the data frame to be summarised
  group_by(DS) %>%   # the grouping variable
  summarise(n_NE = n(),
            Percentage = (n() / nrow(dataset_wmd_RC_final_ex)) * 100)
            

n_wsl <- dataset_wmd_RC_final_ex %>% # the names of the new data frame and the data frame to be summarised
  group_by(WSL) %>%   # the grouping variable
  summarise(n_wsl = n(),
            Percentage = (n() / nrow(dataset_wmd_RC_final_ex)) * 100)
n_wsl

n_wsl



summary_overall <- dataset_wmd_RC_final_ex %>% # the names of the new data frame and the data frame to be summarised
  summarise(mean = mean(RC_NE),  # calculates the mean of each group
            sd = sd(RC_NE)) # calculates the standard deviation of each group
            
view(summary_overall)
