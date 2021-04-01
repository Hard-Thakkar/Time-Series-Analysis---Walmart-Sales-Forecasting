#setwd("C:/Users/K2/OneDrive - Georgia State University/GSU - MSA - Assignment Submission/2nd Semester/MSA 8200 - Predictive Analytics/Week 3")

dev.off()
rm(list=ls())

library("zoo")
library('astsa')
library("readr")
library("forecast")
library("dplyr")
library("pracma")
library("xts")
library("marima")
library("tidyverse")
library("zoo")
library("prophet")
library("data.table")
library("ggplot2")
library("reshape2")


#Importing the datasets
data = read.csv('train.csv')
feature = read.csv("features.csv")
store_raw <- read_csv("stores.csv")

####DATA EXPLORATION
# Correlation heatmap
sales_1 =data[data[,'Store']==1 & data[,'Dept']==1,]
feature$Date = as.character(feature$Date)
df<- sales_1 %>% left_join(store_raw, by = "Store") %>% left_join(feature, by = c("Store", "Date")) 
train_numeric = df %>% select(Weekly_Sales, 'IsHoliday.x', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment')
train_numeric$IsHoliday.x = as.numeric(train_numeric$IsHoliday.x)

cormat <- round(cor(train_numeric),2)
melted_cormat <- melt(cormat)

# Get lower triangle of the correlation matrix
get_lower_tri<-function(cormat){
  cormat[upper.tri(cormat)] <- NA
  return(cormat)
}
# Get upper triangle of the correlation matrix
get_upper_tri <- function(cormat){
  cormat[lower.tri(cormat)]<- NA
  return(cormat)
}

upper_tri <- get_upper_tri(cormat)
melted_cormat <- melt(upper_tri, na.rm = TRUE)

# Heatmap
reorder_cormat <- function(cormat){
  # Use correlation between variables as distance
  dd <- as.dist((1-cormat)/2)
  hc <- hclust(dd)
  cormat <-cormat[hc$order, hc$order]
}

# Reorder the correlation matrix
cormat <- reorder_cormat(cormat)
upper_tri <- get_upper_tri(cormat)
# Melt the correlation matrix
melted_cormat <- melt(upper_tri, na.rm = TRUE)
# Create a ggheatmap
ggheatmap <- ggplot(melted_cormat, aes(Var2, Var1, fill = value))+
  geom_tile(color = "white")+
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                       midpoint = 0, limit = c(-1,1), space = "Lab", 
                       name="Pearson\nCorrelation") +
  theme_minimal()+ # minimal theme
  theme(axis.text.x = element_text(angle = 45, vjust = 1, 
                                   size = 12, hjust = 1))+
  coord_fixed()

# Print the heatmap
dev.off()
print(ggheatmap)

ggheatmap + 
  geom_text(aes(Var2, Var1, label = value), color = "black", size = 4) +
  theme(
    axis.title.x = element_blank(),
    axis.title.y = element_blank(),
    panel.grid.major = element_blank(),
    panel.border = element_blank(),
    panel.background = element_blank(),
    axis.ticks = element_blank(),
    legend.justification = c(1, 0),
    legend.position = c(0.6, 0.7),
    legend.direction = "horizontal")+
  guides(fill = guide_colorbar(barwidth = 7, barheight = 1,
                               title.position = "top", title.hjust = 0.5))#



################################################################################
#DATA PROCESSING

# We will look at the weekly sales of Store 1 and Department 1
sales = data[data[,'Store']==1 & data[,'Dept']==1,3:4]
ft = feature %>%
  filter(Store==1)%>%
  select(Date,Temperature,Fuel_Price,CPI,Unemployment)


################################################################################
#MODELLING BEGINS

#1. SARIMA MODEL WITH NO ADDITIONAL PARAMETERS
# Convert the sales data into time series
sales2 = ts(sales[,2])

# Plot the weekly sales time series
par(mfrow=c(1,2))
plot(sales2, main = 'Weekly Sales') # We can see that there is no clear trend in the plot, and we can only see seasonality.
plot(diff(log(sales2)), main = 'Transformed Weekly Sales') # Plot the transformed data 


# Plot the ACF and PACF
acf2(sales2) 
# ACF tails off and PACF cuts off at lag 7, then we will try a AR(1) with seasonal period 
#of 7 for seasonality. 

# For the non-seasonal part, first we can say ACF cuts off at lag 7 and PACF tails off, 
#so we can try MA(1)
# Or we can say ACF tails off and PACF cuts off at lag 14, so we can try AR(2)
# Or, we can say ACF tails off and PACF tails off, so we try a AR(2,1) model

#Experimenting with model parameters to find the best model
sarima(sales2, 0,0,1,1,0,0,7) 
#For this model, we an see that the p values is around 0 after lag 6, which means the residuals are correlated 

sarima(sales2, 2,0,0,1,0,0,7) 
#For this model, we can see that all the p values are above 0, which means the residuals are independent 

sarima(sales2, 2,0,1,1,0,0,7) 
#For this model, we an see that the p values is 0 at lag 14 and lag 15, which means the residuals are correlated at lag 14 and lag 15 

#To sum up, among the three models, ARMA(2,0)*(1,0) is the best one, so we will choose this model for prediction

#################################
# Model Prediction
################################

# Train test split
n = length(sales2)*0.8
train = sales2[1:114]
test = sales2[115:143]

par(mfrow=c(1,1))
pred = sarima.for(train, n.ahead = 29, 2,0,0,1,0,0,7)
mean((pred$pred-test)^2)
# MSPE is 56753140

####################################################################
#ADDING ADDITINAL PARAMETERS FOR SARIMA MODEL

#1. Temperature
ft1 = merge(ft,sales)
View(ft1)

#lets play with feature - Temperature and see who our fair weather Walmart shoppers are
ftTemp <- ft1[,1:2]
#convert to time series
wmts<-ts(sales[,2],start=c(2010,6), end= c(2012,44), frequency = 52)
ftTempts<- ts(ftTemp[,2],start=c(2010,6), end= c(2012,44), frequency = 52)

#Plot of time series contained dates as points on the graph. 
#Zooreg was used to transform the time series.
ftTempz<-zooreg(ftTemp[,2], start= as.Date('2010-02-05'), end =as.Date('2012-10-26'), deltat = 7)
wmtsz<-zooreg(sales[,2], start= as.Date('2010-02-05'), end =as.Date('2012-10-26'), deltat = 7)

#lets look at our data
plot(wmts)

#Exploring the ACF and PACF of the time series
par(mfrow= c(2,1))
acf(wmts)
pacf(wmts)

#Differencing the time series, and plotting the variation of sales wrt temperature
plot(diff((wmts)))
plot(wmtsz,ftTempz)
#This looks better, we definitely see some sort of trend where sales increase as 
#temperature decreases.
#This likely has to do with the seasonality of sales and not so much outliers on
#a given day.

par(mfrow= c(3,1))
#Look at these three graphs together 
plot(wmtsz)
acf(wmtsz)
pacf(wmtsz)
#Observations:-
#We observe a yearly seasonal pattern in the sales plot

par(mfrow= c(1,1))
ccf(wmtsz,ftTempz)
#Observations:-
#From the CCF plot, we see that there's a negative correlation between the 
#temperature and the weekly sales.


#Model Fitting:
ftTemp <- ft1[,1:2]
#convert to time series
wmts<-ts(sales[,2],start=c(2010,6), end= c(2012,44), frequency = 52)
ftTempts<- ts(ftTemp[,2],start=c(2010,6), end= c(2012,44), frequency = 52)

#trying zooreg to make Time Series better
#The plot of the two series looked wonky highlighting dates instead of dots

ftTempz<-zooreg(ftTemp[,2], start= as.Date('2010-02-05'), end =as.Date('2012-10-26'), deltat = 7)
wmtsz<-zooreg(sales[,2], start= as.Date('2010-02-05'), end =as.Date('2012-10-26'), deltat = 7)

plot(wmtsz,ftTempz)
#Observations:-
#As temperature decreases, sales somewhat increase.
sarima(wmtsz,8,2,4,1,1,1,2,xreg=ftTempz)

#Splitting the training and testing dataset
train = 1:114
test = 115:143

fit <- Arima(wmtsz[train],c(8,2,4),seasonal=list(order=c(1,1,1),period=2),xreg=ftTempz[train,])
fit2 <- Arima(wmtsz[test],c(8,2,4),seasonal=list(order=c(1,1,1),period=2),xreg=ftTempz[test,],model=fit)
onestep <- fitted(fit2)

#Plotting the predicted sales from our fitted model and comparing against the test data
plot(wmtsz)
lines(time(wmtsz)[test],as.vector(onestep),col="red")

#MSPE
mean((wmtsz[test]-as.vector(onestep))^2)     
#MSPE = 8176171


#2. Unemployment
ftemp <- ft1[,c(1,5)]
#convert to time series
wmts<-ts(sales[,2],start=c(2010,6), end= c(2012,44), frequency = 52)
ftempts<- ts(ftemp[,2],start=c(2010,6), end= c(2012,44), frequency = 52)

#Using zooreg to improve the time series
ftempz<-zooreg(ftemp[,2], start= as.Date('2010-02-05'), end =as.Date('2012-10-26'), deltat = 7)
wmtsz<-zooreg(sales[,2], start= as.Date('2010-02-05'), end =as.Date('2012-10-26'), deltat = 7)

#Plot of Walmart sales by week and unemployment.
plot(wmtsz,ftempz)

#CCF plot of sales wrt Unemployment
par(mfrow= c(1,1))
ccf(wmtsz,ftempz)

#Unemployment trend wrt time
plot(ftempz)

#Model Fitting
sarima(wmtsz,8,2,4,1,1,1,2,xreg=ftempz)

#Model Prediction
train = 1:114
test = 115:143


efit <- Arima(wmtsz[train],c(8,2,4),seasonal=list(order=c(1,1,1),period=2),xreg=ftempz[train,])
efit2 <- Arima(wmtsz[test],c(8,2,4),seasonal=list(order=c(1,1,1),period=2),xreg=ftempz[test,],model=efit)
eonestep <- fitted(efit2)

#Plotting the predicted sales from our fitted model with unemployment parameter
#and comparing against the test data
plot(wmtsz)
lines(time(wmtsz)[test],as.vector(eonestep),col="red")

#MSPE 
mean((wmtsz[test]-as.vector(eonestep) )^2)   
#MSPE = 8208238

#######################################################################################
#PROPHET MODELLING
train2 = data[data[,'Store']==1 & data[,'Dept']==1,3:4]

#Renaming the columns to use in the Prophet model
train2 = rename(train2, ds = Date)
train2 = rename(train2, y = Weekly_Sales)

#Splitting the test and training datasets 80:20.
train = head(train2, 114)
test = tail(train2, 29)

###Prophet model fitted with changing points depending on how many times we saw the ACF plot switching directions (a rough estimate)
#ACF Plot
acf(train$y, lag.max = 120)
#Looking at the ACF plot, there are 5 instances where the trend changes significantly.
#Hence we define n.changepoints argument in the Prophet model = 5

#Prophet Model training
m = prophet(train, n.changepoints = 7)

future <- make_future_dataframe(m, periods = 29, freq = 'week')
forecast <- predict(m, future)

#Plotting Prophet Model components
prophet_plot_components(m, forecast)

#Plotting the forecasted sales
plot(m, forecast, xlab="Week", ylab="Sales") +
  add_changepoints_to_plot(m, threshold = 0.1, cp_color = "red", cp_linetype="dashed", trend = TRUE)

#Calculating MSPE
MSPE = mean((tail(forecast$yhat,29)-test$y)^2)
MSPE # = 14669985


#Model Experimentation
#1. Introducing Weekly Seasonality into the model
m2 = prophet(train, n.changepoints = 7, weekly.seasonality = TRUE)
forecast2 <- predict(m2, future)

#Calculating MSPE
MSPE2 = mean((tail(forecast2$yhat,29)-test$y)^2)
MSPE2 # = 15138003

#2. Modeling Holiday Effect 
train = head(data, 114)
holidays = data.frame(holiday = "holiday",
                      ds = as.Date(c(filter(train, IsHoliday==TRUE)$Date)))

train2 = data[data[,'Store']==1 & data[,'Dept']==1,3:4]

train2 = rename(train2, ds = Date)
train2 = rename(train2, y = Weekly_Sales)

#Splitting the test and training datasets 80:20.
train = head(train2, 114)
test = tail(train2, 29)

#Model Training
m3 = prophet(train, n.changepoints = 7, holidays = holidays)
forecast3 <- predict(m3, future)

#Calculating MSPE
MSPE3 = mean((tail(forecast3$yhat,29)-test$y)^2)
MSPE3 # = 15057099

#3. n.changepoints = 25; default value for prophet
m4 = prophet(train, n.changepoints = 25)
forecast4 <- predict(m4, future)

#Calculating MSPE
MSPE4 = mean((tail(forecast4$yhat,29)-test$y)^2)
MSPE4 # = 16699228

################################
#FORECASTING SALES for 14 weeks with chosen parameters beyond available data
train2 = data[data[,'Store']==1 & data[,'Dept']==1,3:4]
train2 = rename(train2, ds = Date)
train2 = rename(train2, y = Weekly_Sales)

m5 = prophet(train2, n.changepoints = 7)
future1 <- make_future_dataframe(m, periods = 43, freq = 'week')
forecast5 <- predict(m5, future1)

#Plotting the forecasted sales for the next 14 weeks
plot(m5, forecast5, xlab="Week", ylab="Sales") +
  add_changepoints_to_plot(m5, threshold = 0.1, cp_color = "red", cp_linetype="dashed", trend = TRUE)

#########################################################################################################