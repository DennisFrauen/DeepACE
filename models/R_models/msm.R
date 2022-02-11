library(here)
library(tidyverse)
library(ipw)


load_data <- function(path, p){
  data_seq <- data.matrix(read.csv(path_data, header = FALSE), rownames.force = NA)
  T <- dim(data_seq)[2]
  n <- dim(data_seq)[1] / p
  
  
  #Convert to tensor
  lst <- lapply(split(seq_len(nrow(data_seq)),(seq_len(nrow(data_seq))-1) %/%p +1),
                function(i) data_seq[i,])
  
  data_arr <- array(0, dim=c(p,T,n))
  for(i in 1:n){
    data_arr[,,i] <- lst[[i]]
  }
  data_arr <- aperm(data_arr,c(3,2,1))
  return(data_arr)
}


#Create dataframe for ipw input
create_data_input_ipw <- function(data){
  n = dim(data)[1]
  T = dim(data)[2]
  p = dim(data)[3]
  
  cnames = vector(length = p)
  cnames[1] = "Y"
  cnames[2] = "A"
  for (i in 1:(p-2)){
    cnames[i+2] = paste("X_",i,sep="")
  }
  
  #Initialize empty dataframe
  data_ipw = as.data.frame(matrix(0, n*T, p))
  colnames(data_ipw) = cnames
  
  #Fill data
  for (i in 1:n){
    data_ipw[((i-1)*T + 1):(T*i),] = data[i,,]
  }
  
  #Add patient and time columns (time starting a 0)
  data_ipw$patient = vector(length = n*T)
  data_ipw$time = vector(length = n*T)
  for (i in 1:n){
    data_ipw[((i-1)*T + 1):(T*i), "patient"] = rep(i, T)
    data_ipw[((i-1)*T + 1):(T*i), "time"] = 0:(T-1)
  }
  return(data_ipw)
}

create_data_input_msm <- function(data_ipw,n,T){

  #Initialize empty dataframe
  data_msm = as.data.frame(matrix(0, n, T+2))
  #Name comumns
  cnames = vector(length = T+2)
  cnames[1] = "Y"
  cnames[2] = "sw"
  for(i in 1:T){
    cnames[i+2] = paste("a_",i,sep="")
  }
  colnames(data_msm) <- cnames
  #Fill df
  for (i in 1:n){
    data_msm[i, "Y"] <- data_ipw[(i-1)*T + T, "Y"]
    data_msm[i, "sw"] <- data_ipw[(i-1)*T + T, "sw"]
    for (t in 1:T){
      data_msm[i, t+2] <- data_ipw[(i-1)*T + t, "A"]
    }
  }
  return(data_msm)
}
#Start of the script------------------------------

#Input specifications and dimensions
args = commandArgs(trailingOnly=TRUE)
p <- strtoi(args[1])
method <- args[2]

#p = 9


#Load data
path <- here()
path <- paste(path,"/models/R_models", sep = "")
path_data = paste(path, "/data_seq.csv", sep = "" )
data <- load_data(path_data, p)

n <- dim(data)[1]
T <- dim(data)[2]
#a_int_1 = rep(1,T)
#a_int_2 = rep(0,T)
#Intervention sequences
a_int_1 = vector(length = T)
a_int_2 = vector(length = T)
for (i in 1:T){
  a_int_1[i] = as.numeric(args[2+i])
  a_int_2[i] = as.numeric(args[2+i + T])
}

#Format data
data_ipw <- create_data_input_ipw(data)
#Create formula for ipw denominator
cnames <- colnames(data_ipw)
form_denom = "~ "
for(i in 3:(length(cnames)-3)){
  s = paste (cnames[i], " + ", sep = "")
  form_denom = paste(form_denom, s, sep = "")
}
form_denom = paste(form_denom, cnames[length(cnames)-2], sep = "")
form_denom = as.formula(form_denom)

#Estimate stabilized inverse probability weights 
ipw <- eval(bquote(ipwtm(exposure = A, family = "binomial",numerator = ~ 1, denominator = .(form_denom),
            id = patient, timevar = time, type = "first", data = data_ipw, link = "logit", trunc = 0.01)))
data_ipw$sw <- ipw$weights.trunc 

#Format data for msm
data_msm <- create_data_input_msm(data_ipw, n, T)

#Create formula for msm
form_msm <- "Y ~ "
cnames <- colnames(data_msm)
for(i in 3:(length(cnames)-1)){
  s = paste (cnames[i], " + ", sep = "")
  form_msm = paste(form_msm, s, sep = "")
}
form_msm = paste(form_msm, cnames[length(cnames)], sep = "")
form_msm = as.formula(form_msm)
msm <- eval(bquote(lm(data = data_msm,formula = .(form_msm), weights = sw)))

#Predict using interventions
data_pred <- as.data.frame(matrix(0, 2, T+2))
colnames(data_pred) <- colnames(data_msm)
data_pred[1,3:(T+2)] <- a_int_1
data_pred[2,3:(T+2)] <- a_int_2

pred <- predict(msm, newdata = data_pred)
ace_est <- pred[1] - pred[2]
cat(ace_est)