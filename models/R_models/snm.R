library(here)
library(gesttools)

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

#Create dataframe for snm input
create_data_input_snm <- function(data){
  n = dim(data)[1]
  T = dim(data)[2]
  p = dim(data)[3]
  
  cnames = vector(length = p+2)
  cnames[1] = "id"
  cnames[2] = "time"
  cnames[3] = "Y"
  cnames[4] = "A"
  for (i in 1:(p-2)){
    cnames[i+4] = paste("X_",i,sep="")
  }
  #Initialize empty dataframe
  data_snm = as.data.frame(matrix(0, n*T, p+2))
  colnames(data_snm) = cnames
  
  #Add patient and time columns (time starting a 1)
  for (i in 1:n){
    data_snm[((i-1)*T + 1):(T*i), "id"] = rep(i, T)
    data_snm[((i-1)*T + 1):(T*i), "time"] = 1:T
  }
  
  #Fill data
  for (i in 1:n){
    data_snm[((i-1)*T + 1):(T*i),3:(p+2)] = data[i,,]
  }
  
  #Add 1lag histories
  cnames_lag <- vector(length = p)
  cnames_lag[1] <- "Lag1_Y"
  cnames_lag[2] <- "Lag1_A"
  for (i in 1:(p-2)){
    cnames_lag[i+2] <- paste("Lag1_X", i, sep = "")
  }
  data_lag <- as.data.frame(matrix(0, n*T, p))
  colnames(data_lag) <- cnames_lag
  data_lag[2:(n*T),] <- data_snm[1:(n*T)-1,3:(p+2)]
  return(cbind(data_snm, data_lag))
}

create_formula_from_strings <- function(response, covariates){
  form_string <- paste(response, " ~ ", sep = "")
  #Add covariates on right hand side
  for(i in 1:(length(covariates)-1)){
    form_string = paste (form_string, covariates[i], sep = "")
    form_string = paste (form_string, " + ", sep = "")
  }
  form_string = paste(form_string, covariates[length(covariates)], sep = "")
  form = as.formula(form_string)
  return(form)
}


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

if (!identical(rep(0,T), a_int_2)){
  print("No ACE can be calculated if treatment is applied in second intervention seq")
  cat(0)
} else{
  #Format data
  data_snm <- create_data_input_snm(data)
  #Outcome models
  y1_model <- create_formula_from_strings("Y", colnames(data_snm)[4:(p+2)])
  y_model <- create_formula_from_strings("Y", colnames(data_snm)[4:length(colnames(data_snm))])
  outcomemodels <- vector(mode = "list", length = T)
  outcomemodels[[1]] <- y1_model
  for (t in 2:T){
    outcomemodels[[t]] <- y1_model
  }
  
  #Propensity model
  #propensitymodel <- create_formula_from_strings("A", colnames(data_snm)[5:length(colnames(data_snm))])
  propensitymodel <- create_formula_from_strings("A", colnames(data_snm)[5:(p+2)])
  
  #G-estimation
  geest <- gestSingle(data = data_snm, idvar = "id", timevar = "time", 
                      Yn = "Y", An = "A", outcomemodels = outcomemodels, 
                      propensitymodel = propensitymodel, type = 3)
  #Return ACE (SNM implies MSN if no treatment is applied in a_int_2)
  ace = sum(a_int_1*geest$psi)
  cat(ace)
}







