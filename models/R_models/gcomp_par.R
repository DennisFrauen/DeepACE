library(here)
library(gfoRmula)

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

#Create dataframe for gcomp input
create_data_input_gcomp <- function(data){
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

#p = 13


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

#Format data
data_gcomp <- create_data_input_gcomp(data)

#Interventions
intvars <- list('A', 'A')
interventions <- list(list(c(static, a_int_1)),
                      list(c(static, a_int_2)))
int_descript <- c('a_int_1', 'a_int_2')

#Covariates including treatment
covnames <- colnames(data_gcomp)[2:p]
covtypes = c("binary" ,rep("normal", p-2))

#Lagged histories
histories <- c(lagged)
histvars <- list(covnames)
covnames_lagged <- vector(length = length(covnames))
for (i in 1:length(covnames)){
  covnames_lagged[i] <- paste("lag1_", covnames[i], sep = "")
}

#Outcome model
ymodel <- create_formula_from_strings("Y", c(covnames, covnames_lagged))

covparams <- list(covmodels = vector(mode = "list", length = length(covnames)))
#Treatment model
a_cov = c(covnames[2:length(covnames)], covnames_lagged)
covparams$covmodels[[1]] <- create_formula_from_strings("A", a_cov)
#Covariate models
for (i in 2:length(covnames)){
  resp <- covnames[i]
  covparams$covmodels[[i]] <- create_formula_from_strings(resp, covnames_lagged)
}


result <- suppressMessages(gformula(obs_data = data_gcomp, id = "patient", time_name = "time", outcome_name = "Y", 
                   outcome_type = "continuous_eof", ymodel = ymodel, covnames = covnames, 
                   covtypes = covtypes, covparams = covparams, intvars = intvars, interventions = interventions,
                   nsimul = 1000, seed = 123, histories = histories, histvars = histvars))
ace <- result$result$`g-form mean`[2] - result$result$`g-form mean`[3]
cat(ace)