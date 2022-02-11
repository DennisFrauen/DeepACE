library(ltmle)
library(here)
library(SuperLearner)
library(arm)
library(xgboost)
library(randomForest)

set.seed(23482374)

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


#Create dataframe for ltmle input
create_data_input <- function(data){
  n = dim(data)[1]
  T = dim(data)[2]
  p = dim(data)[3]
  L_nodes <- c()
  A_nodes <- c()
  Y_nodes <- c()
  data_tmle <- data.frame(matrix(ncol = 0, nrow = n))
  for (t in 1:T){
    #Covariates
    X_t <- data[,t,3:p]
    for (i in 1:(p-2)){
      name_X <- paste("X_",t,sep="")
      name_X <- paste(name_X,"_",sep="")
      name_X <- paste(name_X,i,sep="")
      data_tmle[name_X] <- X_t[,i]
      #Baseline covariates are not included in L_nodes
      if (t > 1){
        L_nodes <- c(L_nodes,name_X)
      }
    }
    #Treatment
    A_t <- data[,t,2]
    name_A <- paste("A_",t,sep="")
    data_tmle[name_A] <- A_t
    A_nodes <- c(A_nodes,name_A)
  }
  #Censoring node (no censoring)
  data_tmle["C"] <- factor(rep("uncensored", n))
  cnode <- vector(length = 1)
  cnode[1] = "C"
  C_nodes <- cnode
  #Outcome
  Y <- data[,T,1]
  data_tmle["Y"] <- Y
  Y_nodes <- "Y"

  return(list(data_tmle = data_tmle, L_nodes = L_nodes, A_nodes = A_nodes, Y_nodes = Y_nodes,
              C_nodes = C_nodes))
}


#Start of the script------------------------------

#Input specifications and dimensions
args = commandArgs(trailingOnly=TRUE)
p <- strtoi(args[1])
method <- args[2]

#p = 138

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

#Create ltmle dataframe
ltmle_list = create_data_input(data)

#Choose method
if(method == "ltmle"){
  result <- suppressMessages(ltmle(ltmle_list$data_tmle, Anodes = ltmle_list$A_nodes, Lnodes=ltmle_list$L_nodes,
                                   Ynodes = ltmle_list$Y_nodes, Cnodes = ltmle_list$C_nodes, 
                                   abar = list(treament = a_int_1, control = a_int_2), 
                                   SL.library = "glm", SL.cvControl = list(V = 5)))
  sum <- summary(result)
  ate_est <- sum$effect.measures$ATE$estimate
  cat(ate_est)
}

if(method == "ltmle_super"){
  SL.lib <- c("SL.glm","SL.randomForest", "SL.xgboost", "SL.gam")
  result <- suppressMessages(ltmle(ltmle_list$data_tmle, Anodes = ltmle_list$A_nodes, Lnodes=ltmle_list$L_nodes,
                                   Ynodes = ltmle_list$Y_nodes, abar = list(treament = a_int_1, control = a_int_2),  
                                   SL.library = list(Q = SL.lib, g = SL.lib), Cnodes = ltmle_list$C_nodes, 
                                   SL.cvControl = list(V = 3), estimate.time = FALSE))
  sum <- summary(result)
  ate_est <- sum$effect.measures$ATE$estimate
  cat(ate_est)
}

if(method == "gcomp"){
  result <- suppressMessages(ltmle(ltmle_list$data_tmle, Anodes = ltmle_list$A_nodes, Lnodes=ltmle_list$L_nodes,
                                   Ynodes = ltmle_list$Y_nodes, abar = list(treament = a_int_1, 
                                control = a_int_2), gcomp = TRUE, Cnodes = ltmle_list$C_nodes))
  sum <- summary(result)
  ate_est <- sum$effect.measures$ATE$estimate 
  cat(ate_est)
}



