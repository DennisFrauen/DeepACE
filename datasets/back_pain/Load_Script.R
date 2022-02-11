library(foreign)
library(tidyverse)


lbp.data <- read.dta("./Data/casey_master_11.dta")


attach(lbp.data)

lbp.data$disability.12m <- rmprop12m   #RENAME VARIABLES
lbp.data$disability.3m <- rmprop3
lbp.data$disability.2w <- rmprop2u
lbp.data$pain.12m <- vasl12m
lbp.data$pain.3m <- vasl3m
lbp.data$pain.2w <- vasl2u

#convert mdi variables to ordinal integers
for(i in c("mdi1", "mdi2", "mdi4", "mdi5", "mdi7", "mdi8", "mdi10") )
  
{  lbp.data[,i] <- as.numeric(ifelse(lbp.data[,i]=="At no time",1,
                                     ifelse(lbp.data[,i]=="Some of the time",
                                            2,
                                            3))) %>%
  as.factor
}

for(i in c("mdi3", "mdi9"))
{  lbp.data[i] <- ifelse(lbp.data[i]=="intet tidspunkt",1,
                         ifelse(lbp.data[i]=="lidt af tiden",2,
                                ifelse(lbp.data[i]=="under halvdelen af tid",3,
                                       ifelse(lbp.data[i]=="over halvdel af tid",4,
                                              ifelse(lbp.data[i]=="det meste tid",5,6
                                              ))))) %>%
  as.numeric %>%
  as.factor

}


lbp.data = lbp.data %>% #psychological
  mutate(okon0 = okon0 %>% as.numeric,
         okom0_dic = okom0_dic %>% as.integer %>% as.numeric - 1,
         oens0 = oens0 %>% as.integer %>% as.numeric - 1,
         obeh0 = obeh0 %>% as.numeric,
         start50 = recode_factor(start50,  "Uoplyst" = NA_character_) %>% as.integer %>% as.numeric - 1,
         start60 = recode_factor(start60,  "Uoplyst" = NA_character_) %>% as.integer %>% as.numeric - 1,
         start70 = recode_factor(start70,  "Uoplyst" = NA_character_) %>% as.integer %>% as.numeric - 1,
         start80 = recode_factor(start80,  "Uoplyst" = NA_character_) %>% as.integer %>% as.numeric - 1,
         START_risk_dic = recode_factor(START_risk,  "medium" = "low") %>% as.integer %>% as.numeric - 1,
         rm150 = recode_factor(rm150,  "uoplyst" = NA_character_) %>% as.integer %>% as.numeric - 1,
         rm170 = recode_factor(rm170,  "uoplyst" = NA_character_) %>% as.integer %>% as.numeric - 1,
         rm210 = recode_factor(rm210,  "uoplyst" = NA_character_) %>% as.integer %>% as.numeric - 1,
         rm230 = recode_factor(rm230,  "uoplyst" = NA_character_) %>% as.integer %>% as.numeric - 1,
         fabq30 = fabq30 %>% as.numeric,
         fabq40 = fabq40 %>% as.numeric,
         fabq110 = fabq110 %>% as.numeric,
         fabq120 = fabq120 %>% as.numeric,
         fabq140 = fabq140 %>% as.numeric,
         mdi1 = mdi1 %>% as.integer %>% as.numeric - 1,
         mdi2 = mdi2 %>% as.integer %>% as.numeric - 1,
         mdi3 = mdi3 %>% as.integer %>% as.numeric - 1,
         mdi4 = mdi4 %>% as.integer %>% as.numeric - 1,
         mdi5 = mdi5 %>% as.integer %>% as.numeric - 1,
         mdi7 = mdi7 %>% as.integer %>% as.numeric - 1,
         mdi8 = mdi8 %>% as.integer %>% as.numeric - 1,
         mdi9 = mdi9 %>% as.integer %>% as.numeric - 1,
         mdi10 = mdi10 %>% as.integer %>% as.numeric - 1,
         # pain
         dlva0 = recode_factor(dlva0,  "uoplyst" = NA_character_) %>% as.integer %>% as.numeric - 1,
         vasl0 = vasl0 %>% as.numeric,
         vasb0_dic = vasb0_dic %>% as.integer %>% as.numeric - 1,
         tlep0 = recode_factor(tlep0,  "uoplyst" = NA_character_) %>% as.integer %>% as.numeric - 1,
         tlda0 = recode_factor(tlda0,  "uoplyst" = NA_character_) %>% as.integer %>% as.numeric - 1,
         start10 = recode_factor(start10,  "uoplyst" = NA_character_) %>% as.integer %>% as.numeric - 1,
         start20 = recode_factor(start20,  "uoplyst" = NA_character_) %>% as.integer %>% as.numeric - 1,
         start90 = start90 %>% as.integer %>% as.numeric - 1,
         rm110 = recode_factor(rm110,  "uoplyst" = NA_character_)  %>% as.integer %>% as.numeric - 1,
         fabq10 = fabq10 %>% as.numeric,
         fabq20 = fabq20 %>% as.numeric,
         domin_bp = recode_factor(domin_bp,  "uoplyst" = NA_character_)  %>% as.integer %>% as.numeric - 1,
         paraspin_debut = recode_factor(paraspin_debut,  "uoplyst" = NA_character_) %>% as.integer %>% as.numeric - 1,
         # physical impairment         
         post_latshift = recode_factor(post_latshift,  "uoplyst" = NA_character_) %>% as.integer %>% as.numeric - 1,
         post_flexdef = recode_factor(post_flexdef,  "uoplyst" = NA_character_) %>% as.integer %>% as.numeric - 1,
         mdtreduce = recode_factor(mdtreduce,  "uoplyst" = NA_character_) %>% as.integer %>% as.numeric - 1,
         mdtpartlyreduce = recode_factor(mdtpartlyreduce,  "uoplyst" = NA_character_) %>% as.integer %>% as.numeric - 1,
         mdtnonreduce = recode_factor(mdtnonreduce,  "uoplyst" = NA_character_) %>% as.integer %>% as.numeric - 1,
         mdtdysfunc = recode_factor(mdtdysfunc,  "uoplyst" = NA_character_) %>% as.integer %>% as.numeric - 1,
         herndiscr = recode_factor(herndiscr,  "uoplyst" = NA_character_) %>% as.integer %>% as.numeric - 1,
         herndiscl = recode_factor(herndiscl,  "uoplyst" = NA_character_) %>% as.integer %>% as.numeric - 1,
         affstrength = recode_factor(affstrength,  "uoplyst" = NA_character_) %>% as.integer %>% as.numeric - 1,
         affsens = recode_factor(affsens,  "uoplyst" = NA_character_) %>% as.integer %>% as.numeric - 1,
         affdtr = recode_factor(affdtr,  "uoplyst" = NA_character_) %>% as.integer %>% as.numeric - 1,
         sisep_comb = sisep_comb %>% as.integer %>% as.numeric - 1,
         siP4_comb = siP4_comb %>% as.integer %>% as.numeric - 1,
         sigaens_comb = sigaens_comb %>% as.integer %>% as.numeric - 1,
         sicompres_comb = sicompres_comb %>% as.integer %>% as.numeric - 1,
         sithrust_comb = sithrust_comb %>% as.integer %>% as.numeric - 1,
         facetextrot = facetextrot %>% as.integer %>% as.numeric - 1,
         musclepalp = recode_factor(musclepalp,  "uoplyst" = NA_character_) %>% as.integer %>% as.numeric - 1,
         musclegroup_palp = recode_factor(musclegroup_palp,  "No muscle pain" = NA_character_),
         triggerpoint = recode_factor(triggerpoint,  "uoplyst" = NA_character_) %>% as.integer %>% as.numeric - 1,
         # activity
         start30 = recode_factor(start30,  "uoplyst" = NA_character_) %>% as.integer %>% as.numeric - 1,
         start40 = recode_factor(start40,  "uoplyst" = NA_character_) %>% as.integer %>% as.numeric - 1,
         rm20 = recode_factor(rm20,  "uoplyst" = NA_character_) %>% as.integer %>% as.numeric - 1,
         rm30 = recode_factor(rm30,  "uoplyst" = NA_character_) %>% as.integer %>% as.numeric - 1,
         rm40 = recode_factor(rm40,  "uoplyst" = NA_character_) %>% as.integer %>% as.numeric - 1,
         rm50 = recode_factor(rm50,  "uoplyst" = NA_character_) %>% as.integer %>% as.numeric - 1,
         rm60 = recode_factor(rm60,  "uoplyst" = NA_character_) %>% as.integer %>% as.numeric - 1,
         rm70 = recode_factor(rm70,  "uoplyst" = NA_character_) %>% as.integer %>% as.numeric - 1,
         rm80 = recode_factor(rm80,  "uoplyst" = NA_character_) %>% as.integer %>% as.numeric - 1,
         rm90 = recode_factor(rm90,  "uoplyst" = NA_character_) %>% as.integer %>% as.numeric - 1,
         rm100 = recode_factor(rm100,  "uoplyst" = NA_character_) %>% as.integer %>% as.numeric - 1,
         rm120 = recode_factor(rm120,  "uoplyst" = NA_character_) %>% as.integer %>% as.numeric - 1,
         rm130 = recode_factor(rm130,  "uoplyst" = NA_character_) %>% as.integer %>% as.numeric - 1,
         rm140 = recode_factor(rm140,  "uoplyst" = NA_character_) %>% as.integer %>% as.numeric - 1,
         rm160 = recode_factor(rm160,  "uoplyst" = NA_character_) %>% as.integer %>% as.numeric - 1,
         rm180 = recode_factor(rm180,  "uoplyst" = NA_character_) %>% as.integer %>% as.numeric - 1,
         rm190 = recode_factor(rm190,  "uoplyst" = NA_character_) %>% as.integer %>% as.numeric - 1,
         rm220 = recode_factor(rm220,  "uoplyst" = NA_character_) %>% as.integer %>% as.numeric - 1,
         fabq50 = fabq50 %>% as.numeric,
         fabq130 = fabq130 %>% as.integer %>% as.numeric - 1,
         facetsit = recode_factor(facetsit,  "uoplyst" = NA_character_) %>% as.integer %>% as.numeric - 1,
         facetwalk = recode_factor(facetwalk,  "uoplyst" = NA_character_) %>% as.integer %>% as.numeric - 1,
         # participation
         bfbe0 = recode_factor(bfbe0,  "uoplyst" = NA_character_) %>% as.integer %>% as.numeric - 1,
         dlsy0 = dlsy0 %>% as.integer %>% as.numeric - 1,
         rm10 = recode_factor(rm10,  "uoplyst" = NA_character_)  %>% as.integer %>% as.numeric - 1,
         rm200 = recode_factor(rm200,  "uoplyst" = NA_character_) %>% as.integer %>% as.numeric - 1,
         fabq60 = fabq60 %>% as.integer %>% as.numeric - 1,
         fabq70 = fabq70 %>% as.integer %>% as.numeric - 1,
         fabq90 = fabq90 %>% as.integer %>% as.numeric - 1,
         fabq100 = fabq100 %>% as.integer %>% as.numeric - 1,
         # contextual factors
         bfor0 = recode_factor(bfor0,  "uoplyst" = NA_character_,  "ved_ikke" = NA_character_) %>% as.integer %>% as.numeric - 1,
         bryg0 = recode_factor(bryg0,  "uoplyst" = NA_character_) %>% as.integer %>% as.numeric - 1,
         fabq80 = fabq80 %>% as.numeric - 1,
         bsex0 = bsex0 %>% as.integer %>% as.numeric - 1,
         budd0 = recode_factor(budd0,  
                               "Uoplyst" = NA_character_,
                               "Anden_udd" = NA_character_) %>% 
           factor(levels = c("Faglig_udd", "Ingen_erhvervsudd", "Kort_videregående_udd", "Mellemlang_videregående_udd", "Lang_videregående_udd")) %>% 
#           fct_reorder("Faglig_udd", "Ingen_erhvervsudd", "Kort_videregående_udd", "Mellemlang_videregående_udd", "Lang_videregående_udd") %>% 
           as.integer %>% as.numeric - 1,
         barb0 = recode_factor(barb0, 
                               "uoplyst" = NA_character_,
                               "medarbejdende" = "andet",
                               "hjemmegående" = "andet"),
         nootherdisease = recode_factor(nootherdisease,  "uoplyst" = NA_character_) %>% as.integer %>% as.numeric - 1,
         heartdisease = recode_factor(heartdisease,  "uoplyst" = NA_character_) %>% as.integer %>% as.numeric - 1,
         asthma = recode_factor(asthma,  "uoplyst" = NA_character_) %>% as.integer %>% as.numeric - 1,
         psychdisease = recode_factor(psychdisease,  "uoplyst" = NA_character_) %>% as.integer %>% as.numeric - 1,
         musculoskeldisease = recode_factor(musculoskeldisease,  "uoplyst" = NA_character_) %>% as.integer %>% as.numeric - 1,
         otherchronicdisease = recode_factor(otherchronicdisease,  "uoplyst" = NA_character_) %>% as.integer %>% as.numeric - 1
         )

# multistate: 
multistate <- c("pain_dis", 
                "romflex", "romext", "romsideglr", "romsidegll", "romrotr", "romrotl",
                "musclegroup_palp", "barb0")

# note psychological:  START_risk in Excel, not in PDF
# note physical impairtment: sisepr/sisepl =: sisep_comb
# note physical impairtment: siP4r/siP4l =: siP4_comb
# note physical impairtment: sigaensr/sigaensl =: sigaens_comb
# note physical impairtment: sicompresr/sicompresl =: sicompres_comb
# note physical impairtment: sithrustr/sithrustl =: sithrust_comb
# note activity: rmprop is missing (in Excel (variables_lca), not in Table_S1.pdf)

#CREATE VARIABLE SUBSETS
outcomes_disability <- c("disability.2w",
                         "disability.3m",
                         "disability.12m")

outcomes_pain_intensity <- c("pain.2w",
                             "pain.3m",
                             "pain.12m")


all.outcomes <- c(outcomes_disability, outcomes_pain_intensity)
all.outcomes.data <- lbp.data[c(outcomes_disability, outcomes_pain_intensity)]

#LCA SUBGROUPS

inputs_lca_subgroup <- c("pp_2stage_modal", "psych_single_modal",
                         "pain_single_modal", "act_single_modal",
                         "part_single_modal", "phyimp_single_modal",
                         "context_single_modal", "pp_single_modal")

Domains <- c("pain_single_modal", "act_single_modal",
             "part_single_modal", "phyimp_single_modal",
             "context_single_modal", "psych_single_modal")

#POSTERIOR PROBABILITIES

pp.2stage <- c("pp_2stage1", "pp_2stage2", "pp_2stage3",
               "pp_2stage4", "pp_2stage5", "pp_2stage6",
               "pp_2stage7", "pp_2stage8", "pp_2stage9")

pp.psych <-  c("psych_single_1", "psych_single_2", "psych_single_3",
               "psych_single_4", "psych_single_5", "psych_single_6",
               "psych_single_7", "psych_single_8")

pp.pain <- c("pain_single_1", "pain_single_2", "pain_single_3",
             "pain_single_4", "pain_single_5", "pain_single_6",
             "pain_single_7")

pp.act <- c("act_single_1", "act_single_2", "act_single_3",
            "act_single_4", "act_single_5", "act_single_6",
            "act_single_7")

pp.part <- c("part_single_1", "part_single_2", "part_single_3",
             "part_single_4", "part_single_5", "part_single_6",
             "part_single_7")

pp.phyimp <- c("phyimp_single_1", "phyimp_single_2", "phyimp_single_3",
               "phyimp_single_4", "phyimp_single_5", "phyimp_single_6")

pp.context <- c("context_single_1", "context_single_2", "context_single_3",
                "context_single_4", "context_single_5", "context_single_6",
                "context_single_7")

pp.domains <- c(pp.psych, pp.pain, pp.act,
                pp.part, pp.phyimp, pp.context)

pp.1stage <- c("pp_single1", "pp_single2", "pp_single3",
               "pp_single4", "pp_single5", "pp_single6",
               "pp_single7")







#BASELINE VARIABLES BY HEALTH DOMAIN

baseline.part <- c("bfbe0", "dlsy0", "rm10", "rm200",
                   "fabq60", "fabq70",
                   "fabq90", "fabq100")

baseline.context <- c("bfor0", "bryg0", "fabq80", "bsex0",
                      "age", "budd0", "barb0",
                      "bhoej0", "bmi" , "htil0","nootherdisease",
                      "heartdisease", "asthma", "psychdisease",
                      "musculoskeldisease", "otherchronicdisease")

baseline.psych <- c("okon0","okom0_dic", "oens0","obeh0",
                    "start50", "start60","start70","start80",
                    "rm150", "rm170", "rm210", "rm230",
                    "fabq30", "fabq40", "fabq110","fabq120", "fabq140",
                    "mdi1", "mdi2", "mdi3", "mdi4", "mdi5",
                    "mdi7", "mdi8", "mdi9", "mdi10")

baseline.pain <- c("dlva0", "vasl0", "vasb0_dic", "tlep0", "tlda0",
                   "start10", "start20", "start90", "rm110", "fabq10",
                   "fabq20", "pain_dis", "domin_bp", "paraspin_debut" )

baseline.phys.impair <- c("post_latshift", "post_flexdef",
                          "romflex", "romext",
                          "romsideglr","romsidegll",
                          "romrotr", "romrotl","mdtreduce",
                          "mdtpartlyreduce", "mdtnonreduce",
                          "mdtdysfunc",
                          "herndiscr", "herndiscl", "affstrength",
                          "affsens", "affdtr",
                          "sisep_comb", "siP4_comb",
                          "sigaens_comb", "sicompres_comb",
                          "sithrust_comb", "facetextrot", "musclepalp",
                          "musclegroup_palp", "triggerpoint")

baseline.activity <- c("start30", "start40", "rm20",
                       "rm30", "rm40", "rm50", "rm60",
                       "rm70", "rm80", "rm90",
                       "rm100", "rm120", "rm130",
                       "rm140","rm160","rm180", "rm190",
                       "rm220", "fabq50",
                       "fabq130", "facetsit", "facetwalk")

domain_variables <- list(
  "Pain" = baseline.pain,
  "Activity" = baseline.activity,
  "Context" = baseline.context,
  "Partner" = baseline.part,
  "Physical Impairment" = baseline.phys.impair,
  "Psychology" = baseline.psych
)

all.baseline <- c(baseline.part, baseline.context, baseline.psych,
                  baseline.pain, baseline.phys.impair,
                  baseline.activity)

all.vars <- c(all.baseline, inputs_lca_subgroup)

existing.tools <- c("START_risk", "quebecclass",
                    "okom0_full","vasb0_full",
                    "vasl0", "rmprop")

#CHANGE DOMAIN SPECIFIC SUBGROUPS TO FACTOR

lbp.data$psych_single_modal   <- as.factor(psych_single_modal)
lbp.data$pain_single_modal    <- as.factor(pain_single_modal)
lbp.data$act_single_modal     <- as.factor(act_single_modal)
lbp.data$part_single_modal    <- as.factor(part_single_modal)
lbp.data$phyimp_single_modal  <- as.factor(phyimp_single_modal)
lbp.data$context_single_modal <- as.factor(context_single_modal)


#CREATE BINARY OUTCOME VARIABLES FOR SIMPLE CLASSIFICATION MODELS

lbp.data$recovered.12m <- as.factor(ifelse(gen12m == "Completely recovered" ,
                             "RECOVERED",
                             ifelse( gen12m == "Much improved",
                                    "RECOVERED", "SUFFERING")))


lbp.data$recovered.2w <- as.factor(ifelse(gen2u == "meget bedre" ,
                             "RECOVERED",
                             ifelse( gen12m == "Bedre",
                                     "RECOVERED", "SUFFERING")))

# library(caret)
# foo <- dummyVars("~ .", data = lbp.data[, all.baseline, drop=FALSE], sep='-')
# X <- predict(foo, newdata = lbp.data[, all.baseline, drop=FALSE]) %>% as.data.frame
# dim(X)
# colnames(X)

#Save outcomes and dataset
write.csv(all.outcomes.data, "./data_preprocessed/outcomes.csv", row.names = FALSE)
baselines = lbp.data[all.baseline]
write.csv(baselines, "./data_preprocessed/baselines.csv", row.names = FALSE)
write.csv(lbp.data, "./data_preprocessed/lbp.csv", row.names = FALSE)
#Text information
write(all.baseline, "./data_preprocessed/baseline_names.txt")
write(inputs_lca_subgroup, "./data_preprocessed/subgroup_names.txt")
