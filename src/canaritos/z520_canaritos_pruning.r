#limpio la memoria
rm( list=ls() )  #remove all objects
gc()             #garbage collection

require("data.table")
require("rpart")
require("rpart.plot")

setwd("." )  #establezco la carpeta donde voy a trabajar
#cargo el dataset
dataset  <- fread( "./competencia_01.csv", na.strings="")

#uso esta semilla para los canaritos
set.seed(1021)

mes_entrenamiento = 202102
mes_aplicacion = 202104

dataset[ foto_mes == mes_entrenamiento ,
				clase_binaria := ifelse( clase_ternaria=="BAJA+2", "POS","NEG" ) ]

#agrego 10 canaritos
for( i in 1:10 ) dataset[ , paste0("canarito", i ) :=  runif( nrow(dataset)) ]

dtrain <- dataset[ foto_mes==mes_entrenamiento ]
dapply <- dataset[ foto_mes==mes_aplicacion ]

#Primero  veo como quedan mis arboles
modelo_original <- rpart(
    formula= "clase_binaria ~ . -clase_ternaria",
    data= dtrain,
    model= TRUE,
    xval= 0,
    cp= -1,
    minsplit= 2, # dejo que crezca y corte todo lo que quiera
    minbucket= 1,
    maxdepth= 30 )

#hago el pruning de los canaritos
#haciendo un hackeo a la estructura  modelo_original$frame
# -666 es un valor arbritrariamente negativo que jamas es generado por rpart
modelo_original$frame[ modelo_original$frame$var %like% "canarito", "complexity"] <- -666
modelo_pruned  <- prune(  modelo_original, -666 )

prediccion_test  <- predict( object=modelo_pruned,
                                 newdata=dapply,
                                 type = "prob")

ganancia_test <- dapply[ prediccion_test[, "POS"] > 1.0/40.0, sum(ifelse(clase_ternaria == "BAJA+2", 780000, -20000)) ]

cat("Ganancia Test: ", ganancia_test, "\n")


pdf(file = "./stopping_at_canaritos.pdf", width=28, height=4)
prp(modelo_pruned, extra=101, digits=5, branch=1, type=4, varlen=0, faclen=0)
dev.off()

