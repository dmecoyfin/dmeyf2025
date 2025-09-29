#limpio la memoria
rm( list=ls() )  #remove all objects
gc()             #garbage collection

require("data.table")
require("rpart")

setwd("." )  #establezco la carpeta donde voy a trabajar
#cargo el dataset
dataset  <- fread( "./competencia_01.csv", na.strings="")

mes_entrenamiento = 202102
mes_aplicacion = 202104

dataset[ foto_mes == mes_entrenamiento ,
				clase_binaria := ifelse( clase_ternaria=="BAJA+2", "POS","NEG" ) ]

dataset_entrenar <- dataset[ foto_mes == mes_entrenamiento ]
dataset_aplicar <- dataset[ foto_mes == mes_aplicacion ]

  #genero el modelo
modelo  <- rpart(formula= "clase_binaria ~ . -clase_ternaria",
                 data= dataset_entrenar,
                 model= TRUE, #quiero que me devuelva el modelo
                 xval= 0,
                 cp= -0.437,
                 minsplit= 1905,
                 maxdepth= 14,
                 minbucket= 1,
                )

#aplico el modelo a los datos en donde entrene
prediccion_train  <- predict( object=  modelo,
                   newdata= dataset_entrenar,
               type = "prob")
ganancia_train <- dataset_entrenar[ prediccion_train[, "POS"] > 1.0/40.0, sum(ifelse(clase_ternaria == "BAJA+2", 780000, -20000)) ]


prediccion_test  <- predict( object=modelo,
                               newdata=dataset_aplicar,
                               type = "prob")
ganancia_test <- dataset_aplicar[ prediccion_test[, "POS"] > 1.0/40.0, sum(ifelse(clase_ternaria == "BAJA+2", 780000, -20000)) ]

cat("Ganancia Train", ganancia_train, "\n" )
cat("Ganancia Test", ganancia_test, "\n" )

#genero el archivo para Kaggle
#fwrite( entrega,
#        file= paste0(".//altura_", vmaxdepth, ".csv"))

