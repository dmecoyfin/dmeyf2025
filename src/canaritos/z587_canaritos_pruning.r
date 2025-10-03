# limpio la memoria
rm(list = ls()) # remove all objects
gc() # garbage collection

require("data.table")
require("rpart")
require("rpart.plot")

setwd(".") # establezco la carpeta donde voy a trabajar

# cargo el dataset
dataset <- fread("./competencia_01.csv")

dir.create("./exp/", showWarnings = FALSE)
dir.create("./exp/EA5870/", showWarnings = FALSE)
setwd("./exp/EA5870")


# uso esta semilla para los canaritos
set.seed(102191)


# agrego canaritos randomizados
dataset2 <- copy(dataset)
# quito algunas variables de dataset2
dataset2[ , numero_de_cliente := NULL ]
dataset2[ , clase_ternaria := NULL ]
dataset2[ , foto_mes := NULL ]

# agrego azar
dataset2[ , azar := runif( nrow(dataset2) ) ]
# randomizo, manteniendo las relaciones entre las variables
setorder( dataset2, azar )
dataset2[ , azar := NULL ]  # borra azar

columnas <- copy(colnames(dataset2))

# creo efectivamente los canaritos
#  1/5  de las variables del dataset
for( i in sample( 1:ncol(dataset2) , round( ncol(dataset)/5 ) )  )
{
  dataset[, paste0("canarito", i) :=  dataset2[ , get(columnas[i]) ]  ]
}



dtrain <- dataset[foto_mes == 202102]
dapply <- dataset[foto_mes == 202104]

# Dejo crecer el arbol sin ninguna limitacion
# sin limite de altura ( 30 es el maximo que permite rpart )
# sin limite de minsplit ( 2 es el minimo natural )
# sin limite de minbukcet( 1 es el minimo natural )
# los canaritos me protegeran
modelo_original <- rpart(
    formula = "clase_ternaria ~ .",
    data = dtrain,
    model = TRUE,
    xval = 0,
    cp = -1,
    minsplit = 2, # dejo que crezca y corte todo lo que quiera
    minbucket = 1,
    maxdepth = 30
)


# hago el pruning de los canaritos
# haciendo un hackeo a la estructura  modelo_original$frame
# -666 es un valor arbritrariamente negativo que jamas es generado por rpart
modelo_original$frame[
    modelo_original$frame$var %like% "canarito",
    "complexity"
] <- -666

modelo_pruned <- prune(modelo_original, -666)

#prediccion <- predict(modelo_pruned, dapply, type = "prob")[, "BAJA+2"]

#entrega <- as.data.table(list(
#    "numero_de_cliente" = dapply$numero_de_cliente,
#    "Predicted" = as.integer(prediccion > 0.025)
#))

prediccion_test  <- predict( object=modelo_pruned,
                                 newdata=dapply,
                                 type = "prob")

ganancia_test <- dapply[ prediccion_test[, "POS"] > 1.0/40.0, sum(ifelse(clase_ternaria == "BAJA+2", 780000, -20000)) ]

cat("Ganancia Test: ", ganancia_test, "\n")

#fwrite(entrega, paste0("stopping_at_canaritos.csv"), sep = ",")

