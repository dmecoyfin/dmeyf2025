#limpio la memoria
rm( list=ls() )  #remove all objects
gc()             #garbage collection

require("data.table")
require("rpart")
require("rpart.plot")

setwd("." )  #establezco la carpeta donde voy a trabajar
#cargo el dataset
dataset  <- fread( "./competencia_01.csv")

#uso esta semilla para los canaritos
set.seed(102191)

#agrego una variable canarito, random distribucion uniforme en el intervalo [0,1]
dataset[ ,  canarito1 :=  runif( nrow(dataset) ) ]

#agrego los siguientes canaritos
for( i in 13:100 ) dataset[ , paste0("canarito", i ) :=  runif( nrow(dataset)) ]


#Primero  veo como quedan mis arboles
  modelo  <- rpart(formula= "clase_ternaria ~ .",
                 data= dataset[ foto_mes==202102 ,],
                 model= TRUE,
                 xval= 0,
                 cp= 0,
                 minsplit= 10,
                 maxdepth= 10)


pdf(file = "./arbol_canaritos.pdf", width=28, height=4)
prp(modelo, extra=101, digits=5, branch=1, type=4, varlen=0, faclen=0)
dev.off()

