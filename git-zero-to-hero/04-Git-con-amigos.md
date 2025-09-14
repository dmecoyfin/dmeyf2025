# Repositorios remotos

En general los sistemas de versionado se utilizan para trabajar
colaborativamente. Para eso uno tiene que publicar su código en algun lado
para que otros lo vean y puedan usarlo y agregar sus cambios.

Rara vez uno tiene un repositorio solo. En general lo usa en algun servidor
como GitHub o GitLab. Para eso git tiene un manejo de repositorios remotos con
`git remote`.

Dentro de nuestro repo probemos:

```shell
$ git remote
$
```

No pasó nada? Agreguemos una `-v` (de verborragia):
```shell
$ git remote -v
$
```

Nada? Bueno, tiene sentido. Nuestro repositorio es local y nunca lo bajamos de
otro lado ni lo publicamos. Probemos tomando un repositorio de algun otro lado

# Clonando repositorios

Busquemos uno en GitHub: el de la materia, por supuesto! https://github.com/dmecoyfin/dmeyf2023

Navegamos a algún lugar de nuestros directorios (uno que no sea ya un
repositorio git) y hacemos `git clone`:
```shell
$ git clone https://github.com/dmecoyfin/dmeyf2023.git
$ cd dmeyf2023
```

Nota: para la materia van a tener que tener su propio repositorio derivado
(fork) del de la materia. Estos pasos son ilustrativos.

Ahora probemos de vuelta que pasa con los remotos:

```shell
$ git remote
origin
$
```

Bueno es algo. Sabemos que existe un remoto que se llama `origin`. Es el
nombre por defecto del remoto de donde se clonó el repo. También podemos ver
más detalle agregando `-v`:

```shell
$ git remote -v
origin  https://github.com/dmecoyfin/dmeyf2023.git (fetch)
origin  https://github.com/dmecoyfin/dmeyf2023.git (push)
```

# Ramas

Es en general buena práctica no hacer commits directos a la rama principal
(`main` o `master`) de un repositorio remoto. Uno hace su código en una rama
propia "descartable" y luego mezcla (`merge`) los cambios a la rama principal.

Pero qué son las ramas? Hablamos de que git maneja un árbol de versiones. Uno
puede tener varias versiones basadas en versiones anteriores y que éstas
diverjan desde un punto común. Se dice que el árbol se dividió en ramas.

Para crear ramas uno puede hacerlo con el commando `git branch`:

```shell
$ git branch mi-version-alternativa
$
```

Con esto, git creó la rama pero no cambió a ella. Para eso hay que usar el
comando `git checkout`:

```shell
$ git checkout mi-version-alternativa
Cambiado a rama 'mi-version-alternativa'
$ git status
En la rama mi-version-alternativa
nada para hacer commit, el árbol de trabajo está limpio
```

> ;) Y qué pasaría si usamos nuestro `git tree` ahora?

Todos los commits que hagamos ahora se harán sobre la rama actual sin afectar
al resto.

# Reconectar ramas

Ahora qué pasa cuando uno quiere incorporar los cambios de una rama a la
principal? Fácil, se usa el comando `git merge`. Pero tiene unas cositas
a tener en cuenta. Uno debe estar en la rama a la que quiere traer los cambios
(moverse con `git checkout` y verificar con `git status`). Luego usar
`git merge <nombre de rama a traer>`. Git hará lo posible por mezclar los
cambios. Si surge un conflicto en el `git status` aparecerá y nos dará las
instrucciones de como avanzar una vez resuelto el conflicto o como deshacer el
merge.

## Conflictos y como resolverlos

Cuando uno hace cambios en una rama en un archivo y otro desarrollador hace
cambios en otra rama pero en el mismo archivo, Git trata de encontrar la forma
de reconciliar los cambios en un único archivo (un archivo úncio mezclado).

Mientras los cambios estén en contextos distintos, Git encontrará la forma de
mezclarlos.

Veamos como se ve esto. Supongamos que primero hacemos una rama `conflictos`
en nuetro "primer-repo". Luego editamos nuestro archivo de texto. Cambiamos a
la rama `master` y volvemos a editar el mismo archivo en el mismo lugar.

Nuestro árbol se verá así:

```shell
$ git tree
* e73a030 (HEAD -> master) Editamos la línea final
| * 1eaac87 (conflictos) Agregamos conflictos
|/
* a69068a Agregamos segunda línea en texto-simple.txt
* 3f3ddc4 Generar archivo texto-simple.txt
$
```

El mismo archivo ahora tiene dos versiones en cada rama!

En la rama `master`:
```
Un primer archivo de texto simple

Agregamos esta línea para ver qué pasó

Editamos esta línea en la rama 'master'
```

y en la rama `conflcitos`:
```
Un primer archivo de texto simple

Agregamos esta línea para ver qué pasó

Agregamos esta línea en la rama 'conflictos'
```

Hagamos un `git merge` desde la rama master:
```shell
$ git checkout master
$ git merge conflictos
Auto-fusionando texto-simple.txt
CONFLICTO (contenido): Conflicto de fusión en texto-simple.txt
Fusión automática falló; arregle los conflictos y luego realice un commit con el resultado.
$ git status
En la rama master
Tienes rutas no fusionadas.
  (arregla los conflictos y ejecuta "git commit"
  (usa "git merge --abort" para abortar la fusion)

Rutas no fusionadas:
  (usa "git add <archivo>..." para marcar una resolución)
        modificados por ambos:  texto-simple.txt

sin cambios agregados al commit (usa "git add" y/o "git commit -a")
```

Cuando Git no pueda mezclar archivos automágicamente se indica en la salida
del comando `git merge`. En rojo aparecerán los archivos con conflictos al hacer
un `git status` marcados como `modificado por ambos`. Y Git nos marcará en
donde están esos conflictos de esta manera al abrir el archivo:

```
Un primer archivo de texto simple

Agregamos esta línea para ver qué pasó
<<<<<<< HEAD

Editamos esta línea en la rama 'master'
||||||| a69068a
=======

Agregamos esta línea en la rama 'conflictos'
>>>>>>> conflictos
```

Mucho para ver. Pero vamos por partes. Las primeras líneas no se modificaron
por lo que resultan contexto para las modificaciones. Luego Git nos muestra un
marcador:

```
<<<<<<< HEAD
```

Una serie de cambios. Luego otro marcador:

```
||||||| a69068a
=======
```

Otra serie de cambios. Y luego un cierre:

```
>>>>>>> conflictos
```

Es la forma de Git de decirnos:

- Desde acá hasta acá es lo que se modificó en la rama actual (lo que esta entre
los primeros dos marcadores)
- Y desde acá hasta acá es lo que se modificó por la otra rama (entre los
últimos dos marcadores). E incluso nos da el nomrbe de la rama al final.

Para resolver el conflicto tenemos que decidir qué línea debe quedar. Es decir,
tenemos que editar el archivo para que quede en su forma final, sin marcadores y
con el contenido que nosotros queremos.

Una vez editado, Git nos dice que se lo indiquemos con `git add` y luego
`git commit`:

```shell
$ git add texteo-simple.txt
$ git commit
... Editamos el comentario del commit o aceptamos el que viene por defecto
```

Y así nos queda nuestro árbol al final:
```shell
$ git tree
*   520f014 (HEAD -> master) Merge branch 'conflictos'
|\
| * 1eaac87 (conflictos) Agregamos conflictos
* | e73a030 Editamos la línea final
|/
* a69068a Agregamos segunda línea en texto-simple.txt
* 3f3ddc4 Generar archivo texto-simple.txt
```
