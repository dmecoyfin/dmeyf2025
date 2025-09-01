# Sé versionar, ahora qué?

Ya conocemos la lógica básica de aditar, commitear y no mucho más. Para qué lo
hacemos?

Resulta que git nos permite ver qué modificamos antes de commitear. Supongamos
que agregamos más cambios a nuestro archivo. Lo editamos para agregarle
contenido:

```text
Un primer archivo de texto simple

Agregamos esta línea para ver qué pasó
```

Y ahora vemos qué nos dice git:
```shell
$ git status
En la rama master
Cambios no rastreados para el commit:
  (usa "git add <archivo>..." para actualizar lo que será confirmado)
  (usa "git restore <archivo>..." para descartar los cambios en el directorio de trabajo)
        modificados:     texto-simple.txt

sin cambios agregados al commit (usa "git add" y/o "git commit -a")
```

Ahora antes de agregarlo al commit, queremos ver que hicimos. Para eso git nos
proporciona una lectura diferencial con `git diff`:

```shell
$ git diff
diff --git a/texto-simple.txt b/texto-simple.txt
index 32c5551..ab9fa89 100644
--- a/texto-simple.txt
+++ b/texto-simple.txt
@@ -1 +1,3 @@
 Un primer archivo de texto simple
+
+Agregamos esta línea para ver qué pasó
```

Git nos muestra qué cambio codificado por colores, mostrando en verde y con un
signo `+` al principio las líneas que se agregaron. También nos dice donde se
modificó, en qué archivo, sobre qué líneas y nos muestra un poco de contexto
para revisar si lo que hicimos tiene sentido.

Bueno agreguémoslo y commiteemos:

```shell
$ git add texto-simple.txt
$ git commit
... # Escriban un texto que explique qué se hizo
```

Ya tenemos dos versiones del archivo. Como vemos eso?

# Ver el histórico de versiones

Git nos permite ver todas las versiones anteriores de nuestro directorio y
todos sus archivos. Usamos `git log`:

```shell
$ git log
commit a69068a1da8b2a62aaf561816ac0d8f19c6a5056 (HEAD -> master)
Author: Joaquín Ignacio Aramendía <samsagax@gmail.com>
Date:   Wed Aug 16 16:42:08 2023 -0300

    Agregamos segunda línea en texto-simple.txt

    Veamos qué pasa con nuestras versiones

commit 3f3ddc451beb8a2f868c103afd89477f55167b57
Author: Joaquín Ignacio Aramendía <samsagax@gmail.com>
Date:   Wed Aug 16 16:16:26 2023 -0300

    Generar archivo texto-simple.txt

    Se genera este archivo de prueba con contenido simple para demostración
    de las capacidades de Git.
$
```

Nos muestra el cambio más reciente arriba de todo, seguido por los anteriores.
Por ahora tenemos sólo dos cambios. Pero fíjense que muestra todo lo que
fuimos escribiendo en nuestros mensajes de commit.

Git llama a un commit por un valor único codificado en sha256. Ese valor
conjuga muchas cosas (que no son tan relevantes ahora) pero es la forma de
llamar a un commit que no le hayamos puesto "nombre". Fíjense tambien entre
parentesis aparece `(HEAD -> master)`. Esto nos indica en qué estado está
nuestro árbol de versiones y nos indica en qué rama estamos trabajando
(`master`).

# Ver una versión en particular (commit)

Esto nos da una herramienta poderosísima. Que pasa si quisieramos ver qué
cambio e nuestro último commit. Para eso usamos `git show <commit-ref>`. La
referencia a un commit se puede hacer por los primeros 6 digitos del sha256,
por el valor completo del sha256 o si el commit tiene una referencia, podemos
usarla (en este caso `HEAD`):

```shell
$ git show HEAD
commit a69068a1da8b2a62aaf561816ac0d8f19c6a5056 (HEAD -> master)
Author: Joaquín Ignacio Aramendía <samsagax@gmail.com>
Date:   Wed Aug 16 16:42:08 2023 -0300

    Agregamos segunda línea en texto-simple.txt

    Veamos qué pasa con nuestras versiones

diff --git a/texto-simple.txt b/texto-simple.txt
index 32c5551..ab9fa89 100644
--- a/texto-simple.txt
+++ b/texto-simple.txt
@@ -1 +1,3 @@
 Un primer archivo de texto simple
+
+Agregamos esta línea para ver qué pasó
```

En este caso nos muestra tanto el mensaje como los cambios en sí que se vieron
con `git diff`.

# Vista de árbol

Se puede generar una linda vista de árbol usando este comando largo

```shell
$ git log --graph --oneline --all --decorate
* a69068a (HEAD -> master) Agregamos segunda línea en texto-simple.txt
* 3f3ddc4 Generar archivo texto-simple.txt
```

Pro tip: Se puede asignar a un alias el comando para que sea más cortito:

```shell
$ git config --global alias.tree 'log --graph --oneline --all --decorate'
```

Ahora se puede obtener el mismo resultado con `git tree`

Y como encaja todo esto con la colaboración?
