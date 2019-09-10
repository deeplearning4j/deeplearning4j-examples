# dl4j-examples_javafx

This directory contains examples using JavaFX

JavaFX is not available by default with all JVMs, and its absence can cause compilation issues.
Consequently, all JavaFX examples are contained in this standalone directory, not part of the main examples. 


### Known issues with JavaFX

If you are running on JDK 1.7 or inferior, the maven-enforcer plugin will require you to set the variable `JAVAFX_HOME` before building.
That variable should point to a directory containing `jfxrt.jar`, a file that is part of the JavaFX 2.0 distrbution.

Please set it to an instance of JavaFX that matches the JDK with which you are trying to use this project. Usually, the Sun JDK comes with JavaFX. However, OpenJDK does not and you may have to install OpenJFX, a free distribution of JavaFX.

Beware that your editor (e.g. IntelliJ) may not be using the JDK that is your system default (and that you may ancounter on the command line).

### On IntelliJ

To run the JavaFX examples from IntelliJ, you'll have to add the `jfxrt.jar` as an exernal dependency of your project. Here's a screencast on how to do it: https://youtu.be/si146q7WkSY

#### On Ubuntu

If you are using OpenJDK, on Ubuntu 16, you can install OpenJFX with `sudo apt-get install openjfx libopenjfx-java`. A typical `JAVAFX_HOME` is then `/usr/share/java/openjfx/jre/lib/ext/`. If you are on Ubuntu 14, you can install OpenJFX with the following process:

- edit `/etc/apt/sources.list.d/openjdk-r-ppa-trusty.list` and uncomment the line for deb-src
- `sudo apt-get update`
- `sudo apt-get install libicu-dev`
- `sudo aptitude build-dep libopenjfx-java`
- `sudo apt-get --compile source libopenjfx-java`
- `ls -1 *.deb|xargs sudo dpkg -i`

#### On JDK 1.8

The Sun version of JDK8 still comes with its own JavaFX, so that there should be no need to configure anything particular there and the build will succeed. If using OpenJDK8, you will still have to install OpenJFX and set `JAVAFX_HOME`, but the maven-enforcer plugin will not catch you — the reason being that it's difficult to distinguish between OpenJDK and Sun's JDK since version 8, with both adoptiong the same Vendor ID.

If you are using OpenJDK 8, install OpenJFX and set JAVAFX_HOME as indicated above. Compile with `mvn clean install -POpenJFX`
