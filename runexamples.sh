#!/usr/bin/env bash
# Run one or all of the examples.

dir=$PWD

help() {
  cat <<-EOF1
usage: $0 [-e|--example class-name] [-c|--choose] [-a|--all]
where:
  -e|--example       Run a single example by class name.
  -c|--choose        List all examples and choose the one to run.
  -a|--all           Run all the examples.
}
EOF1
}

banner() {
    cat <<-EOF2
=======================================================================

    deeplearning4j examples:

    Each example will be executed, then some of them will pop up a
    dialog with a data plot. Quit the data plot application to 
    proceed to the next example.
    
    <ctrl-c> to quit.

=======================================================================
EOF2
}

# Most have class names that end with "Example", but not all.
# So, we use a hack; we search the Java files for "void main" 
# to find all of them.
find_examples() {  
  cd dl4j-examples/src/main/java
  if [ $# -eq 0 ]
    then
      query="*.java"
    else
      query="$1.java"
  fi
  # Find all the Java files with "main" routines, then replace
  # all '/' with '.', then remove extraneous leading '.' and
  # the file extension, yielding the fully-qualified class name.
  find . -name "$query" -exec grep -l 'void main' {} \; | \
    sed "s?/?.?g" | sed "s?^\.*\(.*\)\.java?\1?"
  cd $dir
}

# Invoke with NOOP=echo runexamples.sh
# to echo the command, but not run it.
runit() {
  echo; echo "====== $1"; echo
  $NOOP java -cp dl4j-examples/target/dl4j-examples-*-bin.jar "$1"
}

# Main logic starts here
let example=0
let choose=0
let all=0
let nop=0
case $1 in
  -e|--example)
    let example=1
    ;;
  -c|--choose)
    let choose=1
    ;;
  -a|--all)
    let all=1
    ;;
  *)
    help
    exit 0
    ;;
esac

if [ $example -eq 1 ] 
then

  name=$2
  if [ -z "$name" ] 
    then
      echo "ERROR: Must enter the example's class name"
      exit 1
    else
      eval "arr=($(find_examples "$name"))"
      qname="${arr[0]}"
      if [ -z "$qname" ] 
        then
          echo "ERROR: Can't find the class name"
          exit 1
        else
          runit $qname
      fi
  fi

else

  # The following works because IFS, the "field" separator is \n.
  # So, substituting the result of find_examples into the the
  # string and then evaluating the array definition, produces
  # an array of the class names!
  echo "Loading all examples..."
  eval "arr=($(find_examples))"

  if [ $choose -eq 1 ]
  then  
    let which_one=0
    for index in "${!arr[@]}" # ! returns indices instead
    do
      let i=$index+1
      echo "[$(printf "%2d" $i)] ${arr[$index]}"
    done
    printf "Enter the number of the example to run (q to quit): "
    read which_one
    if [ -z "$which_one" ]
    then
      which_one=0
    elif [ $which_one = 'q' ] # accept 'q' as "quit".
    then
      exit 0
    elif [ $which_one -le 0 -o $which_one -gt ${#arr[@]} ]
    then
      echo "ERROR: Must enter a number between 1 and ${#arr[@]}."
      exit 1
    else
      let which_one=$which_one-1
    fi

    runit ${arr[$which_one]}
  fi
  
  if [ $all -eq 1 ]
  then
    banner
    sleep 10

    ## now loop through the above array
    for e in "${arr[@]}"
    do
      runit "$e"
      sleep 2
    done
  fi

fi
