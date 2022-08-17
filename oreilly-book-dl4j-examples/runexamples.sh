#!/usr/bin/env bash
# Run one or all of the examples.
# Use runexamples.sh -h to see the options.

help() {
  [ -n "$1" ] && echo $1
  cat <<-EOF1
usage: $0 [-h|--help] [-a|-all] [-n|--no-pauses]
where:
  -h|--help          Show help and quit
  -a|--all           Run all the examples. Default is to prompt for which one to run.
  -n|--no-pauses     Don't pause between examples (use with --all).
}
EOF1
}

# This is shown only if all examples are executed.
banner() {
  if [ $all -eq 0 ]
  then
    cat <<-EOF2
=======================================================================

    deeplearning4j examples:

    Each example will be executed, then some of them will pop up a
    dialog with a data plot. Quit the data plot application to proceed
    to the next example.
EOF2
    if [ $pauses -eq 0 ]
    then
    cat <<-EOF2

    We'll pause after each one; hit <return> to continue or <ctrl-c>
    to quit.
EOF2
    fi
    cat <<-EOF3

=======================================================================
EOF3
  fi
}


let all=1
let pauses=0
while [ $# -ne 0 ]
do
  case $1 in
    -h|--h*)
      help
      exit 0
      ;;
    -a|--a*)
      let all=0
      ;;
    -n|--n*)
      let pauses=1
      ;;
    *)
      help "Unrecognized argument $1"
      exit 1
      ;;
  esac
  shift
done

# Most have class names that end with "Example", but not all.
# So, we use a hack; we search the Java files for "void main"
# to find all of them.

dir=$PWD
cd dl4j-examples/src/main/java

find_examples() {
  # Find all the Java files with "main" routines, then replace
  # all '/' with '.', then remove extraneous leading '.' and
  # the file extension, yielding the fully-qualified class name.
  find . -name '*.java' -exec grep -l 'void main' {} \; | \
    sed "s?/?.?g" | sed "s?^\.*\(.*\)\.java?\1?"
}

# The following works because IFS, the "field" separator is \n.
# So, substituting the result of find_examples into the the
# string and then evaluating the array definition, produces
# an array of the class names!
eval "arr=($(find_examples))"

cd $dir


# Invoke with
#   NOOP=echo runexamples.sh
# to echo the command, but not run it.
runit() {
  echo; echo "====== $1"; echo
  $NOOP java -cp ./dl4j-examples/target/dl4j-examples-*-bin.jar "$1"
}

let which_one=0
if [ $all -ne 0 ]
then

  for index in "${!arr[@]}"   # ! returns indices instead
  do
    let i=$index+1
    echo "[$(printf "%2d" $i)] ${arr[$index]}"
  done
  printf "Enter a number for the example to run: "
  read which_one
  if [ -z "$which_one" ]
  then
    which_one=0
  elif [ $which_one = 'q' ]  # accept 'q' as "quit".
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

else

  banner

  ## now loop through the above array
  for e in "${arr[@]}"
  do
    runit "$e"
    if [ $pauses -eq 0 ]
    then
      echo; echo -n "hit return to continue: "
      read toss
    fi
  done
fi
