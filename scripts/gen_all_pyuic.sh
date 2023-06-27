!/bin/bash

for ui in *.ui; do
    out=$(echo "$ui" | sed 's/.ui/_ui.py/')
    echo "Converting UI: $ui"
    pyside6-uic "$ui" -o "$out"
done
