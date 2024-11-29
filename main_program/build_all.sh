#!/bash/bin

#Run each setup script
for file in setup_*.py; do
    echo "Found file; $file"
    python $file build_ext -fi
    echo "==="
done
