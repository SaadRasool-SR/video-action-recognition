#!/usr/bin/env bash
# utils/setup_ucf101.sh

# 1) Create the data folder structure
mkdir -p data/ucf101/raw
mkdir -p data/ucf101/splits

# 2) Go into data/ucf101
cd data/ucf101

# The user might have manually downloaded UCF101.rar here in data/ucf101/
if [ -f UCF101.rar ]; then
    echo "Moving UCF101.rar into raw/ folder..."
    mv UCF101.rar raw/
fi

# 3) Now check if we have UCF101.rar in raw/ to extract
if [ -f raw/UCF101.rar ]; then
    echo "Extracting UCF101.rar in raw/..."
    cd raw
    unrar x UCF101.rar
    # If unrar created a top-level UCF-101 folder, move its contents up
    if [ -d "UCF-101" ]; then
        echo "Detected UCF-101/ top-level folder, moving contents up..."
        mv UCF-101/* .
        rmdir UCF-101
    fi
else
    echo "No UCF101.rar found in data/ucf101 or data/ucf101/raw."
    echo "Please manually download UCF101.rar and place it in data/ucf101/, then re-run this script."
fi

# 4) Back to data/ucf101
cd ..

echo "UCF101 directory setup complete."
echo "If you haven't placed the official train/test split files yet,"
echo "please put them in data/ucf101/splits/.  E.g., trainlist01.txt, testlist01.txt."
echo "Done."
