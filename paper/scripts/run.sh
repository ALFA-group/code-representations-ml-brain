set -e
echo "making paper plots and tables..."
mkdir -p ../plots ../tables/raw ../tables/latex ../stats/raw ../stats/latex
python scores.py
python stats.py
python plots.py
python tables.py
