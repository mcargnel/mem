#!/bin/bash
# LaTeX cleanup script - removes auxiliary files generated during compilation

echo "Cleaning up LaTeX auxiliary files..."

# Remove common LaTeX auxiliary files
rm -f main.aux main.lof main.log main.lot main.out main.toc

# Remove any .synctex.gz files (used for forward/inverse search)
rm -f *.synctex.gz

# Remove any .bbl and .blg files (bibliography related)
rm -f *.bbl *.blg

echo "Cleanup complete!"
