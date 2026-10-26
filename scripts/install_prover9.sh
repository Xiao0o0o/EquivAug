#!/bin/bash
# Build Prover9 / Mace4 (LADR-2009-11A) from source.
#
# Usage:
#   bash scripts/install_prover9.sh [install_dir]     # default: ./third_party
#   export PATH=<install_dir>/LADR-2009-11A/bin:$PATH
#
# On Debian/Ubuntu you can alternatively run: sudo apt-get install prover9
set -e

INSTALL_DIR=${1:-third_party}
mkdir -p $INSTALL_DIR
cd $INSTALL_DIR

if [ ! -d LADR-2009-11A ]; then
    [ -f LADR-2009-11A.tar.gz ] || wget -q https://www.cs.unm.edu/~mccune/prover9/download/LADR-2009-11A.tar.gz
    tar xzf LADR-2009-11A.tar.gz
fi
cd LADR-2009-11A

# Fixes for modern toolchains: put -lm after the objects (needed with --as-needed linkers)
sed -i 's/$(CC) $(CFLAGS) -lm -o \(.*\) ..\/ladr\/libladr.a$/$(CC) $(CFLAGS) -o \1 ..\/ladr\/libladr.a -lm/' provers.src/Makefile
# ... and allow the tentative definitions that GCC >= 10 rejects by default
make all XFLAGS=-fcommon

echo
echo "Done. Add Prover9/Mace4 to your PATH:"
echo "  export PATH=$(pwd)/bin:\$PATH"
