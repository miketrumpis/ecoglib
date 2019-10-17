# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SPHINXAPIDOC = sphinx-apidoc
SOURCEDIR     = source
BUILDDIR      = build
PROJDIR = $(PWD)/../ecoglib
# this is to prevent any weirdness when importing vis stuff
QT_API := ""

# sed exec arguments would depend on linux versus mac
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
    SED = sed -i
endif
ifeq ($(UNAME_S),Darwin)
    SED = gsed -i
endif

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

apidoc:
	$(SPHINXAPIDOC) -fe --maxdepth=3 -o $(SOURCEDIR)/apidocs $(PROJDIR)
	# sphinx-apidoc doesn't allow setting maxdepth on subpackages
	# find $(SOURCEDIR)/apidocs -name "*.rst" -and -not -name "modules.rst" -exec $(SED) "s/.. toctree::/.. toctree::\n   :maxdepth: 1/g" {} +
	# sphinx toctree is very indentation sensitive, make it uniform
	# sed -i "s/    /   /g" $(SOURCEDIR)/apidocs/*.rst
	@echo
	@echo "Build finished. The apidoc files are in $(BUILDDIR)/apidocs."

gabilan_publish: apidoc html
	@echo "Copying HTML content to /var/www/"
	sudo rsync -avu --delete $(BUILDDIR)/html/ /var/www/ecoglib/

all: apidoc html
	@echo "Made"

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
