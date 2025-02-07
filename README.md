# python_research_template

A template for anyone in the Ahmed Lab looking to do python work. If this is your first time using this, you should read through our writeup on [Setting up a robust software experiment framework](https://publish.obsidian.md/ahmedlab/atoms/set+up+a+robust+software+experiment+framework#2+Using+the+Github+Template)
# Pre-requisites

- [Anaconda](https://www.anaconda.com/download)
- Have [git](https://git-scm.com) installed, and have your [SSH keys fully set up](https://publish.obsidian.md/ahmedlab/atoms/setting+up+github+SSH)

# Next Steps

0. Run `python setup.py` and follow the prompt(s)
1. Create the conda env by running `conda env create -f environment.yml`
2. Install any packages you may need either via [conda install X](https://docs.anaconda.com/free/anaconda/packages/install-packages/#installing-a-conda-package) or [pip install X](https://pip.pypa.io/en/stable/cli/pip_install/) where X is your package name.
3. After installing all your packages, run `conda env export --no-builds > environment.yml` to output a build-able version of your packages for reproducibility
4. Edit the `main.py` to be a top-level runner. This ensures consistency across all our projects.
5. Delete the contents of this `README.md` and include a descriptive writeup of your own project. Optionally, leave the section on recreating the environment below.

# Data Leakage, Reproducibility and Versioning

See the [data/README](./data/README) for more information, but TL;DR you will want to be cognizant of how you handle your data.

# Recreating the environment

Run `conda env create -f environment.yml` to recreate the conda environment. Use cases include working on a different computer, recreating our results, or using the code as a base to work off of.
