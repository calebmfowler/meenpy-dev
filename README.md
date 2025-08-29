# Welcome to `meenpy`!

This library contains a few classes enabling succinct articulation and starage of equations, efficient assembly of systems of those equations, and powerful solving capabilities for those assembled systems. The current best examples of use are found in `test.py`, in particular `test_mized_system`.

## Installation and Usage Instructions

`meenpy` is not currently published, so you'll need to clone the library and manually pull any uppdates. As such, a prerequisite to this setup is [git](https://docs.github.com/en/get-started/git-basics/set-up-git) and some rudimentary [command line proficiency](https://swcarpentry.github.io/shell-novice/). After you have git, you can `clone` the repository to copy the source code.

```
    git clone https://github.com/calebmfowler/meenpy-dev.git
    cd meenpy-dev
```

`meenpy` uses the package manager `uv` which stores the python enviroment setup. A call to `uv` is used then to run any code, for example, to run the testing suite, you would enter the following.

```
    uv run test.py
```

Likewise, in order to use `meenpy` you simply need to make run a `*.py` file while in the `meenpy-dev` directory using

```
    uv run my_project.py
```