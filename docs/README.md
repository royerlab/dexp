# **dexp** documentation

**dexp** documentation dependencies can be installed with `pip`

```shell script
  python -m pip install -r requirements-docs.txt
```

or `conda`

```shell script
  conda create -n dexpdocs -c conda-forge --file requirements-docs.txt
  conda activate dexpdocs
```


After running ``make html`` the generated HTML documentation can be found in
the ``build/html`` directory. Open ``build/html/index.html`` to view the home
page for the documentation.
