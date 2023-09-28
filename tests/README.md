# Running Tests Locally

To run the tests first install `pytest`:

```
pip install pytest
```

Also install all packages related to be tested:

```
pip install datasets timm
```

Then `cd` into the fastdup site-packages root directory. 

For example

```
cd /home/dnth/anaconda3/envs/tests/lib/python3.10/site-packages/fastdup
```

and run

```
pytest
```

If all test passes, the output should look like the following:

```shell
========================================================================================== test session starts ===========================================================================================
platform linux -- Python 3.10.13, pytest-7.4.2, pluggy-1.3.0
rootdir: /media/dnth/Active-Projects/fastdup
plugins: anyio-3.7.1, cov-4.1.0, mock-3.11.1
collected 5 items                                                                                                                                                                                        

tests/examples/test_datasets_example.py .                                                                                                                                                          [ 20%]
tests/unit/test_datasets.py ....                                                                                                                                                                   [100%]

============================================================================================ warnings summary ============================================================================================
tests/examples/test_datasets_example.py::test_datasets_example
  /home/dnth/anaconda3/envs/tests/lib/python3.10/site-packages/fastdup/fastdup_controller.py:385: UserWarning: No connected components found, try using a lower threshold
    warnings.warn(f'No connected components found, try using a lower threshold')

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
====================================================================================== 5 passed, 1 warning in 9.12s ======================================================================================
```

# Adding More Tests
Include new tests in the approriate folders.

+ `unit/` - Put all unit tests here. 
+ `examples/` - Put all examples tests here.