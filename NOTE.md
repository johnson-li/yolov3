======NOTE======

Python 3.8 has a bug that deletes shared memory on exit (https://bugs.python.org/issue39959). To solve this problem, I modified the python system module to delete the share memory only if it created.

[Not needed because there is another bug in multiprocessing in Python 3.8] Python 3.8 has a bug on AutoProxy (https://stackoverflow.com/questions/46779860/multiprocessing-managers-and-custom-classes). I added 'manager_owned=True' to 'def AutoProxy'.
