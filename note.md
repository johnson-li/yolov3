======NOTE======

Python 3.8 has a bug that deletes shared memory on exit (https://bugs.python.org/issue39959). To solve this problem, I modified the python system module to delete the share memory only if it created.

