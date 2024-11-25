## How to run analysis tool!
Simply go into the main directory and run main.py with chosen testcase path as argument.

So open a terminal and do "py main.py ..\test_cases\test_case_binomial\" for binomial test. 

This requires that chosen path contains streams.csv and topology.csv files.

This results in the creation of solution.csv in given directory, which contains result for each stream, overall result, and runtime.

Warning: The tool does not expect headers on the stream.csv or topology.csv files!