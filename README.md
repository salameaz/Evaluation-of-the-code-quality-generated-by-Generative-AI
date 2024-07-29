# Evaluation-of-the-code-quality-generated-by-GENERATION-AI-Dataset

This is the repository of the all of the data used in the paper "Evaluation of the code quality generated by GENERATION AI", From the **Promts** for the generative AI to there **Answers** and the **Evaluation Results**

# Instructions for recreating the experiment
Process Overview
----------------

1. **Data Collection**
   - A dataset of problems is collected from LeetCode.(you can use the problems we chose availabe [here](original_prompts.csv))
   - Two subsets of problems are created: an experiment problems dataset and a benchmark problems dataset(we used systematic sampling).

2. **Code Generation**
   - The AI models used in the benchmark are:
   - ChatGPT 4.0, Microsoft Copilot(free version), Claude 3.0 and google gemini are used to generate code solutions for the problems in the benchmark problem datasets.

3. **Generated Code Compilation**
   - The generated code from each model is compiled into a one [file](Benchmark_answers.csv)

4. **Code Quality Evaluation**
   - The generated code is evaluated on three primary metrics:
     - **Code correctness**:  will be measured on a scale of 5 levels
         o correct: The code passes all test cases
         o Partially correct: the code has no errors, but the output differs from the expected output in at most 50% of the test cases.
         o incorrect: The code has no errors, but the output differs from the expected output in more than 50% of the test cases.
         o Compilation error: The submitted code cannot be compiled.
         o Runtime error: The code fails in at least one test case due to a runtime error (ie division by zero, etc.).
      - **Code Readability**: will be measured using a checklist as a tool for evaluating the generated code based on predefined rules.
         The rules:
            ● A line should not be longer than 120 characters.
            ● The length of the functions should not be more than 20 lines.
            ● Functions should have no more than three arguments.
            ● There should be no nested loops more than one level deep.
            ● There should be no more than one sentence per line.
            ●Consistent naming conventions for variables, functions and classes
            ●Clear and descriptive comments explaining the purpose and logic of the code
            ● Correct indentation and formatting for better visual organization
            ●Using meaningful and self-documenting variable and function names
            ● Adherence to coding style guides and Python-specific conventions
         The final grade will be calculated as the ratio between the number of criteria the code successfully met and the total of all criteria, multiplied by 100.

      - **Code Efficiency**: will be measured by analyzing the complexity of running time and space of the algorithm theoretically without running the code. The result is expressed using a large O notation, which represents the upper limit of the growth rate of the algorithm.


5. **Ranking and Comparison**
   - Choosing the TOP 2 of all of the 
   - A comparative analysis is conducted to identify the strengths and weaknesses of each model based on the evaluation metrics.

6. **Statistical Testing**
   - Statistical tests are performed to ensure the validity and reliability of the evaluation results.
   - The final rankings and insights are compiled into a comprehensive report.
