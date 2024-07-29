# Evaluation-of-the-code-quality-generated-by-GENERATION-AI-Dataset

This is the repository of the all of the data used in the paper "Evaluation of the code quality generated by GENERATION AI", From the **Promts** for the generative AI to there **Answers** and the **Evaluation Results**

# Instructions for recreating the experiment
Process Overview
----------------

1. **Data Collection**
   - A dataset of problems is collected from LeetCode.(you can use the problems we chose [here](./path/to/your/file.py))
   - Two subsets of problems are created: an experiment problems dataset and a benchmark problems dataset.

2. **Code Generation**
   - Multiple AI models (e.g., OpenAI Codex, DeepMind AlphaCode, Microsoft GitHub Copilot, and others) are used to generate code solutions for the problems in both datasets.

3. **Generated Code Compilation**
   - The generated code from each model is compiled into a unified format for further analysis.

4. **Code Quality Evaluation**
   - The generated code is evaluated on three primary metrics:
     - **Code Correctness**: Verifies if the generated code provides the correct output for given inputs.
     - **Code Readability**: Assesses the clarity and comprehensibility of the code.
     - **Code Complexity**: Measures the complexity of the code, including factors like cyclomatic complexity and maintainability.

5. **Ranking and Comparison**
   - The evaluated metrics are used to rank the code generated by each AI model.
   - A comparative analysis is conducted to identify the strengths and weaknesses of each model based on the evaluation metrics.

6. **Statistical Testing**
   - Statistical tests are performed to ensure the validity and reliability of the evaluation results.
   - The final rankings and insights are compiled into a comprehensive report.
