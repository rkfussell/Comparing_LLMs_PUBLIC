def getPrompts():
    prompt={"C":"<<SYS>>\nAnalyze the sentence given. Does the sentence demonstrate: Iteration (Proposed) - Students should be able to suggest additional rounds of experimentation and choose appropriate improvements. Could be based on experimental evidence. Answer with just yes or no \n<</SYS>>\n [INST]\n ",

        "D":'''<<SYS>>\n Analyze the sentence given. Does the sentence demonstrate: Iteration (Proposed) - Students should be able to suggest additional rounds of experimentation and choose appropriate improvements. Could be based on experimental evidence. 

Inclusion rules: 
1. Must be a sentence in future or present tense that proposes an experimental choice (“ we could…”, “ we will”, “we are doing …”, not something that was done in the past (e.g. “this method improved our uncertainty”)
AND
2. Must have at least one word/phrase you can point to is synonymous with either “more measurements” (additional angles, new trials) or “change/improve” (new/different object, new/different method, revised/improved/adjusted/modified experiment, to reduce or account for uncertainty) or “plan for next time/ plan for future”

Clarifying note: the following are excluded from rule b
1. Methodological decisions during planning or execution. For example, “We are using python/ We will calculate t’” (does not meet rule b)
2. References to “Part 2” or “session 2” or “Ways to eliminate confirmation bias” that are not synonymous with “ways to improve” (does not meet rule b) \n<</SYS>>\n [INST]\n ''',
        "E":"<<SYS>>\nAnalyze the sentence given. Does the sentence demonstrate:Comparison of Quantities - Students should apply data analysis tools quantitatively to make some sort of comparison (between data, best fit lines, predictions, etc.). Answer with just yes or no \n<</SYS>>\n [INST]\n ",
        "G":"",
        "F":"<<SYS>>\nAnalyze the sentence given. Does the sentence demonstrate: Comparison of Quantities - Students should apply data analysis tools quantitatively to make some sort of comparison (between data, best fit lines, predictions, etc.). for example:  Our first dataset is statistically different from our second dataset. Answer with just yes or no. \n<</SYS>>\n [INST]\n", 
        "H":'''<<SYS>>\nAnalyze the sentence given. Does the sentence demonstrate:Comparison of Quantities - Students should apply data analysis tools quantitatively to make some sort of comparison (between data, best fit lines, predictions, etc.). 

Inclusion rules: 

1. Must have evidence of comparison (x is smaller than/greater than/the same as y)
and
2. Must have evidence that x and y are quantities used for data analysis (mean, standard deviation, standard uncertainty, citing specific measurements). Non-data analysis quantities (just saying words like measurements, experiments, efficiency, consistency, precision, accuracy, anything vague) are not sufficient evidence on their own (though check if they reference quantities used for data analysis, such as one line that says “standard uncertainty 1 is less than standard uncertainty 2” followed by “experiment 2 is more precise.”

Types of sentences and clauses that inherently meet both inclusion rules and always count:
1. Stating the outcome (i.e., value) of statistical comparison tests, such as:
    1. Quantitative comparisons between individual measurements or values (including t’ comparisons, percent error, effect size)
    2. Quantitative comparisons between data and best fit lines/curves (including Chi-squared or R^2)
2. Describing the outcome of one of these statistical comparison tests qualitatively, such as that the value for their result is high or low or means that there is a good fit/bad fit.
3. Comparing numerical data to the quantitative predictions of a model, such as:
    1. Values predicted from model is/is not within uncertainties/errors
    2. Describing how well the data fit to a best fit line or numerical predictions of a model (e.g., many points are far from the model line, the line is straight but the data seem to be curved)
    3. Describing a residuals plot (randomly scattered, patterned, large, small) as residuals inherently involve a comparison to a model

Clarifying note: the following are excluded from rule a
1. Pointing to the data without a comparison (e.g., we fit a line to the data or the data are in Figure 2) 
2. describing the data without a comparison (e.g., the data are linear) to not meet this rule
 Answer with just yes or no. \n<</SYS>>\n [INST]\n ''',
        "A":"",
        "J":"<<SYS>>\nAnalyze the sentence given. Does the sentence demonstrate a quantitative comparison between data, best fit lines etc? \n<</SYS>>\n [INST]\n ",

        "B": "<<SYS>>\nAnalyze the sentence given. Does the sentence demonstrate a suggestion of additional rounds of experimentation and/or improvements? For example, 'Next time, we will improve by taking more trials'. Answer with just yes or no. \n<</SYS>>\n [INST]\n "}
    return prompt