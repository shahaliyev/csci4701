# Project

The team project must involve training a neural network. Each team will choose its own topic, data, and modeling approach within the constraints described below. The purpose of the project is not to achieve the highest possible performance, but to explore a clearly stated question related to deep learning through experimentation and analysis. Your grade will be based on how clearly you define your goal, how reasonably you design and carry out your experiments, and how well you interpret and justify your results.

## Timeline

| Component | Deadline | Weight |
|---|---|---:|
| Team Formation | February 3 | - |
| Project Proposal | February 10 | 10% |
| Project Milestone 1 | March 12 | 45% |
| Project Milestone 2 | April 28 | 45% |

## Team Formation

Students must form their own teams of up to three members. It is the responsibility of students to find teammates through class interaction, course communication channels, or personal contacts. Students who do not register a team by the deadline may proceed individually and complete the project alone. No extensions or special accommodations will be provided for team formation.

Registering a team requires emailing the instructor before the deadline with all team member emails in carbon copy (CC) and submitting the names and IDs of all team members. Partial submissions or informal agreements are not considered valid.

## Project Proposal

The project proposal consists only of completing the proposal form[^form] with [Heilmeier catechism](https://www.darpa.mil/about/heilmeier-catechism). No separate document is required. The proposal is evaluated as _pass_, _revise_, or _reject_. All required answers must meet the minimum requirements below for the proposal to pass, and each answer must be between 100 and 200 words.

[^form]: Form link is shared via Blackboard.

For **What are you trying to do?**, the answer must clearly state a specific and concrete goal or question. It must be possible to determine exactly what will be studied, tested, or analyzed. The answer must not consist of broad statements such as "we will explore deep learning methods" or restate the project title without an explicit objective.

For **How is it done today, and what are the limits of current practice?**, the answer must describe at least one commonly used approach relevant to the project and at least one limitation of that approach. The description does not need to be detailed or literature-based, but it must be coherent and directly connected to the stated project goal. Generic or copy-pasted descriptions that could apply to any project do not meet this requirement.

For **What is new in your approach and why do you think it will be successful?**, the answer must explicitly state what will be changed, tested, compared, or examined relative to the usual approach described above. "New" does not mean novel research; it means new within the scope of this project. The answer must make clear what the team plans to do differently, not simply that they will "try a different model" without explanation.

For **If you are successful, what difference will it make?**, the answer must explain what will be learned or demonstrated if the project works as intended. The answer must go beyond claiming improved performance and explain why the expected outcome is meaningful in the context of the stated goal.

For **What are the risks?**, the answer must identify at least one realistic technical, practical, or methodological risk. The answer must explain how this risk could affect the results or how conclusions might change if the risk materializes. Stating that there are no risks or listing only trivial issues does not meet this requirement.

For **How long will it take (in hours)? How much will it cost? Why?**, the answer must include numerical estimates for time and computational cost, even if approximate, together with a short justification. Estimates must be plausible for a one-semester project. Answers without numbers or with clearly unrealistic estimates do not meet this requirement.

For **How exactly will you check your success?**, the answer must specify at least one concrete evaluation method. This may include metrics, comparisons between models or settings, plots, or qualitative inspections. The answer must make clear how the team will decide whether the project goal has been achieved. Vague statements such as "we will analyze the results" are insufficient.

For **Dataset choice**, the answer must clearly state which dataset will be used or how the dataset will be constructed. The dataset must be public and feasible to use within one semester. The answer must justify why this dataset is appropriate for the stated goal and explain why the choice is not trivial. If the dataset is commonly used, the proposal must clearly state what modification, condition, or focused question makes the project non-standard. Anyone should be able to access and explore the dataset via the provided link.

A proposal passes only if all answers meet the minimum requirements above. A proposal is marked revise if one or more answers are unclear, incomplete, or weak but can be corrected without changing the core project idea. Full credit is awarded once the revision is approved. A proposal is rejected if the project goal is unclear, the evaluation method or dataset is missing, or the project is infeasible or reduces to a direct replication of a standard example. A rejected proposal that is resubmitted as a substantially different proposal is eligible for a maximum of 5%.

## Project Milestone 1

The purpose of Milestone 1 is to demonstrate that the project is technically viable and that meaningful progress toward the stated goal has been made.

By Milestone 1, the graded notebook must clearly restate the project goal and show a complete, working training pipeline implemented in PyTorch. This includes loading and preprocessing the chosen dataset, defining the model, training the model, and producing results. This component accounts for 15% of the milestone grade and is evaluated based on technical correctness and whether the code runs end-to-end.

The submission must include at least one completed experiment that directly relates to the stated project goal. The experiment must be clearly described, and the experimental setup must be reasonable for answering the project question. This component accounts for 15% of the milestone grade and is evaluated based on alignment between the experiment and the stated goal.

Concrete results must be presented, such as metrics, plots, or qualitative examples. The evaluation method described in the proposal must now be implemented in practice. This component accounts for 10% of the milestone grade and is evaluated based on whether the results are meaningful and correctly produced.

The repository and the main Jupyter notebook must be organized, readable, and reproducible, with clear explanations of what was done and why. This component accounts for 5% of the milestone grade.

## Project Milestone 2

The purpose of Milestone 2 is to provide a complete and coherent answer to the original project question.

By Milestone 2, the submission must present a complete set of experiments that directly address the stated goal. This includes any comparisons, variations, or analyses needed to support the conclusions. This component accounts for 20% of the milestone grade and is evaluated based on the completeness and relevance of the experiments.

The graded notebook must clearly interpret the results and explain what they show. Conclusions must follow logically from the evidence presented, and claims must be neither exaggerated nor unsupported. This component accounts for 15% of the milestone grade and is evaluated based on the quality of reasoning and interpretation.

The submission must include a clear discussion of limitations and failures, including what did not work and how these issues affect the conclusions. This component accounts for 5% of the milestone grade.

The repository and Jupyter notebook must be well structured, readable, and reproducible, making it possible to follow the full workflow without external explanation. This component accounts for 5% of the milestone grade.

## Submission Requirements

Project Milestones 1 and 2 must be submitted as a single public Git repository hosted on GitHub. The repository must be accessible without authentication and must remain available until final grades are released. All team members are expected to contribute to the repository. Submissions where substantial contributions come from only one team member may be penalized regardless of the stated allocation of responsibilities.

Students must submit to Blackboard both the link to the GitHub repository and a ZIP archive of the same repository as a backup. The ZIP archive must correspond exactly to the submitted repository and will be used only in cases where the public repository becomes unavailable. Discrepancies between the repository and the ZIP archive may result in penalties. The same GitHub repository must be used for both Milestone 1 and Milestone 2. For Milestone 2, students must submit an updated ZIP archive of the same repository reflecting the final state of the project.

The repository must be well organized and must contain modular code implementing the full training and evaluation pipeline in PyTorch. Core functionality such as data loading, model definition, training, and evaluation must be implemented as reusable Python modules rather than as a single monolithic script or notebook.

The repository must include a main Jupyter notebook (`.ipynb`) that will be used for grading. This notebook must run end-to-end without modification on Google Colab and must reproduce the reported results by importing and using the modular code provided in the repository.

The repository must include a `README.md` file that serves as the primary project document. The README must clearly restate the project goal and provide a concise description of the problem being studied and the approach taken. It must include a table allocating responsibilities among team members, listing each member's name together with their specific contributions to the project.

The README must also describe how to install dependencies, how to run the graded notebook, and how to reproduce the reported results on Google Colab. In addition, the README must present the main experimental results and include a clear analysis of the findings, explaining what was observed, what worked, what did not work, and why. Brief descriptions, including superficial AI-generated text, do not meet this requirement.

!!! warning "Important"
    You may use an LLM to assist with brainstorming throughout the project. However, you are fully responsible for understanding, implementing, and defending all aspects of your project. Any explanation, result, or conclusion included in your submissions must be something you can justify. Work that is generic, poorly grounded, or inconsistent with the submitted results will be penalized regardless of how it was generated. Clear structure and concise explanations are expected, but grading focuses on content and correctness.

Deadline policy is noted in the [assessments](02_assessments.md) page.
