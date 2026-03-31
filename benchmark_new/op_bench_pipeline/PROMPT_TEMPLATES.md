# Prompt Templates

## Modified OP-Bench Judge Prompt

This template is used for the adjusted multi-perspective evaluation setting.

```text
You are given a benchmark question, a user's free-response opinion, and an LLM-generated answer.

Question: {question}

User free response: {freeresponse}

LLM-generated response: {llm_response}

Your task is to judge how well the response represents the user's view. For multi-perspective answers, do not automatically penalize the response simply because it includes several perspectives. Instead, determine whether the response contains at least one clear and meaningful perspective that matches the user's expressed opinion.

Use the following rubric:
1 = no meaningful match, or the response is mostly inconsistent with the user's view.
2 = slight or vague overlap, but the match is weak.
3 = partial match, but the relevant perspective is diluted, buried, or not central to the response.
4 = a clear matching perspective is present and sufficiently prominent.
5 = the response contains a strong, explicit, and central perspective that clearly represents the user's view.

Output only a single integer from 1 to 5.
```

## Original OP-Bench Judge Summary

The original OP-Bench setup instead asks whether the whole response would feel representative to a single user. This distinction matters for models that intentionally generate broader multi-perspective answers.
