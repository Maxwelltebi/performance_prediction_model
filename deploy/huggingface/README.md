---
library_name: sklearn
tags:
  - tabular-classification
  - endpoints-template
---

# Student Performance Random Forest

Portfolio learning model predicting grades a-f from 13 inputs. Includes a custom
Hugging Face EndpointHandler; uploading this repository does not create an API.
POST a JSON object with an `inputs` object containing the StudentInput fields.
The response contains `grade`, `label`, `confidence`, and `breakdown`.

Both artifacts and the shared preprocessing code must be deployed at the same
repository commit. Artifacts were serialized with scikit-learn 1.6.1. The complete
original training environment is unknown; the supplied requirements need validation.

Known limitation: the training notebook refitted its scaler on the test split.
Correct training, reevaluate, and replace both artifacts before treating predictions
or reported accuracy as validated. Confidence is an uncalibrated model probability.
This release is for learning deployment, not validated educational decisions.
