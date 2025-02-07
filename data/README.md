# Data Handling

## raw_data

This contains your raw data (which you might want to add to your `.gitignore`) so that it doesn't show up on github. Github has a file size limit of 100 Mb, which isn't that much.

**Never** overwrite or modify the data here.

## processed_data

This contains various checkpoints of the changes you've made. For example, you've tried taking the `log` of your data, or you have `squared` it. Store those changes here and perhaps add a `README.md` to keep track of the transformations and why you did them.

You don't need to store **every** change individually - use your best judgment.

## modeling_data

This folder contains any changes you might make to your data right before modeling. How does this differ from `processed_data`? processed_data might contain general purpose modifications for all models you might try, but this folder might contain model-specific changes.

E.g. you normalized your data for your linear regression, but you don't necessarily want to normalize the data for **all** your models. You might want to save the normalized data here as a "cache", especially if normalizing takes a long time. This will speed up turnaround time significantly.
