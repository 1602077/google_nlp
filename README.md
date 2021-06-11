# google_nlp

```sentiment_analysis.py``` is a python script that uses GCP's natural language API to assign a sentiment score from one of the following buckets [Very Positive, Positive, Neutral, Negative, Very Negative] to passages of free text. The pipeline set up in this script is designed to process the output of an excel download of a google survey. However the functions here can be easily adjusted to perform sentiment analysis on any form of text.

## Setting up environment

- You will require a GCP project to proccess the request - full documentation on setting up a GCP project can be found [here](https://cloud.google.com/resource-manager/docs/creating-managing-projects).
- Once your GCP project is set up, you should ensure that your local machine has the proper authentications to call the API - an explaination of set this up is given [here](https://cloud.google.com/docs/authentication/production).
- Finally you will need to installed the scripts dependecies [Pandas (1.2.3); OpenPyXL (3.0.7); Google Cloud (2.0.0)] to ensure it can execute. This can be installed as follows: 
``` pip install --upgrade google-cloud-language numpy pandas openpyxl```.

If you do not have ```pip``` installed run the following:
```python3 -m pip install --upgrade pip```.

Alternatively, you may choose to use the ```conda``` package manager:
```
conda create --name gcp_env
conda install -n gcp_env google-cloud-language numpy pandas openpyxl
conda activate gcp_env
```

## Script functionality overview
```pivot_data(input_data)```

```sentiment_analysis_entity(text_content)```

```sentiment_analysis_response(text_content)```

```sentiment_pipeline(df, overall, entity)```

## Running script

## Modifying to suit your dataset
