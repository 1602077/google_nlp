# google_nlp

```sentiment_analysis.py``` is a python script that uses GCP's natural language API to assign a sentiment score from one of the following buckets [Very Positive, Positive, Neutral, Negative, Very Negative] to passages of free text. The pipeline set up in this script is designed to process the output of an excel download of a google survey. However the functions here can be easily adjusted to perform sentiment analysis on any form of text.

## Setting up environment

- You will require a GCP project to proccess the request - full documentation on setting up a GCP project can be found [here](https://cloud.google.com/resource-manager/docs/creating-managing-projects).
- Once your GCP project is set up, you should ensure that your local machine has the proper authentications to call the API - an explaination of set this up procedure is given [here](https://cloud.google.com/docs/authentication/production).
- Finally you will need to install the required libraries for the script [Pandas (1.2.3); OpenPyXL (3.0.7); Google Cloud (2.0.0)] to ensure it can execute. This can be installed as follows: 
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

There are four main functions which are called across the data processing pipeline. ```pivot_data()``` transforms the input file and saves the output to ```../preprocess_data```. 

This is then passed as a dataframe through ```sentiment_pipeline(df, overall, entity)```, which can optionally call either ```sentiment_analysis_response()``` or ```sentiment_analysis_entity()``` by specifying either ```overall``` or ```entity``` as ```True```. 

Setting the boleans for these two values is particularly important on large datasets due to the compute time and cost associated with calling the API. Calculating sentiment overall for a given response and calculating the sentiment of individual entities contained within that response requires two seperate calls to the Natural Language API (a service which is not free if you are making > 5k calls / month). Any operation which requires a large number of calls will not be instanteous, in my experience the API can typically process in excess of around 6,000 responses per hour. As this script was for a series of ad-hoc requests, I did not look into optimising my code (to minimise the number of calls) and simply chose to leave it to run overnight to process my data. 

Below I've included the doc strings and some potential modifications for each function may be useful depending on your dataset and/or use case.

### ```pivot_data(input_data)```
```
    """
    Takes an input excel file, filters for free text questions and then pivots from a "wide" to "long" dataframe
    params:
    ------------------------------------------------------------------------------------------------------------
    input_data: xlsx survey data to be analysed e.g. google form output
    returns:
    ------------------------------------------------------------------------------------------------------------
    df:        saved into the follow dir "preprocess_data/pivoted_data.csv", containing the following columns:
                    UniqueID, type string: Unique, anonymised identifier of respondent
                    Question, type string: Question being answered or unique identifier to this question
                    Response, type string: Individual's response to the question
    """
```

```pivot_data()``` is designed to process a excel file, however the pandas library can easily accomdate other file formats by modifying line 35.

<center>

  | File Format     | Input Function       |
  |-----------------|----------------------|
  | csv             | ```pd.read_csv()```  |
  | sql database    | ```pd.read_sql()```  |
  | Google Bigquery | ```pd.read_gbq()```  |
  | json            | ```pd.read_json()``` |
  
</center>

I've also stripped out any redudant columns that are not related to calculating sentiment from my dataset to prevent any subsequent models from getting too big. This is performed by editing lines 36 and 37.

Consider you wanted to remove the following two columns from your dataset (email adress, time of completion):
```
cols_to_drop = [‘email address’, ‘time of completion’]
df.drop(cols_to_drop, axis=1, inplace=True)
```
The ```pivot_data()``` function is currently set up to process data from the following schema:
| Unique Identifier | Question 1 | Question 2 | Question 3 |
|-------------------|------------|------------|------------|
| user_001          | Q1 answer  | Q2 answer  | Q3 answer  |
| user_002          | ...        | ...        | ...        |

to this:

| Unique Identifier | Question   | Answer    |
|-------------------|------------|-----------|
| user_001          | Question 1 | Q1 answer |
| user_001          | Question 2 | Q2 answer |
| user_001          | Question 3 | Q3 answer |
| ...               | ...        | ...       |

All columns not inside the list ```key_cols``` will be unpivoted, with their column headers being placed in the ‘Question’ column and their values into the ‘Answer’ column  of the unpivoted dataframe. If you have separate columns for ‘Email’ or ‘Name’ you may want to include these in the key_cols list so that your data can be easily filtered by these fields when being visualised.

### ```sentiment_analysis_entity(text_content)```
```
    """
    Analyses the global sentiment of a single input string, returning it's sentiment and sentiment magnitude.
    This makes a call to Google's Natural Language API (https://cloud.google.com/natural-language).
    It will require setting up a GCP project to process this request and  ensuring that proper authentication has 
    been granted to the local machine running this script through setting up a service account
    and generating the required keys inside of the project through which this request is being processed.
    params:
    ------------------------------------------------------------------------------------------------------------
    text_content: type str - response of which sentiment is being analysed
    returns:
    ------------------------------------------------------------------------------------------------------------
    List containing the following elements:
    document_sentiment.score: type float - Specifies sentiment of "text_content" in the range [-1, 1],
                                                    where -1 = very negative and 1 = very positive
    document_magnitude.score: type float - Strength of emotion being conveyed in the range of [0, unbounded],
                                                   where the higher the number the more strongly worded.
    """
```

E.g. By inputting the sentence “My favourite food is ice cream!”, this function will return a sentiment score of 0.9 and a magnitude of 0.9.

```
>>>   sentiment_analysis_response(“My favourite food is ice cream!”)
>>>   [0.9, 0.9]
```


### ```sentiment_analysis_response(text_content)```
```
    """
    Detects all proper nouns in the input "text_content" and outputs the type, sentiment, salience and magnitude in
    relation to wider context for each proper noun (named entity).
    This makes a call to Google's Natural Language API (https://cloud.google.com/natural-language).
    It will require setting up a GCP project to process this request and  ensuring that proper authentication has 
    been granted to the local machine running this script through setting up a service account
    and generating the required keys inside of the project through which this request is being processed.
    params:
    ------------------------------------------------------------------------------------------------------------
    text_content: type str - response to analyse individual entity sentiment
    returns:
    ------------------------------------------------------------------------------------------------------------
    entities:           type str - list of named entities (proper nouns) detected in response
    entity_type:        type str - list of classifications of entity type (e.g. Person, Organisation etc)
    entity_salience:    type float - list of saliences (importance of noun in sentence) of named entities
    entity_sentiment:   type float - list of the sentiment of named entities in entitiesin the range of [-1,1] as in
                                    sentiment_analysis_response
    entity_magnitude:   type float, list of the magnitude of sentiment of named entities in entities as in
                                    sentiment_analysis_response
    """
```
As before to bring this to life slightly suppose we inputted the following “I love ice cream, but I hate cauliflower!”.

```
>>>   sentiment_analysis_entity(“I love ice cream, but I hate cauliflower!”)
>>>   [[ice cream, cauliflower], [consumer good, consumer good], [0.93, 0.07], [0.3, -0.2], [0.6,0.2]]
```
Reconstructing this output as a table:
| Name         | Type          | Salience | Sentiment | Magnitude |
|--------------|---------------|----------|-----------|-----------|
| Ice cream    | Consumer Good | 0.93     | 0.3       | 0.6       |
| Cauliflower  | Consumer Good | 0.07     | -0.2      | 0.2       |

N.B Despite the fact that this sentence has three nouns in google’s API recognises that ice cream and food are the same entity in the context of this sentence.

### ```sentiment_pipeline(df, overall, entity)```
```
    """
    Pipeline:
    1. Preprocessing dataset pivoting into a "long" format through calling pivot_data().
    2. Calculates the sentiment and magnitude at the response level and outputs this as "sentiment_by_response.csv"
    3. Detects all proper nouns in the response and then assigns a type, sentiment, salience and magnitude per noun
    Steps 2. and 3. are both processing the input_data as outputted by pivot_data() through calls to Google's Natural
    Language API, the only difference between the two is the granularity of the sentiment being calculated.
    params:
    ------------------------------------------------------------------------------------------------------------
    df: input dataframe that has been preprocess by pivot_data and then split into chunks to account for
             any errors or internet timeout when processing whole file
    overall: type bool - if True calculates the overall sentiment for each response by running sentiment_by_response().
                         i.e. runs step 2. in pipeline.
    entity: type bool - if True calculates sentiment of each entity in each response by running sentiment_by_entity().
                        i.e. runs step 3. in pipeline.
    returns:
    ------------------------------------------------------------------------------------------------------------
    2 csv files both saved in the output_data dir:
        sentiment_by_response.csv: Response sentiment analysed at the global level
        sentiment_by_entity.csv:   Response sentiment analysed for each proper noun in the responses
    """
```
One particularly important operation is the sentiment bucketing performed in this pipeline, I currently have partioned the continous range of sentiment scores into 5 bins.
```
cut_bins = [-1, -0.6, -0.2, 0.2, 0.6, 1]
cut_labels = ["Very Negative", "Negative", "Neutral", "Positive", "Very Positive"]
response_df['Sentiment Buckets'] = pd.cut(response_df['Sentiment'], bins=cut_bins, labels=cut_labels)
```
For some applications this may not be appropriate to split your dataset so finely, especially if you were planning to analyse social media posts where nlp has a tedency to missclasify due to the lack of context provided in such short posts. The following 3 bin classification is reccomended in google's documentation.
```
cut_bins = [-1, -0.25, 0.25, 1]
cut_labels = [“Negative”, “Neutral”, “Positive”]
```

## Running script

Finally, you can run the script as follows:
```
export GOOGLE_APPLICATION_CREDENTIALS="~/filepath_to_project/project_name/api_keyname.json"
sentiment_pipeline(pivot_data("input_data/input_filename.xlsx"), overall=True, entity=True)

```
