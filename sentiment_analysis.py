#!/usr/bin/env python
# coding: utf-8
#
# Sentiment Analysis using GCP Natural Language API
# Author: Jack Munday
#
# Library dependencies: pandas (1.2.3), openpyxl (3.0.7), numpy (1.20.1), google cloud (2.0.0)

import pandas as pd
from google.cloud import language_v1
import os
import json


def pivot_data(input_data, filetype, unique_id, slicer_columns, columns_to_drop):
    """
    Takes an input survey file, filters for free text questions and then pivots from a "wide" to "long" dataframe

    params:
    ------------------------------------------------------------------------------------------------------------
    input_data:         directory survey data to be analysed e.g. google form output
    filetype:           file type of the survey data specified by input_data dir
    unique_id:          name of unique identifier columns, if not present assigns row number to this value
    slicer_columns:     these are the columns which you will want to use as your slicer values in power bi to
                        filter the data
    columns_to_drop:    columns in the data which do not contain free text responses, e.g. numerical

    returns:
    ------------------------------------------------------------------------------------------------------------
    df:        saved into the follow dir "preprocess_data/pivoted_data.csv", containing the following columns:
                    UniqueID, type string: Unique, anonymised identifier of respondent
                    Question, type string: Question being answered or unique identifier to this question
                    Response, type string: Individual's response to the question
    """
    if filetype == "xlsx":
        df = pd.read_excel(input_data)
    elif filetype == "csv":
        df = pd.read_csv(input_data)
    else:
        print("Unsupported input file entered")
        return 0

    if columns_to_drop:
        df.drop(columns_to_drop, axis=1, inplace=True)

    # generate a unique row id if dataset does not have a unique specifier already
    if unique_id == "None":
        df["unique id"] = [i for i in range(0, df.shape[0])]
        df = df[["unique id"] + [col for col in df.columns if col != "unique id"]]
    else:
        df.rename(columns={unique_id: "unique id"}, inplace=True)

    key_cols = ["unique id", *slicer_columns]

    # subtracts key_cols from all cols in df to get the columns which need to be pivoted
    pivot_cols = [x for x in list(df) if x not in key_cols]

    df = df.melt(id_vars=key_cols, value_vars=pivot_cols, var_name='Question', value_name='Response')
    df.dropna(subset=['Response'], inplace=True)
    # limit dataset to one row for test to limit calls to gcp
    df = df.head(1)
    df.to_csv('output_data/pivoted_data.csv', index=False)
    return df


def sentiment_analysis_response(text_content):
    """
    Analyses the global sentiment of a single input string, returning it's sentiment and sentiment magnitude.

    This makes a call to Google's Natural Language API (https://cloud.google.com/natural-language).
    It will require setting up a GCP project to process this request, ensuring that
    proper authentication has been granted to the local machine running this script through setting up a service account
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
    client = language_v1.LanguageServiceClient()
    type_ = language_v1.Document.Type.PLAIN_TEXT
    language = "en"
    document = {"content": text_content, "type_": type_, "language": language}
    encoding_type = language_v1.EncodingType.UTF8

    response = client.analyze_sentiment(request={'document': document, 'encoding_type': encoding_type})
    return [response.document_sentiment.score, response.document_sentiment.magnitude]


def sentiment_analysis_entity(text_content):
    """
    Detects all proper nouns in the input "text_content" and outputs the type, sentiment, salience and magnitude in
    relation to wider context for each proper noun (named entity).

    This makes a call to Google's Natural Language API (https://cloud.google.com/natural-language).
    It will require setting up a GCP project to process this request, ensuring that
    proper authentication has been granted to the local machine running this script through setting up a service account
    and generating the required keys inside of the project through which this request is being processed.

    params:
    ------------------------------------------------------------------------------------------------------------
    text_content: type str - response to analyse individual entity sentiment

    returns:
    ------------------------------------------------------------------------------------------------------------
    entities:           type str - list of named entities (proper nouns) detected in response
    entity_type:        type str - list of classifications of entity type (e.g. Person, Organisation etc)
    entity_salience:    type float - list of saliencies (importance of noun in sentence) of named entities
    entity_sentiment:   type float - list of the sentiment of named entities in entities in the range of [-1,1] as in
                                    sentiment_analysis_response
    entity_magnitude:   type float, list of the magnitude of sentiment of named entities in entities as in
                                    sentiment_analysis_response
    """
    client = language_v1.LanguageServiceClient()
    type_ = language_v1.Document.Type.PLAIN_TEXT
    language = "en"
    document = {"content": text_content, "type_": type_, "language": language}
    encoding_type = language_v1.EncodingType.UTF8

    response = client.analyze_entity_sentiment(request={'document': document, 'encoding_type': encoding_type})

    entities, entity_type, entity_salience, entity_sentiment, entity_magnitude = [], [], [], [], []

    for entity in response.entities:
        # loop over all named entities (proper nouns) that have been detected in the input string and append the
        # properties as classified by the google API to their respective lists.
        entities.append(entity.name)
        entity_type.append(language_v1.Entity.Type(entity.type_).name)
        entity_salience.append(entity.salience)
        entity_sentiment.append(entity.sentiment.score)
        entity_magnitude.append(entity.sentiment.magnitude)
    return [entities, entity_type, entity_salience, entity_sentiment, entity_magnitude]


def sentiment_pipeline(df, overall, entity, slicer_columns, num_sentiment_cats):
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
    ######################################################################
    # PREPROCESSING OF INPUT DATA
    ######################################################################
    # Partitions to map sentiment.score against a categorical value
    if num_sentiment_cats == 5:
        cut_bins = [-1, -0.6, -0.2, 0.2, 0.6, 1]
        cut_labels = ["Very Negative", "Negative", "Neutral", "Positive", "Very Positive"]
    elif num_sentiment_cats == 3:
        cut_bins = [-1, -0.25, 0.25, 1]
        cut_labels = ["Negative", "Neutral", "Positive"]
    else:
        print("Please enter either 3 or 5 in 'num_sentiment_cats in input_params.json")
        return

    ######################################################################
    # SENTIMENT ANALYSIS OF EACH RESPONSE
    ######################################################################
    if overall:
        response_df = pd.DataFrame()
        for response in df['Response']:
            # get row number of current response in df being processed
            indx = df[df['Response'] == response].index[0]
            current_id = df.loc[indx, 'unique id']
            current_question = df.loc[indx, 'Question']

            if slicer_columns:
                # slicer_columns and slicer_column_values dynamically account for the fact that the user may
                # want to include extra columns that are not text responses to allow for filtering in visuals

                slicer_column_values = [df.loc[indx, str(column)] for column in slicer_columns]
                response_sentiment, response_magnitude = sentiment_analysis_response(response)

                response_df_columns = ['unique id', *slicer_columns, 'Question', 'Response', 'Sentiment', 'Magnitude']
                response_df_values = [current_id, *slicer_column_values, current_question, response, response_sentiment,
                                      response_magnitude]

                response_df_loop = pd.DataFrame([response_df_values], columns=response_df_columns, index=[0])
            else:
                response_sentiment, response_magnitude = sentiment_analysis_response(response)
                response_df_loop = pd.DataFrame({'unique id': current_id,
                                                 'Question': current_question,
                                                 'Response': response,
                                                 'Sentiment': response_sentiment,
                                                 'Magnitude': response_magnitude
                                                 }, index=[0])

            response_df = response_df.append(response_df_loop, ignore_index=True)
        response_df['Sentiment Buckets'] = pd.cut(response_df['Sentiment'], bins=cut_bins, labels=cut_labels)

        response_df.to_csv('output_data/sentiment_by_response.csv', index=False)
        print(response_df.head())

    ######################################################################
    # SENTIMENT ANALYSIS OF NAMED ENTITIES (PROPER NOUNS)
    ######################################################################
    if entity:
        entity_df = pd.DataFrame()
        for response in df['Response']:
            # get row number of current response in df being processed, which allows referenced to "uID" and "Question"
            # via indx so that when the outputs from API call are appended into "entity_df" we have kept track of its
            # inputs.
            indx = df[df['Response'] == response].index[0]
            current_id = df.loc[indx, 'unique id']
            current_question = df.loc[indx, 'Question']
            ##################################################################
            # MODIFY HERE IF YOU HAVE ADDED EXTRA VALUES TO KEY_COLS
            # Suppose you had added 'Grade' to key_cols: key_cols = ['Survey Instance ID', 'Grade']
            # uncomment the below and edit entity_df_loop accordingly
            # current_grade = df.loc[indx, 'Question']
            #################################################################
            # TODO MODIFY ENTITY SENTIMENT TO ACCOUNT FOR VARYING LENGTH OF SLICER_COLUMNS
            entity_list, type_list, salience_list, sentiment_list, magnitude_list = sentiment_analysis_entity(response)

            # loop over all entities detected in response, as sentiment_analysis_entity returns a series of list with
            # one index per entity in response
            for i in range(len(entity_list)):
                #############################################################################
                # APPEND ALL VALUES FROM 'key_cols' AND THEIR RESPECTIVE SENTIMENT METRICS TO DF
                # Suppose had added Grade to "key_cols": key_cols = ['Survey Instance ID', 'Grade']
                # Then you would need to insert the following line into entity_df_loop:
                # 'Grade': current_grade,
                #############################################################################
                entity_df_loop = pd.DataFrame({'unique id': current_id,
                                               'Question': current_question,
                                               'Response': response,
                                               'Entity': entity_list[i],
                                               'Type': type_list[i],
                                               'Salience': salience_list[i],
                                               'Sentiment': sentiment_list[i],
                                               'Magnitude': magnitude_list[i]}, index=[0])
                entity_df = entity_df.append(entity_df_loop, ignore_index=True)
        entity_df['Sentiment Buckets'] = pd.cut(entity_df['Sentiment'], bins=cut_bins, labels=cut_labels)

        entity_df.to_csv('output_data/sentiment_by_entity.csv', index=False)
    return


def main():

    #############################################################################
    # LOAD INPUT PARAMS CONFIG FILE
    #############################################################################
    with open('input_data/input_params.json') as f:
        params = json.load(f)

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = params['api_key_directory']

    filetype = params['file_directory'].partition(".")[2]

    # TODO Fix if possible, as slightly forcing it here using (0/1) - couldn't read in the booleans from json properly
    overall, entity = bool(int(params['overall'])), bool(int(params['entity']))

    
    #############################################################################
    # RUN SENTIMENT PIPELINE
    #############################################################################
    pivoted_df = pivot_data(params['file_directory'], filetype=filetype, unique_id=params['uID_column'],
                            slicer_columns=params['slicer_columns'], columns_to_drop=params['columns_to_drop'])

    sentiment_pipeline(pivoted_df, overall=overall, entity=entity, slicer_columns=params['slicer_columns'],
                       num_sentiment_cats=params['num_sentiment_cats'])
    return


# TODO ADJUST COMMENTS FOR WHOLE SCRIPT ACCORDINGLY
if __name__ == "__main__":
    main()
