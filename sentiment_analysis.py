#!/usr/bin/env python
# coding: utf-8
#
# Sentiment Analysis using GCP Natural Language API
# Author: Jack Munday
#
# This script processes the output of a google survey excel file, taking it from a "wide" format with a col per question
# to "long" format with a col for question id and another col for response. The free text responses are then given a
# sentiment score through making calls to Google's Natural Language API. This script can then either be run inside the
# google console or on your local machine provided you have granted the proper service accounts / keys to the project.
#
# Library dependencies: pandas (1.2.3), openpyxl (3.0.7), numpy (1.20.1), google cloud (2.0.0)
#

import pandas as pd
from google.cloud import language_v1


def pivot_data(input_data):
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

    df = pd.read_excel(input_data)
    #cols_to_drop = ['']
    #df.drop(cols_to_drop, axis=1, inplace=True)

    # if your dataset doesn't have a unique user ID, then please uncomment the two lines below to generate an ID
    # (this uses row number to keep track of unique response). In the case of a google form survey output the excel file
    # has one line per respondent so using row numbers works fine. If you have multiple rows in your dataset per
    # response this will not be an appropriate uID. This may be useful if you have the emails of respondents
    # and would like to anonymise the dataset yet still keep track of the responses of a given individual

    # df['uID'] = [i for i in range(df.shape[0])]
    # df = df[['uID'] + [col for col in df.columns if col != 'uID']]

    # columns that are not to be pivoted in the df, these are cols that you would like to keep track of for all rows in
    # the dataset. You may for example have have a col for "Team" or "Grade" in here, these would be good to include in
    # "key_cols" so that the data is filterable by these fields when being visualised.
    key_cols = ['uID']
    # subtracts key_cols from all cols in df to get the columns which need to be pivoted
    pivot_cols = [x for x in list(df) if x not in key_cols]

    df = df.melt(id_vars=key_cols, value_vars=pivot_cols, var_name='Question', value_name='Response')
    df.dropna(subset=['Response'], inplace=True)
    df = df.head(10)
    df.to_csv('preprocess_data/pivoted_data.csv', index=False)
    return df


def sentiment_analysis_response(text_content):
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


def sentiment_pipeline(df, overall, entity):
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
    cut_bins = [-1, -0.6, -0.2, 0.2, 0.6, 1]
    cut_labels = ["Very Negative", "Negative", "Neutral", "Positive", "Very Positive"]

    ######################################################################
    # SENTIMENT ANALYSIS OF EACH RESPONSE
    ######################################################################
    if overall:
        response_df = pd.DataFrame()
        for response in df['Response']:
            # get row number of current response in df being proccessed
            indx = df[df['Response'] == response].index[0]
            current_id = df.loc[indx, 'uID']
            current_question = df.loc[indx, 'Question']
            response_sentiment, response_magnitude = sentiment_analysis_response(response)
            response_df_loop = pd.DataFrame({'uID': current_id,
                                   'Question': current_question,
                                   'Response': response,
                                   'Sentiment': response_sentiment,
                                   'Magnitude': response_magnitude
            }, index=[0])
            response_df = response_df.append(response_df_loop, ignore_index=True)
        response_df['Sentiment Buckets'] = pd.cut(response_df['Sentiment'], bins=cut_bins, labels=cut_labels)

        response_df.to_csv('output_data/sentiment_by_response_TEST.csv', index=False)

    ######################################################################
    # SENTIMENT ANALYSIS OF NAMED ENTITIES (PROPER NOUNS)
    ######################################################################
    if entity:
        entity_df = pd.DataFrame()
        for response in df['Response']:
            # get row number of current response in df being proccessed, which allows referenced to "uID" and "Question"
            # via indx so that when the outputs from API call are appended into "entity_df" we have kept track of its
            # inputs.
            indx = df[df['Response'] == response].index[0]
            current_id = df.loc[indx, 'uID']
            current_question = df.loc[indx, 'Question']
            ##################################################################
            # MODIFY HERE IF YOU HAVE ADDED EXTRA VALUES TO KEY_COLS
            # Suppose you had added 'Grade' to key_cols: key_cols = ['uID', 'Grade']
            # uncomment the below and edit entity_df_loop accordinly
            # current_grade = df.loc[indx, 'Question']
            #################################################################

            entity_list, type_list, salience_list, sentiment_list, magnitude_list = sentiment_analysis_entity(response)

            # loop over all entities detected in response, as sentiment_analysis_entity returns a series of list with
            # one index per entity in response
            for i in range(len(entity_list)):
                #############################################################################
                # APPEND ALL VALUES FROM 'key_cols' AND THEIR RESPECTIVE SENTIMENT METIRCS TO DF
                # Suppose had added Grade to "key_cols": key_cols = ['uID', 'Grade']
                # Then you would need to uncomment the following line into entity_df_loop:
                # 'Grade': current_grade,
                #############################################################################
                entity_df_loop = pd.DataFrame({'uID': current_id,
                                               'Question': current_question,
                                               'Response': response,
                                               'Entity': entity_list[i],
                                               'Type': type_list[i],
                                               'Salience': salience_list[i],
                                               'Sentiment': sentiment_list[i],
                                               'Magnitude': magnitude_list[i]
                }, index=[0])
                entity_df = entity_df.append(entity_df_loop, ignore_index=True)
        entity_df['Sentiment Buckets'] = pd.cut(entity_df['Sentiment'], bins=cut_bins, labels=cut_labels)

        entity_df.to_csv('output_data/sentiment_by_entity.csv', index=False)
    return


if __name__ == "__main__":
    sentiment_pipeline(pivot_data("input_data/input_filename.xlsx"), overall=True, entity=True)
