import sys
from google.cloud import language_v1

def sentiment_analysis(text_content=None):
    """
    Analyses the global sentiment of an input string, outputting it's sentiment (in a range of [-1, 1]
    where a 1 is highly positive and -1 highly negative) and magnitude (in a range of [0, ∞], conveying
    the weighing of each sentiment score. These scores are calculated globally, per sentence and per entity.
    Each sentiment is calculated using Google's Natural Language API.
    Args:
      text_content The text content to analyze
    """

    client = language_v1.LanguageServiceClient()
    type_ = language_v1.Document.Type.PLAIN_TEXT
    language = "en"
    document = {"content": text_content, "type_": type_, "language": language}
    encoding_type = language_v1.EncodingType.UTF8

    sentence_overall = client.analyze_sentiment(request = {'document': document, 'encoding_type': encoding_type})
    sentence_entites = client.analyze_entity_sentiment(request = {'document': document, 'encoding_type': encoding_type})

    print('#########################################################################\n')
    print(u"Document sentiment score: {}".format(sentence_overall.document_sentiment.score))
    print(u"Document sentiment magnitude: {}\n".format(sentence_overall.document_sentiment.magnitude))
    print('#########################################################################\n')

    for sentence in sentence_overall.sentences:
        print(u"Sentence text: {}".format(sentence.text.content))
        print(u"Sentence sentiment score: {}".format(sentence.sentiment.score))
        print(u"Sentence sentiment magnitude: {}\n".format(sentence.sentiment.magnitude))
    print('#########################################################################\n')
            
    for entity in sentence_entites.entities:
        print(u"Representative name for the entity: {}".format(entity.name))
        print(u"Entity type: {}".format(language_v1.Entity.Type(entity.type_).name))
        # Get the salience score associated with the entity in the [0, 1.0] range
        print(u"Salience score: {}".format(entity.salience))
        sentiment = entity.sentiment
        print(u"Entity sentiment score: {}".format(sentiment.score))
        print(u"Entity sentiment magnitude: {}\n".format(sentiment.magnitude))
    return 1

if __name__ == "__main__":
    sentiment_analysis(*sys.argv[1:])
 
