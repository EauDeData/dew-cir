## This receives an image
## And a vector DB with train data
## And train labels
## And returns a graph


"""


Image:[IMAGE-EMBEDDING | CLASS_EMBEDDING] --{has crop}--> CROP[DATE-EMBEDDING (predicted) | CLASS_EMBEDDING]
CROP[DATE-EMBEDDING | CLASS_EMBEDDING] ---{from year}--> YEAR[DATE-EMBEDDING | CLASS_EMBEDDING]
CROP[DATE-EMBEDDING | CLASS_EMBEDDING] --{KNN}--> CROP[DATE-EMBEDDING | CLASS_EMBEDDING]

"""