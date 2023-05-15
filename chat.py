import pandas as pd

from preprocessing import TfChat
EXIT_KEYWORDS = [
    "quit", "exit", "goodbye", "bye", "stop", "end", "cancel", "close",
    "finish", "done", "no more", "i'm done", "thanks, that's all",
    "that's it", "wrap up", "terminate", "abandon", "cease", "halt", "sign off"]
POSITIVE_EXITS = ['yes', 'please']
df = pd.read_csv('doggo_df.csv')
tf = TfChat()
response = tf.tf_idf_response('Exercise')


helper = True
user_response = input('Wuff, Wuff, I\'m here to answer any questions about Labradoodles, how can I help?').lower()
while helper:
    if user_response not in EXIT_KEYWORDS:
        print(tf.tf_idf_response(user_response))
        user_response = input('wuff, wuff, what else can do you want to know?').lower()
    else:
        user_response = input(' wuff, wuff hooman. Would you like to stop chatting?').lower()
        if user_response.lower() in EXIT_KEYWORDS or user_response.lower() in POSITIVE_EXITS:
            print('wuff, wuff, paw paw hooman :(')
            helper = False
