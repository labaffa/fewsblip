from pathlib import Path
import pandas as pd
from models.blip import blip_decoder
from predict import Predictor
import os


#==================================================
#                                   Functions
#===================================================
def caption_image(path, p, df):
    #function that get as input the images folder and the predict function to output caption dataset and the errors during the process
    columns = ['id', 'media_filename', 'channel', "Main Caption", 'time']
    dataset = pd.DataFrame(columns=columns)
    r = 0
    errors = []
    for fold in os.listdir(path):
        if os.path.isdir(path + fold):
            for img in os.listdir(path + fold):
                print(f"""{img}""")
                f_img = f"""{fold}/{img}"""
                try:
                    caption, caption1 = p.predict(str(path + f_img),
                                                  task='image_captioning', question=None, caption=None)
                    # Attempt to assign values based on the prediction and data in df
                    media_filename = f_img.split('.png')[0]
                    dataset.at[r, 'media_filename'] = media_filename
                    if len(df[df['media_filename'] == media_filename]) > 0:
                        try:
                            dataset.at[r, "id"] = str(df[df['media_filename'] == media_filename]['id'].item())
                        except Exception as e:
                            errors.append(('Problem with id for image', media_filename, e))
                            dataset.at[r, "id"] = None
                        try:
                            dataset.at[r, "channel"] = df[df['media_filename'] == media_filename]['username'].item()
                        except Exception as e:
                            errors.append(('Problem with the channel for image', media_filename, e))
                            dataset.at[r, "channel"] = None
                        try:
                            dataset.at[r, "channel_id"] = str(
                                df[df['media_filename'] == media_filename]['channel_id'].item())
                        except Exception as e:
                            errors.append(('Problem with the channel_id for image', media_filename, e))
                            dataset.at[r, "channel_id"] = None
                        try:
                            dataset.at[r, "time"] = df[df['media_filename'] == media_filename]['timestamp'].item()
                        except Exception as e:
                            errors.append(('Problem with the time for image', media_filename, e))
                            dataset.at[r, "time"] = None
                    else:
                        dataset.at[r, "id"] = None
                        dataset.at[r, "channel"] = None
                        dataset.at[r, "time"] = None

                    dataset.at[r, "Main Caption"] = caption

                    for i, cap in enumerate(caption1):
                        dataset.at[r, f"Captions_{i}"] = cap
                except Exception as e:
                    errors.append(('Problem with the caption for image', media_filename, e))
                r += 1
                if r % 100 == 0:
                    print(r)
    return dataset, errors



def match_dataset(df_caption, df_original):
    """
    Merge caption data with original dataset using a unique identifier.
    """
    df_caption['unique_id'] = df_caption['id'].astype(str) + '_' + df_caption['channel'].astype(str)
    df_original['unique_id'] = df_original['id'].astype(str) + '_' + df_original['channel'].astype(str)
    return df_original.merge(df_caption, on='unique_id', how='left')


if __name__ == '__main__':
    df = pd.read_csv('/home/pcruciata/Togo_files/togo.tsv', sep='\t', header=0)#input file
    path = #TODO directory where the Telegram images are
    p = Predictor() #initialization of the class Predictor
    print(f'Image captioning starting')
    dataset, errors = caption_image(path, p, df) # function to execute the image captioning, that output dataset with the captioning and the error txt
    dataset.to_csv(f'/home/pcruciata/Togo_files/captions_Togo3.csv')#TODO #path to save the caption dataset file
    with open(f"/home/pcruciata/Togo_files/errorsTogo3.txt", "w") as output:# saving the error file as txt
        output.write(str(errors))
    df_final = match_dataset(df, dataset)# match caption with the original file
    df_final.to_csv(f'/home/pcruciata/Togo_files/Final_dataset_togo2.csv')#TODO #path to save the caption dataset file
