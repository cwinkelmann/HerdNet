# snippet to modify training data so it conforms with the data pipeline
import pandas as pd

import pandas

# class_dict = {
#     1: 'Hartebeest',
#     2: 'Buffalo',
#     3: 'Kob',
#     4: 'Warthog',
#     5: 'Waterbuck',
#     6: 'Elephant',
# }

class_dict = {
    1: 'iguana',
}

def _set_labels_species(cls_dict: dict, df: pandas.DataFrame) -> None:
    """
    Reverted Function from train _set_species_labels
    If labels is present but species is not
    """

    assert 'labels' in df.columns and 'species' not in df.columns, "Species column must not be present and labels column must be present"
    # cls_dict = dict(map(reversed, cls_dict.items()))
    df['species'] = df['labels'].map(cls_dict)

if __name__ == "__main__":

    # rename the columns

    # read the csv file
    # df_val_patches = pd.read_csv("../data_iguana/val_patches/gt.csv")
    #
    # _set_labels_species(class_dict, df_val_patches)
    #
    # df_val_patches.to_csv("../data_iguana/val_patches/gt.csv", index=False)


    df_train_patches = pd.read_csv("../data_iguana/train_patches/gt.csv")

    _set_labels_species(class_dict, df_train_patches)

    df_train_patches.to_csv("../data_iguana/train_patches/gt.csv", index=False)